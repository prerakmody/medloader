import pdb
import time
import json
import itertools
import traceback
import numpy as np
from pathlib import Path

import tensorflow as tf
# tf.debugging.set_log_device_placement(True)

import medloader.dataloader.utils as utils
import medloader.dataloader.config as config 

class HaNMICCAI2015Dataset:
    """
    The 2015 MICCAI Challenge contains CT scans of the head and neck along with annotations for 9 organs.
    It contains train, train_additional, test_onsite and test_offsite folders

    Dataset link: http://www.imagenglab.com/wiki/mediawiki/index.php?title=2015_MICCAI_Challenge
    """

    def __init__(self, data_dir, dir_type
                    , dimension=3, grid=True, resampled=False, mask_type=config.MASK_TYPE_ONEHOT
                    , transforms=[], filter_grid=False, random_grid=False
                    , parallel_calls=None, deterministic=False
                    , patient_shuffle=False
                    , debug=False, single_sample=False):

        self.name = '{}_MICCAI2015'.format(config.HEAD_AND_NECK)

        # Params - Source 
        self.data_dir = data_dir
        self.dir_type = dir_type

        # Params - Spatial (x,y)
        self.dimension = dimension
        self.grid=grid
        self.resampled=resampled
        self.mask_type = mask_type

        # Params - Transforms/Filters
        self.transforms = transforms
        self.filter_grid = filter_grid
        self.random_grid = random_grid

        # Params - Memory related
        self.patient_shuffle = patient_shuffle
        
        # Params - TFlow Dataloader related
        self.parallel_calls = parallel_calls # [1, tf.data.experimental.AUTOTUNE]
        self.deterministic = deterministic

        # Params - Debug
        self.debug = debug
        self.single_sample = single_sample

        # Data items
        self.data = {}
        self.paths_img = []
        self.paths_mask = []
        self.cache = {}
        self.filter = None

        # Function calls
        self._download()
        self._init_data()

    def __len__(self):
        if self.grid:
            if self.filter is None and self.filter_grid is False:
                return 200*len(self.paths_img) # i.e. approx 200 grids per volume
            else:
                sampler_perc_data = 1.0 - getattr(config, self.name)['GRID_3D']['SAMPLER_PERC'] + 0.1
                return int(200*len(self.paths_img)*sampler_perc_data)
        else:
            return len(self.paths_img)

    def _download(self):
        self.dataset_dir = Path(self.data_dir).joinpath(self.name)
        self.dataset_dir_raw = Path(self.dataset_dir).joinpath(config.DIRNAME_RAW)
        self.dataset_dir_processed = Path(self.dataset_dir).joinpath(config.DIRNAME_PROCESSED)
        
        self.dataset_dir_datatypes = ['train', 'train_additional', 'test_offsite', 'test_onsite']
        self.dataset_dir_datatypes_ranges = [328+1,479+1,746+1,878+1]
        self.dataset_dir_processed_train_2D = Path(self.dataset_dir_processed).joinpath('train', config.DIRNAME_SAVE_2D)
        self.dataset_dir_processed_train_3D = Path(self.dataset_dir_processed).joinpath('train', config.DIRNAME_SAVE_3D)
        
        if not Path(self.dataset_dir_raw).exists() or not Path(self.dataset_dir_processed).exists():
            print ('')
            print (' ------------------ HaNMICCAI2015 Dataset ------------------')

        if not Path(self.dataset_dir_raw).exists():
            print ('')
            print (' ------------------ Download Data ------------------')
            from medloader.dataloader.extractors.han_miccai2015 import HaNMICCAI2015Downloader
            downloader = HaNMICCAI2015Downloader(self.dataset_dir_raw, self.dataset_dir_processed)
            downloader.download()
            downloader.sort(self.dataset_dir_datatypes, self.dataset_dir_datatypes_ranges)
            
        if not Path(self.dataset_dir_processed_train_2D).exists() or not Path(self.dataset_dir_processed_train_3D).exists():
            print ('')
            print (' ------------------ Process Data ------------------')
            from medloader.dataloader.extractors.han_miccai2015 import HaNMICCAI2015Extractor
            extractor = HaNMICCAI2015Extractor(self.name, self.dataset_dir_raw, self.dataset_dir_processed, self.dataset_dir_datatypes)
            if not Path(self.dataset_dir_processed_train_3D).exists():
                extractor.extract3D()
            if not Path(self.dataset_dir_processed_train_2D).exists():
                extractor.extract2D()
            print ('')
            print (' --------------------------------------------------')
            print ('')
    
    def _init_data(self):

        self.DIR_TYPES_TEST = ['test_offsite', 'test_onsite']
        
        self.patient_meta_info = {}
        self.path_img_csv = ''
        self.path_mask_csv = ''

        # Step 1 - Define global paths
        self.data_dir_processed = Path(self.data_dir).joinpath(self.name, config.DIRNAME_PROCESSED, self.dir_type)
        self.data_dir_processed_2D = Path(self.data_dir_processed).joinpath(config.DIRNAME_SAVE_2D)
        self.data_dir_processed_3D = Path(self.data_dir_processed).joinpath(config.DIRNAME_SAVE_3D)
        
        # Step 2 - Get paths for 2D/3D
        if self.dimension == 2:
            if self.resampled is False:
                self.path_img_csv = Path(self.data_dir_processed_2D).joinpath(config.FILENAME_CSV_IMG)
                self.path_mask_csv = Path(self.data_dir_processed_2D).joinpath(config.FILENAME_CSV_MASK)
            else:
                self.path_img_csv = Path(self.data_dir_processed_2D).joinpath(config.FILENAME_CSV_IMG_RESAMPLED)
                self.path_mask_csv = Path(self.data_dir_processed_2D).joinpath(config.FILENAME_CSV_MASK_RESAMPLED)
        elif self.dimension == 3:
            if self.resampled is False:
                self.path_img_csv = Path(self.data_dir_processed_3D).joinpath(config.FILENAME_CSV_IMG)
                self.path_mask_csv = Path(self.data_dir_processed_3D).joinpath(config.FILENAME_CSV_MASK)
            else:
                self.path_img_csv = Path(self.data_dir_processed_3D).joinpath(config.FILENAME_CSV_IMG_RESAMPLED)
                self.path_mask_csv = Path(self.data_dir_processed_3D).joinpath(config.FILENAME_CSV_MASK_RESAMPLED)

        # Step 3 - Get patient meta info
        if self.resampled is False:
            print (' - [Warning][HaNMICCAI2015Dataset]: This datalader is not extracting 3D Volumes which have been resampled to the same 3D voxel spacing')
        for patient_dir in Path(self.data_dir_processed_3D).iterdir():
            if config.FILE_EXTENSION_CSV not in Path(patient_dir).parts[-1]:
                for patient_file in Path(patient_dir).iterdir():
                    if config.FILE_EXTENSION_JSON in Path(patient_file).parts[-1]:
                        with open(patient_file, 'r') as fp:
                            patient_id = Path(patient_dir).parts[-1]

                            brainstem_idxs = None
                            if self.resampled is False:
                                brainstem_idxs = json.load(fp)[config.TYPE_VOXEL_ORIGSHAPE][config.KEYNAME_MEAN_BRAINSTEAM]
                            else:
                                try:
                                    brainstem_idxs = json.load(fp)[config.TYPE_VOXEL_RESAMPLED][config.KEYNAME_MEAN_BRAINSTEAM]
                                except:
                                    print (' - [ERROR][HaNMICCAI2015Dataset] There is no resampled data and you have set resample=True')
                                    print ('  -- Delete the data/HaNMICCAI2015Dataset/processed directory and set config.VOXEL_RESO to a tuple of pixel spacing values for x,y,z axes')
                                    print ('  -- Exiting now! ')
                                    import sys; sys.exit(1)
                            mean_brainstem_idx = np.array(brainstem_idxs).astype(config.DATATYPE_VOXEL_IMG).tolist()
                            self.patient_meta_info[patient_id] = mean_brainstem_idx    

        # Step 4 - Get file paths
        if Path(self.path_img_csv).exists() and Path(self.path_mask_csv).exists():
            self.paths_img = utils.read_csv(self.path_img_csv)
            self.paths_mask = utils.read_csv(self.path_mask_csv)
        else:
            print (' - [ERROR] Issue with path')
            print (' -- [ERROR] self.path_img_csv : ({}) {}'.format(Path(self.path_img_csv).exists(), self.path_img_csv ))
            print (' -- [ERROR] self.path_mask_csv: ({}) {}'.format(Path(self.path_mask_csv).exists(), self.path_mask_csv ))
            pdb.set_trace()
        
        # Step 5 - Meta for labels
        self.LABEL_MAP = getattr(config, self.name)['LABEL_MAP']
        self.LABELID_BACKGROUND = getattr(config, self.name)['LABELID_BACKGROUND']

        # Step 6 - Meta for grid sampling
        if self.dimension == 3 and self.grid:
            if self.grid:
                grid_3D_params = getattr(config, self.name)['GRID_3D']
                self.grid_size = grid_3D_params['SIZE']
                self.grid_overlap = grid_3D_params['OVERLAP']
                self.SAMPLER_PERC = grid_3D_params['SAMPLER_PERC']
                self.RANDOM_SHIFT_MAX = grid_3D_params['RANDOM_SHIFT_MAX']
                self.RANDOM_SHIFT_PERC = grid_3D_params['RANDOM_SHIFT_PERC']

                self.w_grid, self.h_grid, self.d_grid = self.grid_size
                self.w_overlap, self.h_overlap, self.d_overlap = self.grid_overlap
            else:
                self.DIMS_3D = getattr(config, self.name)['DIMS']['3D_SAMPLER']
                        
    def get_voxel_stats(self, show=False):

        spacing_x = []
        spacing_y = []
        spacing_z = []

        info_img_path = Path(self.data_dir).joinpath(self.name, config.DIRNAME_PROCESSED, self.dir_type, config.DIRNAME_SAVE_2D, config.FILENAME_VOXEL_INFO)

        if Path(info_img_path).exists():
            import json
            with open(str(info_img_path), 'r') as fp:
                data = json.load(fp)
                for patient_id in data:
                    spacing_info = data[patient_id][config.TYPE_VOXEL_ORIGSHAPE]['spacing']
                    spacing_x.append(spacing_info[0])
                    spacing_y.append(spacing_info[1])
                    spacing_z.append(spacing_info[2])
            
            if show:
                if len(spacing_x) and len(spacing_y) and len(spacing_z):
                    import matplotlib.pyplot as plt
                    f,axarr = plt.subplots(1,3)
                    axarr[0].hist(spacing_x); axarr[0].set_title('Voxel Spacing (X)')
                    axarr[1].hist(spacing_y); axarr[1].set_title('Voxel Spacing (Y)')
                    axarr[2].hist(spacing_z); axarr[2].set_title('Voxel Spacing (Z)')
                    plt.suptitle(self.name)
                    plt.show()
            
        else:
            print (' - [ERROR][get_voxel_stats()] Path issue: info_img_path: ', info_img_path)
        
        return spacing_x, spacing_y, spacing_z

    def generator(self):

        try:
            
            if len(self.paths_img) and len(self.paths_mask):
                
                # Step 1 - Create basic generator
                dataset = None
                if self.dimension == 2:
                    # The code crashes if we dont include the comma in the “args” tuple
                    dataset = tf.data.Dataset.from_generator(self._generator2D
                        , output_types=(config.DATATYPE_TF_FLOAT32, config.DATATYPE_TF_FLOAT32, config.DATATYPE_TF_INT32, tf.string)
                        ,args=())
                elif self.dimension == 3:
                    dataset = tf.data.Dataset.from_generator(self._generator3D
                        , output_types=(config.DATATYPE_TF_INT16, config.DATATYPE_TF_UINT8, config.DATATYPE_TF_INT32, tf.string)
                        ,args=())
                
                # Step 2 - Get 3D data
                if self.dimension == 3:
                    dataset = dataset.map(self._get_data_3D , num_parallel_calls=self.parallel_calls , deterministic=self.deterministic)
                    
                # Step 3 - Filter function
                if self.filter_grid:
                    dataset = dataset.filter(self.filter.execute)

                # Step 4 - Data augmentations
                if len(self.transforms):
                    for transform in self.transforms:
                        try:
                            dataset = dataset.map(transform.execute, num_parallel_calls=self.parallel_calls, deterministic=self.deterministic)
                        except:
                            traceback.print_exc()
                            print (' - [ERROR] Issue with transform: ', transform.name)
                else:
                    print 
                    print ('')
                    print (' - [INFO][HaNMICCAI2015Dataset] No transformations available!', self.dir_type)
                    print ('')
                
                # Step 6 - Return
                return dataset
            
            else:
                return None

        except:
            traceback.print_exc()
            pdb.set_trace()
            return None

    def _get_paths(self, idx):
        patient_id = ''
        study_id = ''
        path_img, path_mask = '', ''

        if self.debug:
            path_img = Path(self.paths_img[0]).absolute()
            path_mask = Path(self.paths_mask[0]).absolute()
            path_img, path_mask = self.path_debug_3D(path_img, path_mask)
        else:
            path_img = Path(self.paths_img[idx]).absolute()
            path_mask = Path(self.paths_mask[idx]).absolute()

        if path_img.exists() and path_mask.exists():
            patient_id = Path(path_img).parts[-2]
            study_id = Path(path_img).parts[-4]
        else:
            print (' - [ERROR] Issue with path')
            print (' -- [ERROR] path_img : ', path_img)
            print (' -- [ERROR] path_mask: ', path_mask)
        
        return path_img, path_mask, patient_id, study_id

    def _generator3D(self):

        try:
            
            # Step 0 - Init
            res = []

            # Step 1 - Get patient idxs
            idxs = np.arange(len(self.paths_img)).tolist()  #[:3]
            if self.single_sample: idxs = idxs[2:4]
            if self.patient_shuffle: np.random.shuffle(idxs)

            # Step 2 - Proceed on the basis of grid sampling or full-volume (self.grid=False) sampling
            if self.grid:
                
                # Step 2.1 - Get grid sampler info for each patient-idx
                sampler_info = {}
                for idx in idxs:
                    path_img = Path(self.paths_img[idx]).absolute() 
                    path_img_parts = list(path_img.parts)
                    path_img_parts[-1] = config.FILENAME_VOXEL_INFO
                    path_json = Path(*path_img_parts)
                    with open(path_json) as fp:
                        voxelinfo = json.load(fp)
                        if config.TYPE_VOXEL_RESAMPLED in str(path_img):
                            voxel_shape = voxelinfo[config.TYPE_VOXEL_RESAMPLED][config.KEYNAME_SHAPE]    
                        else:
                            voxel_shape = voxelinfo[config.TYPE_VOXEL_ORIGSHAPE][config.KEYNAME_SHAPE]

                        grid_idxs_width = utils.split_into_overlapping_grids(voxel_shape[0], len_grid=self.grid_size[0], len_overlap=self.grid_overlap[0])
                        grid_idxs_height = utils.split_into_overlapping_grids(voxel_shape[1], len_grid=self.grid_size[1], len_overlap=self.grid_overlap[1])   
                        grid_idxs_depth = utils.split_into_overlapping_grids(voxel_shape[2], len_grid=self.grid_size[2], len_overlap=self.grid_overlap[2])
                        sampler_info[idx] = list(itertools.product(grid_idxs_width,grid_idxs_height,grid_idxs_depth))
                        
                # Step 2.2 - Loop over all patients and their grids
                # Note - Grids of a patient are extracted in order
                for i, idx in enumerate(idxs):
                    path_img, path_mask, patient_id, study_id = self._get_paths(idx)
                    if path_img.exists() and path_mask.exists():
                        for sample_info in sampler_info[idx]:
                            grid_idxs = sample_info
                            meta1 = [idx] + [grid_idxs[0][0], grid_idxs[1][0], grid_idxs[2][0]] # only include w_start, h_start, d_start
                            meta2 ='-'.join([self.name, study_id, patient_id])
                            path_img = str(path_img)
                            path_mask = str(path_mask)
                            res.append((path_img, path_mask, meta1, meta2))
            
            else:
                for i, idx in enumerate(idxs):
                    path_img, path_mask, patient_id, study_id = self._get_paths(idx)
                    if path_img.exists() and path_mask.exists():
                        meta1 = [idx] + [0,0,0] # dummy for w_start, h_start, d_start
                        meta2 ='-'.join([self.name, study_id, patient_id])
                        path_img = str(path_img)
                        path_mask = str(path_mask)
                        res.append((path_img, path_mask, meta1, meta2))

            # Step 3 - Yield
            for each in res:
                path_img, path_mask, meta1, meta2 = each
                # print (' ------------- path_img: ', Path(path_img).parts[-2])

                vol_img_npy, vol_mask_npy, spacing = self._get_cache_item_old(path_img, path_mask)
                if vol_img_npy is None and vol_mask_npy is None:
                    vol_img_npy, vol_mask_npy, spacing = self._get_volume_from_path(path_img, path_mask)    
                    self._set_cache_item_old(path_img, path_mask, vol_img_npy, vol_mask_npy, spacing)
                
                spacing = tf.constant(spacing, dtype=tf.int32)
                vol_img_npy_shape = tf.constant(vol_img_npy.shape, dtype=tf.int32)
                meta1 = tf.concat([meta1, spacing, vol_img_npy_shape], axis=0)

                yield (vol_img_npy, vol_mask_npy, meta1, meta2)

            # patient_id_global = None
            # vol_img_npy, vol_mask_npy, spacing = None, None, None
            # for each in res:
            #     path_img, path_mask, meta1, meta2 = each
            #     patient_id_running = Path(path_img).parts[-2]

            #     if patient_id_running != patient_id_global:
            #         print (' - patient_id_running, patient_id_global: ', patient_id_running, patient_id_global)
            #         vol_img_npy, vol_mask_npy, spacing = self._get_volume_from_path(path_img, path_mask)
            #         patient_id_global = patient_id_running
                
            #     spacing = tf.constant(spacing, dtype=tf.int32)
            #     vol_img_npy_shape = tf.constant(vol_img_npy.shape, dtype=tf.int32)
            #     meta1 = tf.concat([meta1, spacing, vol_img_npy_shape], axis=0)

            #     yield (vol_img_npy, vol_mask_npy, meta1, meta2)

        except:
            traceback.print_exc()
            pdb.set_trace()
            yield ('','',[],'')

    def _get_cache_item_old(self, path_img, path_mask):
        if 'img' in self.cache and 'mask' in self.cache:
            if path_img in self.cache['img'] and path_mask in self.cache['mask']:
                # print (' - [_get_cache_item()] ')
                return self.cache['img'][path_img], self.cache['mask'][path_mask], self.cache['spacing']
            else:
                return None, None, None
        else:
            return None, None, None
    
    def _set_cache_item_old(self, path_img, path_mask, vol_img, vol_mask, spacing):
        # print (' - [_set_cache_item() ]: ', vol_img.shape, vol_mask.shape)
        self.cache = {
            'img': {path_img: vol_img}
            , 'mask': {path_mask: vol_mask}
            , 'spacing': spacing
        }
    
    def _set_cache_item(self, path_img, path_mask, vol_img, vol_mask, spacing):
        if len(self.cache) == 0:
            self.cache = {path_img: [vol_img, vol_mask, spacing]}
            self.cache_id = {path_img:0}
        elif len(self.cache) == 1:
            self.cache[path_img] = [vol_img, vol_mask, spacing]
            self.cache_id[path_img] = 1
        elif len(self.cache) == 2:
            max_order_id = max(self.cache_id.values())
            for path_img_ in self.cache_id:
                if self.cache_id[path_img_] == max_order_id - 1:
                    self.cache.pop(path_img_)
            self.cache[path_img] = [vol_img, vol_mask, spacing]
            self.cache_id[path_img] = max_order_id+1
    
    def _get_cache_item(self, path_img, path_mask):
        # print (' - self.cache.keys(): ', self.cache.keys())
        if path_img in self.cache:
            return self.cache[path_img]
        else:
            return None, None, None
    
    def _get_volume_from_path(self, path_img, path_mask, verbose=False):
        if verbose: t0 = time.time()
        vol_img_sitk = utils.read_mha(path_img)
        vol_img_npy = utils.sitk_to_array(vol_img_sitk)

        vol_mask_sitk = utils.read_mha(path_mask)
        vol_mask_npy = utils.sitk_to_array(vol_mask_sitk)     

        spacing = np.array(vol_img_sitk.GetSpacing())
        if verbose: print (' - [HaNMICCAI2015Dataset._get_volume_from_path()] Time: ({}):{}s'.format(Path(path_img).parts[-2],  round(time.time() - t0,2)))
        
        # Send to GPU
        return tf.cast(vol_img_npy, dtype=config.DATATYPE_TF_INT16), tf.cast(vol_mask_npy, dtype=config.DATATYPE_TF_UINT8), tf.constant(spacing*100, dtype=config.DATATYPE_TF_INT32)
        
        # Keep on CPU
        # return vol_img_npy.astype(np.int16), vol_mask_npy.astype(np.uint8), (spacing*100).astype(np.int32)

    @tf.function
    def _get_new_grid_idx(self, start, end, max):
        
        print (' - [HaNMICCAI2015Dataset][_get_new_grid_idx()] dir_type: ', self.dir_type)
        start_prev = start
        if tf.random.uniform([], minval=0, maxval=1, dtype=tf.dtypes.float32) <= self.RANDOM_SHIFT_PERC:
            
            delta_left = start
            delta_right = max - end
            
            shift_voxels =  tf.random.uniform([], minval=0, maxval=self.RANDOM_SHIFT_MAX, dtype=tf.dtypes.int32)

            if delta_left > self.RANDOM_SHIFT_MAX and delta_right > self.RANDOM_SHIFT_MAX:
                if tf.random.uniform([], minval=0, maxval=1, dtype=tf.dtypes.float32) <= self.RANDOM_SHIFT_PERC:
                    start = start - shift_voxels
                    end = end - shift_voxels
                else:
                    start = start + shift_voxels
                    end = end + shift_voxels
            elif delta_left > self.RANDOM_SHIFT_MAX and delta_right <= self.RANDOM_SHIFT_MAX:
                start = start - shift_voxels
                end = end - shift_voxels
            elif delta_left <= self.RANDOM_SHIFT_MAX and delta_right > self.RANDOM_SHIFT_MAX:
                start = start + shift_voxels
                end = end + shift_voxels

        return start_prev, start, end

    @tf.function
    def _get_data_3D(self, vol_img, vol_mask, meta1, meta2):

            vol_img_npy = None
            vol_mask_npy = None

            # Step 1 - Proceed on the basis of grid sampling or full-volume (self.grid=False) sampling
            if self.grid:
                
                # Step 1.1 - Get raw images/masks and extract grid
                w_start = meta1[1]
                w_end   = w_start + self.grid_size[0]
                h_start = meta1[2]
                h_end   = h_start + self.grid_size[1]
                d_start = meta1[3]
                d_end   = d_start + self.grid_size[2]

                # Step 1.2 - Randomization of grid 
                if self.random_grid:
                    w_max = meta1[7]
                    h_max = meta1[8]
                    d_max = meta1[9]

                    w_start_prev = w_start
                    d_start_prev = d_start
                    w_start_prev, w_start, w_end = self._get_new_grid_idx(w_start, w_end, w_max)
                    h_start_prev, h_start, h_end = self._get_new_grid_idx(h_start, h_end, h_max)
                    d_start_prev, d_start, d_end = self._get_new_grid_idx(d_start, d_end, d_max)

                    meta1_diff = tf.convert_to_tensor([0,w_start - w_start_prev, h_start - h_start_prev, d_start - d_start_prev,0,0,0,0,0,0])
                    meta1 = meta1 + meta1_diff
                    
                # Step 1.3 - Extracting grid
                vol_img_npy = vol_img[w_start:w_end, h_start:h_end, d_start:d_end]
                vol_mask_npy = vol_mask[w_start:w_end, h_start:h_end, d_start:d_end]

            # Step 2 - One-hot or not
            vol_mask_classes = []
            label_ids_mask = []
            label_ids = sorted(list(self.LABEL_MAP.values()))
            if self.mask_type == config.MASK_TYPE_ONEHOT:
                vol_mask_classes = tf.concat([tf.expand_dims(tf.math.equal(vol_mask_npy, label), axis=-1) for label in label_ids], axis=-1) # [H,W,D,L]
                label_ids_mask = tf.cast(tf.reduce_any(vol_mask_classes, axis=[0,1,2]), dtype=tf.int32)
                # tf.print(' - label_ids_mask: ', label_ids_mask.numpy())

            elif self.mask_type == config.MASK_TYPE_COMBINED:
                vol_mask_classes = vol_mask_npy
                unique_classes = np.unique(vol_mask_npy).astype(np.uint8)
                for label_id in label_ids:
                    if label_id in unique_classes: label_ids_mask.append(1)
                    else: label_ids_mask.append(0)

            # Step 3 - Dtype conversion and expading dimensions            
            if self.mask_type == config.MASK_TYPE_ONEHOT:
                x = tf.cast(tf.expand_dims(vol_img_npy, axis=3), dtype=tf.float32) # [H,W,D,1]
            else:
                x = tf.cast(vol_img_npy, dtype=tf.float32) # [H,W,D]

            y = tf.cast(vol_mask_classes, dtype=tf.float32) # [H,W,D,L]

            # Step 4 - Append info to meta1
            # label_ids_mask = tf.constant(label_ids_mask, dtype=tf.int32)
            meta1 = tf.concat([meta1, label_ids_mask], axis=0)

            # Step 5 - return
            return (x,y,meta1, meta2)
        
    def path_debug_3D(self, path_img, path_mask):
        patient_number = '0522c0251'
        path_img_parts = list(Path(path_img).parts)

        path_img_parts = list(Path(path_img).parts)
        path_img_parts[-2] = patient_number
        path_img = Path(*path_img_parts)

        path_mask_parts = list(Path(path_mask).parts)
        path_mask_parts[-2] = patient_number
        path_mask = Path(*path_mask_parts)

        if 0:
            utils.print_debug_header()
            print (' - path_img : ', path_img.parts[-2:])
            print (' - path_mask: ', path_mask.parts[-2:])

        self.paths_img = [path_img]
        self.paths_mask = [path_mask] 

        return path_img, path_mask 