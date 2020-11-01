import os
import pdb
import time
import json
import traceback
import numpy as np
from pathlib import Path

import tensorflow as tf
if len(tf.config.list_physical_devices('GPU')):
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

import medloader.dataloader.utils as utils
import medloader.dataloader.config as config 
import medloader.dataloader.tensorflow.augmentations as aug

class HaNMICCAI2015Dataset:
    """
    The 2015 MICCAI Challenge contains CT scans of the head and neck along with annotations for 9 organs.
    It contains train, train_additional, test_onsite and test_offsite folders

    Dataset link: http://www.imagenglab.com/wiki/mediawiki/index.php?title=2015_MICCAI_Challenge
    """

    def __init__(self, data_dir, dir_type, dimension=2, mask_type='one_hot', transforms=[]
                    , resampled=False, in_memory=False, debug=False):

        self.name = '{}_MICCAI2015'.format(config.HEAD_AND_NECK)

        # Params
        self.data_dir = data_dir
        self.dir_type = dir_type
        self.dimension = dimension
        self.mask_type = mask_type
        self.transforms = transforms
        self.resampled = resampled
        self.debug = debug
        self.in_memory = in_memory
        
        # Data items
        self.data = {}
        self.paths_img = []
        self.paths_mask = []

        # Function calls
        self._download()
        self._init_data()

    def __len__(self):
        return len(self.paths_img)*self.total_grids

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
        for patient_dir in Path(self.data_dir_processed_3D).iterdir():
            if config.FILE_EXTENSION_CSV not in Path(patient_dir).parts[-1]:
                for patient_file in Path(patient_dir).iterdir():
                    if config.FILE_EXTENSION_JSON in Path(patient_file).parts[-1]:
                        with open(patient_file, 'r') as fp:
                            patient_id = Path(patient_dir).parts[-1]
                            if self.resampled is False:
                                brainstem_idxs = json.load(fp)[config.TYPE_VOXEL_ORIGSHAPE][config.KEYNAME_MEAN_BRAINSTEAM]
                            else:
                                brainstem_idxs = json.load(fp)[config.TYPE_VOXEL_RESAMPLED][config.KEYNAME_MEAN_BRAINSTEAM]

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
        
        # Step 5 - Other meta
        self.LABEL_MAP = getattr(config, self.name)['LABEL_MAP']
        self.LABELID_BACKGROUND = getattr(config, self.name)['LABELID_BACKGROUND']
        self.DIMS_3D = getattr(config, self.name)['DIMS']['3D_SAMPLER']

        self.w_total = self.DIMS_3D['MIDPOINT_EXTENSION_W_LEFT'] + self.DIMS_3D['MIDPOINT_EXTENSION_W_RIGHT']
        self.h_total = self.DIMS_3D['MIDPOINT_EXTENSION_H_BACK'] + self.DIMS_3D['MIDPOINT_EXTENSION_H_FRONT']
        self.d_total = self.DIMS_3D['MIDPOINT_EXTENSION_D_TOP']  + self.DIMS_3D['MIDPOINT_EXTENSION_D_BOTTOM']

        self.w_grid = (self.w_total + ((self.DIMS_3D['GRIDS_SAMPLE_W']-1)*self.DIMS_3D['GRIDS_OVERLAP_W']))//self.DIMS_3D['GRIDS_SAMPLE_W']
        self.h_grid = (self.h_total + ((self.DIMS_3D['GRIDS_SAMPLE_H']-1)*self.DIMS_3D['GRIDS_OVERLAP_H']))//self.DIMS_3D['GRIDS_SAMPLE_H']
        self.d_grid = (self.d_total + ((self.DIMS_3D['GRIDS_SAMPLE_D']-1)*self.DIMS_3D['GRIDS_OVERLAP_D']))//self.DIMS_3D['GRIDS_SAMPLE_D']

        assert self.w_grid % 16 == 0, " - [ERROR][HaNMICCAI2015Dataset] w_grid should be a multiple of 16"
        assert self.h_grid % 16 == 0, " - [ERROR][HaNMICCAI2015Dataset] h_grid should be a multiple of 16"
        assert self.d_grid % 16 == 0, " - [ERROR][HaNMICCAI2015Dataset] d_grid should be a multiple of 16"

        self.grid_start_idxs_w = [grid_id*self.w_grid - grid_id*self.DIMS_3D['GRIDS_OVERLAP_W'] for grid_id in range(self.DIMS_3D['GRIDS_SAMPLE_W'])]
        self.grid_start_idxs_h = [grid_id*self.h_grid - grid_id*self.DIMS_3D['GRIDS_OVERLAP_H'] for grid_id in range(self.DIMS_3D['GRIDS_SAMPLE_H'])]
        self.grid_start_idxs_d = [grid_id*self.d_grid - grid_id*self.DIMS_3D['GRIDS_OVERLAP_D'] for grid_id in range(self.DIMS_3D['GRIDS_SAMPLE_D'])]

        import itertools
        self.grid_start_idxs = list(itertools.product(*[self.grid_start_idxs_w, self.grid_start_idxs_h, self.grid_start_idxs_d]))
        self.total_grids = len(self.grid_start_idxs)
                        
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
                        , output_types=(tf.string, tf.string, config.DATATYPE_TF_INT32, tf.string)
                        ,args=())

                # step 2 - Give the generator its own thread
                dataset = dataset.map(lambda x,y,meta1,meta2: (x,y,meta1,meta2), num_parallel_calls=tf.data.experimental.AUTOTUNE) # dataset gets its own thread
                
                if self.dimension == 3:
                    dataset = dataset.map(
                            lambda path_img,path_mask,meta1,meta2: tf.py_function(func=self.get_data_3D
                                                                                    , inp=[path_img,path_mask, meta1, meta2]
                                                                                    , Tout=[config.DATATYPE_TF_FLOAT32, config.DATATYPE_TF_FLOAT32, config.DATATYPE_TF_INT32, tf.string]
                                                                                )
                                    , num_parallel_calls=tf.data.experimental.AUTOTUNE
                    )
                    
                    # if self.mask_type == config.MASK_TYPE_ONEHOT:
                    #     if 1:
                    #         dir_cache = Path(self.dataset_dir).joinpath(config.DIRNAME_PROCESSED, self.dir_type, config.DIRNAME_SAVE_3D, config.DIRNAMENAME_CACHED_TF)
                    #         dir_cache.mkdir(parents=True, exist_ok=True)
                    #         filepath_cache = Path(dir_cache).joinpath(config.DIRNAMENAME_CACHED_TF)
                    #         dataset = dataset.cache(str(filepath_cache))
                    #         dataset = dataset.shuffle(self.__len__(), reshuffle_each_iteration=True)
                    #     else:
                    #         dataset = dataset.cache()
                    #         dataset = dataset.shuffle(self.__len__(), reshuffle_each_iteration=True)
                
                # Step 3 - Data augmentations
                if len(self.transforms):
                    for transform in self.transforms:
                        try:
                            dataset = dataset.map(transform.execute, num_parallel_calls=tf.data.experimental.AUTOTUNE)

                        except:
                            traceback.print_exc()
                            print (' - [ERROR] Issue with transform: ', transform.name)
                else:
                    print ('')
                    print (' - [INFO][HaNMICCAI2015Dataset] No transformations available!')
                    print ('')
                
                # Step 4 - Other stuff
                return dataset
            
            else:
                return None

        except:
            traceback.print_exc()
            pdb.set_trace()
            return None

    def _generator3D(self):
        try:

            # Step 1.1 - Get idxs (shuffled)
            idxs = np.arange(len(self.paths_img))
            
            # Step 1.2 - Get grid sampler info for each idx
            sampler_info = {}
            if self.dir_type in self.DIR_TYPES_TEST:
                for idx in idxs:
                    sampler_info[idx] = list(self.grid_start_idxs)
            else:
                np.random.shuffle(idxs)
                for idx in idxs:
                    np.random.shuffle(self.grid_start_idxs)
                    if np.random.random() < 0.000001:
                        sampler_info[idx] = {
                            'sampler_info_w' : np.random.choice(self.grid_w_midpoint, config.SAMPLER_GRIDS, replace=False).tolist()
                            , 'sampler_info_h' : np.random.choice(self.grid_h_midpoint, config.SAMPLER_GRIDS, replace=False).tolist()
                        }
                    else:
                        sampler_info[idx] = list(self.grid_start_idxs)   

            # Step 2 - Loop over all patients and their grids
            for _ in range(self.total_grids):
                for _,idx in enumerate(idxs):
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
                        yield ('','',[],'')
                    
                    overlap_info = np.array([self.DIMS_3D['GRIDS_OVERLAP_W'], self.DIMS_3D['GRIDS_OVERLAP_H'], self.DIMS_3D['GRIDS_OVERLAP_D']]).astype(np.int32).tolist()
                    midpoint_info = np.array(self.patient_meta_info[patient_id]).astype(np.int32).tolist()
                    meta1 = [idx] + midpoint_info + list(sampler_info[idx].pop())
                    meta2 ='-'.join([self.name, study_id, patient_id])
                    path_img = str(path_img)
                    path_mask = str(path_mask)
                    
                    yield (path_img, path_mask, meta1, meta2)

        except:
            traceback.print_exc()
            pdb.set_trace()
            yield ('','',[],'')

    def _generator2D(self):
        try:
            idxs = np.arange(len(self.paths_img))
            np.random.shuffle(idxs)
            
            LABEL_MAP = getattr(config, self.name)['LABEL_MAP']
            for _,idx in enumerate(idxs):
                
                slice_img = []
                slice_mask_classes = []
                patient_id = ''
                study_id = ''
                slice_id = ''

                # Step 1 - Check if in memory
                if idx not in self.data:

                    path_img, path_mask = '', ''
                    if self.debug:
                        path_img = Path(self.paths_img[0]).absolute()
                        path_mask = Path(self.paths_mask[0]).absolute()
                        path_img, path_mask = self.path_debug_2D(path_img, path_mask)
                    else:
                        path_img = Path(self.paths_img[idx]).absolute()
                        path_mask = Path(self.paths_mask[idx]).absolute()

                    if Path(path_img).exists() and Path(path_mask).exists():
                        
                        patient_id = Path(path_img).parts[-3]
                        study_id = Path(path_img).parts[-5]
                        slice_id = Path(path_img).parts[-1].split('_')[1]

                        # Step 2.1 - Get img masks
                        slice_img = np.load(path_img)
                        slice_mask = np.load(path_mask)

                        # Step 3 - One-hot or not
                        if self.mask_type == config.MASK_TYPE_ONEHOT:
                            label_ids = list(LABEL_MAP.values())
                            slice_mask_classes = np.zeros((slice_img.shape[0], slice_img.shape[1], len(label_ids)))
                            for label_id in label_ids:
                                label_idxs = np.argwhere(slice_mask == label_id)
                                slice_mask_classes[label_idxs[:,0], label_idxs[:,1], label_id] = 1
                        else:
                            slice_mask_classes = slice_mask

                        if self.in_memory:
                            self.data[idx] = {'slice_img': slice_img, 'slice_mask_classes':slice_mask_classes}
                    
                    else:
                        print (' - [ERROR] Issue with path')
                        print (' -- [ERROR] path_img : ({}) {}'.format(Path(path_img).exists(), path_img ))
                        print (' -- [ERROR] path_mask: ({}) {}'.format(Path(path_mask).exists(), path_mask ))
                
                else:
                    slice_img = self.data[idx]['slice_img']
                    slice_mask_classes = self.data[idx]['slice_mask_classes']

                
                crop_info = self.patient_meta_info[patient_id]
                meta1 = [idx, crop_info[0], crop_info[1]]
                meta2 = '-'.join([self.name, study_id, patient_id, slice_id])
                
                yield (np.expand_dims(slice_img, axis=2), slice_mask_classes, meta1, meta2)
        
        except:
            traceback.print_exc()
            pdb.set_trace()

    def get_data_3D(self, path_img, path_mask, meta1, meta2):

        path_img = path_img.numpy().decode('utf-8')
        path_mask = path_mask.numpy().decode('utf-8')
        
        if Path(path_img).exists() and Path(path_mask).exists():

            # Step 1.1 - Get raw images/masks and crop (to save on memory)
            midpoint_info = meta1[1:4]
            midpoint_info = meta1[1:4]
            w_start = midpoint_info[0] - self.DIMS_3D['MIDPOINT_EXTENSION_W_LEFT']
            w_end   = midpoint_info[0] + self.DIMS_3D['MIDPOINT_EXTENSION_W_RIGHT']
            h_start = midpoint_info[1] - self.DIMS_3D['MIDPOINT_EXTENSION_H_FRONT']
            h_end   = midpoint_info[1] + self.DIMS_3D['MIDPOINT_EXTENSION_H_BACK']
            d_start = midpoint_info[2] - self.DIMS_3D['MIDPOINT_EXTENSION_D_BOTTOM']
            d_end   = midpoint_info[2] + self.DIMS_3D['MIDPOINT_EXTENSION_D_TOP']

            slice_img_sitk = utils.read_mha(path_img)
            slice_img = utils.sitk_to_array(slice_img_sitk)
            slice_img = slice_img[w_start:w_end, h_start:h_end, d_start:d_end]
            slice_mask_sitk = utils.read_mha(path_mask)
            slice_mask = utils.sitk_to_array(slice_mask_sitk)
            slice_mask = slice_mask[w_start:w_end, h_start:h_end, d_start:d_end]

            # Step 1.2 - Extract grid
            slice_img_shape_old = slice_img.shape
            slice_mask_shape_old = slice_mask.shape

            grid_start_w_idx = meta1[4]
            grid_start_h_idx = meta1[5]
            grid_start_d_idx = meta1[6]
            grid_end_w_idx = grid_start_w_idx + self.w_grid
            grid_end_h_idx = grid_start_h_idx + self.h_grid
            grid_end_d_idx = grid_start_d_idx + self.d_grid
            slice_img  = slice_img[grid_start_w_idx:grid_end_w_idx , grid_start_h_idx:grid_end_h_idx, grid_start_d_idx:grid_end_d_idx]
            slice_mask = slice_mask[grid_start_w_idx:grid_end_w_idx , grid_start_h_idx:grid_end_h_idx, grid_start_d_idx:grid_end_d_idx]
            assert slice_img.shape  == (self.w_grid,self.h_grid,self.d_grid), '\n\n[ERROR] patient: '  + str(meta2.numpy()) + '\n' +  str(slice_img.shape) + str(meta1[4:].numpy()) + ' -- ' + str(slice_img_shape_old)
            assert slice_mask.shape == (self.w_grid,self.h_grid,self.d_grid), '\n\n[ERROR] patient: ' + str(meta2.numpy()) + '\n' + str(slice_mask.shape) + str(meta1[4:].numpy()) + ' -- ' + str(slice_mask_shape_old)
            
            # Step 2 - One-hot or not
            if self.mask_type == config.MASK_TYPE_ONEHOT:
                label_ids = sorted(list(self.LABEL_MAP.values()))
                label_ids_mask = []
                slice_mask_classes = np.zeros((slice_img.shape[0], slice_img.shape[1], slice_img.shape[2], len(label_ids)))
                for label_id in label_ids:
                    label_idxs = np.argwhere(slice_mask == label_id)
                    if len(label_idxs):
                        slice_mask_classes[label_idxs[:,0], label_idxs[:,1], label_idxs[:,2], label_id] = 1
                        label_ids_mask.append(1)
                    else:
                        label_ids_mask.append(0)
    

            elif self.mask_type == config.MASK_TYPE_COMBINED:
                slice_mask_classes = slice_mask
                slice_mask_classes = tf.expand_dims(slice_mask_classes, axis=3)
            
            x = tf.cast(tf.expand_dims(slice_img, axis=3), dtype=tf.float32)
            y = tf.cast(slice_mask_classes, dtype=tf.float32)

            # Step 3 - Add masks for one-hot data
            if self.mask_type == config.MASK_TYPE_ONEHOT:
                label_ids_mask = tf.constant(label_ids_mask, dtype=tf.int32)
                meta1 = tf.concat([meta1, label_ids_mask], axis=0)

            # Step 4 - return
            return (x,y,meta1, meta2)
        
        else:
            print (' - [ERROR] Issue with path')
            print (' -- [ERROR] path_img : ', path_img)
            print (' -- [ERROR] path_mask: ', path_mask)
            return ([], [], meta1, meta2)

    def path_debug_2D(self, path_img, path_mask):
        patient_id = '0522c0195'   # [0522c0125/40, 0522c0009/72, 0522c0195/70]
        slice_number = 70
        
        path_img_parts = list(Path(path_img).parts)
        if self.resampled:
            path_img_parts[-1] = config.FILENAME_IMG_RESAMPLED_NPY_2D.format(patient_id, slice_number)
        else:
            path_img_parts[-1] = config.FILENAME_IMGNPY_2D.format(patient_id, slice_number)
        path_img_parts[-3] = patient_id
        path_img = Path(*path_img_parts)
        
        path_mask_parts = list(Path(path_mask).parts)
        if self.resampled:
            path_mask_parts[-1] = config.FILENAME_MASK_RESAMPLED_NPY_2D.format(patient_id, slice_number)
        else:      
            path_mask_parts[-1] = config.FILENAME_MASKNPY_2D.format(patient_id, slice_number)
        path_mask_parts[-3] = patient_id
        path_mask = Path(*path_mask_parts)
        
        utils.print_debug_header()
        print (' - path_img : ', path_img.parts[-2:])
        print (' - path_mask: ', path_mask.parts[-2:])

        self.paths_img = [path_img]
        self.paths_mask = [path_mask] 

        return path_img, path_mask 
    
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