

import pdb
import tqdm
import nrrd
import copy
import traceback
import numpy as np
from pathlib import Path


import medloader.dataloader.config as config
import medloader.dataloader.utils as utils

class HaNMICCAI2015Downloader:

    def __init__(self, dataset_dir_raw, dataset_dir_processed):
        self.dataset_dir_raw = dataset_dir_raw 
        self.dataset_dir_processed = dataset_dir_processed

    def download(self):
        self.dataset_dir_raw.mkdir(parents=True, exist_ok=True)
        # Step 1 - Download .zips and unzip them
        urls_zip = ['http://www.imagenglab.com/data/pddca/PDDCA-1.4.1_part1.zip'
                    , 'http://www.imagenglab.com/data/pddca/PDDCA-1.4.1_part2.zip'
                    , 'http://www.imagenglab.com/data/pddca/PDDCA-1.4.1_part3.zip']
        
        import concurrent
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for url_zip in urls_zip:
                filepath_zip = Path(self.dataset_dir_raw, url_zip.split('/')[-1])

                # Step 1.1 - Download .zip and then unzip it
                if not Path(filepath_zip).exists():
                    executor.submit(utils.download_zip, url_zip, filepath_zip, self.dataset_dir_raw)
                else:
                    executor.submit(utils.read_zip, filepath_zip, self.dataset_dir_raw)
                
                # Step 1.2 - Unzip .zip
                # executor.submit(utils.read_zip(filepath_zip, self.dataset_dir_raw)

    def sort(self, dataset_dir_datatypes, dataset_dir_datatypes_ranges):
        
        print ('')
        import tqdm
        import shutil
        import numpy as np

        # Step 1 - Make necessay directories
        self.dataset_dir_raw.mkdir(parents=True, exist_ok=True)
        for each in dataset_dir_datatypes:
            path_tmp = Path(self.dataset_dir_raw).joinpath(each)
            path_tmp.mkdir(parents=True, exist_ok=True)

        # Step 2 - Sort
        with tqdm.tqdm(total=len(list(Path(self.dataset_dir_raw).glob('0522*'))), desc='Sorting', leave=False) as pbar:
            for path_patient in self.dataset_dir_raw.iterdir():
                if '.zip' not in path_patient.parts[-1]: #and path_patient.parts[-1] not in dataset_dir_datatypes:
                    try:
                        patient_number = Path(path_patient).parts[-1][-3:]
                        if patient_number.isdigit():
                            folder_id = np.digitize(patient_number, dataset_dir_datatypes_ranges)
                            shutil.move(src=str(path_patient), dst=str(Path(self.dataset_dir_raw).joinpath(dataset_dir_datatypes[folder_id])))
                            pbar.update(1)
                    except:
                        traceback.print_exc()
                        pdb.set_trace()

class HaNMICCAI2015Extractor:
    """
    More information on the .nrrd format can be found here: http://teem.sourceforge.net/nrrd/format.html#space
    """

    def __init__(self, name, dataset_dir_raw, dataset_dir_processed, dataset_dir_datatypes):
        self.name = name
        self.dataset_dir_raw = dataset_dir_raw 
        self.dataset_dir_processed = dataset_dir_processed 
        self.dataset_dir_datatypes = dataset_dir_datatypes
        self.folder_prefix = '0522'
        
        self._init_constants()
    
    def _init_constants(self):
        self.DATATYPE_ORIG = '.nrrd'
        self.IMG_VOXEL_FILENAME = 'img.nrrd'
        self.MASK_VOXEL_FILENAME = 'mask.nrrd'
        self.LABEL_MAP = getattr(config, self.name)['LABEL_MAP']
        self.IGNORE_LABELS = getattr(config,self.name)['IGNORE_LABELS']

        self.KEYNAME_PIXEL_SPACING = 'space directions'
        self.KEYNAME_ORIGIN = 'space origin'
        self.KEYNAME_SHAPE = 'sizes'
    
    def extract3D(self):
        
        import concurrent
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # for dir_type in ['train']:
                # self._extract3D_patients(Path(self.dataset_dir_raw).joinpath(dir_type))
            for dir_type in self.dataset_dir_datatypes:
                executor.submit(self._extract3D_patients, Path(self.dataset_dir_raw).joinpath(dir_type))
        
        print ('')
        print (' - Note: You can view the 3D data in visualizers like MeVisLab or 3DSlicer')
        print ('')
    
    def _extract3D_patients(self, dir_dataset):
        dir_type = Path(dir_dataset).parts[-1]
        paths_global_voxel_img = []
        paths_global_voxel_mask = []

        dir_type_idx = self.dataset_dir_datatypes.index(dir_type)
        with tqdm.tqdm(total=len(list(dir_dataset.glob('*'))), desc='[3D][{}] Patients: '.format(dir_type), disable=False, position=dir_type_idx) as pbar:
            for _, patient_dir_path in enumerate(dir_dataset.iterdir()):
                voxel_img_filepath, voxel_mask_filepath = self._extract3D_patient(patient_dir_path)
                paths_global_voxel_img.append(voxel_img_filepath)
                paths_global_voxel_mask.append(voxel_mask_filepath)
                pbar.update(1)
        
        if len(paths_global_voxel_img) and len(paths_global_voxel_mask):
            paths_global_voxel_img = list(map(lambda x: str(x), paths_global_voxel_img))
            paths_global_voxel_mask = list(map(lambda x: str(x), paths_global_voxel_mask))
            utils.save_csv(Path(self.dataset_dir_processed).joinpath(dir_type, config.DIRNAME_SAVE_3D, config.FILENAME_CSV_IMG), paths_global_voxel_img)
            utils.save_csv(Path(self.dataset_dir_processed).joinpath(dir_type, config.DIRNAME_SAVE_3D, config.FILENAME_CSV_MASK), paths_global_voxel_mask)
            
            if len(config.VOXEL_RESO):
                paths_global_voxel_img_resampled = list(map(lambda x: x.replace(config.FILENAME_IMG_3D, config.FILENAME_IMG_RESAMPLED_3D), paths_global_voxel_img) )
                paths_global_voxel_mask_resampled = list(map(lambda x: x.replace(config.FILENAME_MASK_3D, config.FILENAME_MASK_RESAMPLED_3D), paths_global_voxel_mask) )
                utils.save_csv(Path(self.dataset_dir_processed).joinpath(dir_type, config.DIRNAME_SAVE_3D, config.FILENAME_CSV_IMG_RESAMPLED), paths_global_voxel_img_resampled)
                utils.save_csv(Path(self.dataset_dir_processed).joinpath(dir_type, config.DIRNAME_SAVE_3D, config.FILENAME_CSV_MASK_RESAMPLED), paths_global_voxel_mask_resampled)
    
    def _extract3D_patient(self, patient_dir):

        try:
            voxel_img, voxel_img_headers, voxel_mask, voxel_mask_headers = self._get_data3D(patient_dir) 

            dir_type = Path(patient_dir).parts[-2]
            patient_id = Path(patient_dir).parts[-1]
            return self._save_data3D(dir_type, patient_id, voxel_img, voxel_img_headers, voxel_mask, voxel_mask_headers)

        except:
            print (' - [ERROR][_extract_patient()] path_folder: ', patient_dir.parts[-1])
            traceback.print_exc()
            pdb.set_trace()

    def _get_data3D(self, patient_dir):
        try:
            voxel_img, voxel_mask = [], []
            voxel_img_headers, voxel_mask_headers = {}, {}
            if Path(patient_dir).exists():

                # Step 1 - Get Voxel Data
                path_voxel_img = Path(patient_dir).joinpath(self.IMG_VOXEL_FILENAME)
                voxel_img, voxel_img_headers = self._get_voxel_img(path_voxel_img)

                # Step 2 - Get Mask Data
                path_voxel_mask = Path(patient_dir).joinpath(self.MASK_VOXEL_FILENAME)
                voxel_mask, voxel_mask_headers = self._get_voxel_mask(path_voxel_mask)
                
            else:
                print (' - Error: Path does not exist: patient_dir', patient_dir)

            return voxel_img, voxel_img_headers, voxel_mask, voxel_mask_headers

        except:
            print (' - [ERROR][get_data()] patient_dir: ', patient_dir.parts[-1])
            traceback.print_exc()
            pdb.set_trace()
    
    def _get_voxel_img(self, path_voxel, histogram=False):
        try:
            if path_voxel.exists():
                voxel_img_data, voxel_img_header = nrrd.read(str(path_voxel))  # shape=[H,W,D]

                if histogram:
                    import matplotlib.pyplot as plt
                    plt.hist(voxel_img_data.flatten())
                    plt.show()

                return voxel_img_data, voxel_img_header
            else:
                print (' - Error: Path does not exist: ', path_voxel)
        except:
            traceback.print_exc()
            pdb.set_trace()
    
    def _get_voxel_mask(self, path_voxel_mask):
        try:
            if Path(path_voxel_mask).exists():
                voxel_mask_data, voxel_mask_headers = nrrd.read(str(path_voxel_mask)) 
                return voxel_mask_data, voxel_mask_headers
            else:
                path_mask_folder = Path(*Path(path_voxel_mask).parts[:-1]).joinpath('structures')
                voxel_mask_data, voxel_mask_headers = self._merge_masks(path_mask_folder)
                return voxel_mask_data, voxel_mask_headers

        except:
            traceback.print_exc()
            pdb.set_trace()
    
    def _merge_masks(self, path_mask_folder):
        try:
            voxel_mask_full = []
            voxel_mask_headers = {}
            if Path(path_mask_folder).exists():
                with tqdm.tqdm(total=len(list(Path(path_mask_folder).glob('*{}'.format(self.DATATYPE_ORIG)))), leave=False, disable=True) as pbar_mask:
                    for filepath_mask in Path(path_mask_folder).iterdir():
                        voxel_mask, voxel_mask_headers = nrrd.read(str(filepath_mask))
                        class_name = Path(filepath_mask).parts[-1].split(self.DATATYPE_ORIG)[0]
                        class_id = -1
                        if class_name in self.LABEL_MAP:
                            class_id = self.LABEL_MAP[class_name]
                        if class_id not in self.IGNORE_LABELS:
                            if len(voxel_mask_full) == 0: 
                                voxel_mask_full = copy.deepcopy(voxel_mask)    
                            idxs = np.argwhere(voxel_mask > 0)
                            voxel_mask_full[idxs[:,0], idxs[:,1], idxs[:,2]] = class_id
                            if 0:
                                print (' - [merge_masks()] class_id:', class_id, ' || name: ', class_name, ' || idxs: ', len(idxs))
                                print (' --- [merge_masks()] label_ids: ', np.unique(voxel_mask_full))
                        pbar_mask.update(1)
                        
                path_mask = Path(*Path(path_mask_folder).parts[:-1]).joinpath(self.MASK_VOXEL_FILENAME)
                nrrd.write(str(path_mask), voxel_mask_full, voxel_mask_headers)
                        
            else:
                print (' - Error with path_mask_folder: ', path_mask_folder)
            
            return voxel_mask_full, voxel_mask_headers
        
        except:
            traceback.print_exc()
            pdb.set_trace()

    def _save_data3D(self, dir_type, patient_id, voxel_img, voxel_img_headers, voxel_mask, voxel_mask_headers):
        try:
            
            # Step 1 - Create directory
            voxel_savedir = Path(self.dataset_dir_processed).joinpath(dir_type, config.DIRNAME_SAVE_3D)
            
            # Step 2.1 - Save voxel
            voxel_img_headers_new = {config.TYPE_VOXEL_ORIGSHAPE:{}}
            voxel_img_headers_new[config.TYPE_VOXEL_ORIGSHAPE][config.KEYNAME_PIXEL_SPACING] = voxel_img_headers[self.KEYNAME_PIXEL_SPACING][voxel_img_headers[self.KEYNAME_PIXEL_SPACING] > 0].tolist()
            voxel_img_headers_new[config.TYPE_VOXEL_ORIGSHAPE][config.KEYNAME_ORIGIN] = voxel_img_headers[self.KEYNAME_ORIGIN].tolist()
            voxel_img_headers_new[config.TYPE_VOXEL_ORIGSHAPE][config.KEYNAME_SHAPE] = voxel_img_headers[self.KEYNAME_SHAPE].tolist()
            voxel_img_headers_new[config.TYPE_VOXEL_ORIGSHAPE][config.KEYNAME_OTHERS] = voxel_img_headers
            
            if len(voxel_img) and len(voxel_mask):
                resample_save = False
                if len(config.VOXEL_RESO):
                    resample_save = True
                
                # Find average HU value in the brainstem
                brainstem_idxs = np.argwhere(voxel_mask == 1)
                brainstem_idxs_mean = np.mean(brainstem_idxs, axis=0)
                voxel_img_headers_new[config.TYPE_VOXEL_ORIGSHAPE][config.KEYNAME_MEAN_BRAINSTEAM] = brainstem_idxs_mean.tolist()

                return utils.save_as_mha(voxel_savedir, patient_id, voxel_img, voxel_img_headers_new, voxel_mask, resample_save=resample_save)
                

        except:
            traceback.print_exc()
            pdb.set_trace()

    def extract2D(self):
        import concurrent
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for dir_type in self.dataset_dir_datatypes:
                executor.submit(self._extract2D_patients, Path(self.dataset_dir_processed).joinpath(dir_type, config.DIRNAME_SAVE_3D))
                # break

    def _extract2D_patients(self, dir_dataset):
        try:

            paths_global_2D_img = []
            paths_global_2D_mask = []
            paths_global_2D_img_resampled = []
            paths_global_2D_mask_resampled = []
            voxel_shape_info = {}

            dir_type = Path(dir_dataset).parts[-2]
            dir_type_index = self.dataset_dir_datatypes.index(dir_type)

            with tqdm.tqdm(total=len(list(dir_dataset.glob(self.folder_prefix + '*'))), desc='[2D][{}] Patients: '.format(dir_type)
                            , disable=False, position=dir_type_index) as pbar_patients_2D:
                for patient_counter, patient_dir_path in enumerate(dir_dataset.iterdir()):
                    if '.csv' in patient_dir_path.parts[-1] or '.json' in patient_dir_path.parts[-1] : continue

                    for voxel_type in [config.TYPE_VOXEL_ORIGSHAPE, config.TYPE_VOXEL_RESAMPLED]:
                        paths_image_img, paths_image_mask, voxel_shape = self._extract2D_patient(patient_dir_path, voxel_type)
                        for voxel_type in paths_image_img:
                            if voxel_type == config.TYPE_VOXEL_ORIGSHAPE:
                                paths_global_2D_img.extend(paths_image_img[voxel_type])
                                paths_global_2D_mask.extend(paths_image_mask[voxel_type])
                            elif voxel_type == config.TYPE_VOXEL_RESAMPLED:
                                paths_global_2D_img_resampled.extend(paths_image_img[voxel_type])
                                paths_global_2D_mask_resampled.extend(paths_image_mask[voxel_type])

                        if len(voxel_shape):
                            for patient_id in voxel_shape:
                                if patient_id in voxel_shape_info:
                                    for voxel_type in voxel_shape[patient_id]:
                                        voxel_shape_info[patient_id][voxel_type] = voxel_shape[patient_id][voxel_type]
                                else:
                                    voxel_shape_info[patient_id] = voxel_shape[patient_id]
                         
                    
                    pbar_patients_2D.update(1)

            if len(paths_global_2D_img) and len(paths_global_2D_mask):
                utils.save_csv(Path(self.dataset_dir_processed).joinpath(dir_type, config.DIRNAME_SAVE_2D, config.FILENAME_CSV_IMG), paths_global_2D_img)
                utils.save_csv(Path(self.dataset_dir_processed).joinpath(dir_type, config.DIRNAME_SAVE_2D, config.FILENAME_CSV_MASK), paths_global_2D_mask)

                if len(config.VOXEL_RESO):
                    utils.save_csv(Path(self.dataset_dir_processed).joinpath(dir_type, config.DIRNAME_SAVE_2D, config.FILENAME_CSV_IMG_RESAMPLED), paths_global_2D_img_resampled)
                    utils.save_csv(Path(self.dataset_dir_processed).joinpath(dir_type, config.DIRNAME_SAVE_2D, config.FILENAME_CSV_MASK_RESAMPLED), paths_global_2D_mask_resampled)
                
                import json
                filepath_json = Path(self.dataset_dir_processed).joinpath(dir_type, config.DIRNAME_SAVE_2D, config.FILENAME_VOXEL_INFO)
                with open(filepath_json, 'w') as fp:
                    json.dump(voxel_shape_info, fp, cls=utils.NpEncoder, sort_keys=True, indent=4)

        except:
            traceback.print_exc()
            pdb.set_trace()

    def _extract2D_patient(self, patient_dir, voxel_type):
        try:
            import skimage
            import skimage.io

            paths_img = []
            paths_mask = []
            voxel_shape = {}

            if voxel_type == config.TYPE_VOXEL_RESAMPLED:
                if len(config.VOXEL_RESO) == 0:
                    return {voxel_type:paths_img}, {voxel_type:paths_mask}, voxel_shape 

            if Path(patient_dir).exists():
                dir_type = Path(patient_dir).parts[-3]
                patient_id = Path(patient_dir).parts[-1]
                path_processed_patient_img = Path(self.dataset_dir_processed).joinpath(dir_type, config.DIRNAME_SAVE_2D, patient_id, config.DIRNAME_IMG_2D)
                path_processed_patient_mask = Path(self.dataset_dir_processed).joinpath(dir_type, config.DIRNAME_SAVE_2D, patient_id, config.DIRNAME_MASK_2D)
                path_processed_patient_img.mkdir(parents=True, exist_ok=True)
                path_processed_patient_mask.mkdir(parents=True, exist_ok=True)

                try:

                    # STEP 1 - GET VOXEL DATA
                    path_voxel_img, path_voxel_mask = '',''
                    if voxel_type == config.TYPE_VOXEL_ORIGSHAPE:
                        path_voxel_img  = Path(patient_dir, config.FILENAME_IMG_3D)
                        path_voxel_mask = Path(patient_dir, config.FILENAME_MASK_3D)
                    elif voxel_type == config.TYPE_VOXEL_RESAMPLED:
                        path_voxel_img  = Path(patient_dir, config.FILENAME_IMG_RESAMPLED_3D)
                        path_voxel_mask = Path(patient_dir, config.FILENAME_MASK_RESAMPLED_3D)

                    voxel_img_sitk  = utils.read_mha(str(path_voxel_img)) # [H,W,D]
                    voxel_img_data = utils.sitk_to_array(voxel_img_sitk)
                    voxel_img_spacing = voxel_img_sitk.GetSpacing()
                    
                    voxel_mask_sitk = utils.read_mha(str(path_voxel_mask))
                    voxel_mask_data = utils.sitk_to_array(voxel_mask_sitk)

                    # STEP 2 - LOOP OVER INTERESTING SLICES
                    # slice_ids = range(voxel_img_data.shape[2])
                    slice_ids = np.where(np.sum(voxel_mask_data, axis=(0,1)) > 0)[0]
                    for slice_id in slice_ids:
                        slice_img  = voxel_img_data[:,:,slice_id]
                        slice_mask = voxel_mask_data[:,:,slice_id]
                        
                        path_img_npy, path_mask_npy = '',''
                        if voxel_type == config.TYPE_VOXEL_ORIGSHAPE:
                            path_img_npy = Path(path_processed_patient_img).joinpath(config.FILENAME_IMGNPY_2D.format(patient_id, slice_id))
                            path_mask_npy = Path(path_processed_patient_mask).joinpath(config.FILENAME_MASKNPY_2D.format(patient_id, slice_id))
                        elif voxel_type == config.TYPE_VOXEL_RESAMPLED:
                            path_img_npy = Path(path_processed_patient_img).joinpath(config.FILENAME_IMG_RESAMPLED_NPY_2D.format(patient_id, slice_id))
                            path_mask_npy = Path(path_processed_patient_mask).joinpath(config.FILENAME_MASK_RESAMPLED_NPY_2D.format(patient_id, slice_id))
                        np.save(path_img_npy, slice_img)
                        np.save(path_mask_npy, slice_mask)
                        
                        if 0:
                            import medloader.dataloader.utils_viz as utils_viz
                            utils_viz.viz_slice_simple(slice_img, slice_mask)

                        paths_img.append(str(path_img_npy.absolute()))
                        paths_mask.append(str(path_mask_npy.absolute()))
                        voxel_shape[patient_id] = {voxel_type : {'shape' : voxel_img_data.shape } }
                        voxel_shape[patient_id][voxel_type]['slice_ids'] = list(slice_ids)
                        voxel_shape[patient_id][voxel_type]['spacing'] = voxel_img_spacing

                except:
                    traceback.print_exc()
                    pdb.set_trace()

            else:
                print (' - [ERROR][convert_3d_to_2d()] patient_dir does not exist: ', patient_dir)

            return {voxel_type:paths_img}, {voxel_type:paths_mask}, voxel_shape

        except:
            traceback.print_exc()
            pdb.set_trace()
