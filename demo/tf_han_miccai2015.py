import pdb
import numpy as np
from pathlib import Path

import medloader.dataloader.utils as utils
import medloader.dataloader.config as config
import medloader.dataloader.utils_viz as utils_viz
import medloader.dataloader.tensorflow.augmentations as aug
from medloader.dataloader.tensorflow.dataset import ZipDataset


def get_dataset_han_miccai2015_2D(data_dir, dir_type, transforms=[], resampled=False, debug=False):
    
    from medloader.dataloader.tensorflow.han_miccai2015 import HaNMICCAI2015Dataset

    if len(transforms) == 0:
        transforms = [
                aug.Crop2D()
                # , aug.NormalizeMinMax()
                # , aug.Rotate()
            ]

    dataset = HaNMICCAI2015Dataset(data_dir=data_dir
                    , dir_type=dir_type
                    , resampled=resampled, transforms=transforms
                    , debug=debug)

    return dataset

def get_dataset_han_miccai2015_3D_sampler(data_dir, mask_type, dir_type, transforms=[], resampled=False, debug=False):
    
    from medloader.dataloader.tensorflow.han_miccai2015_sampler import HaNMICCAI2015Dataset

    dataset = HaNMICCAI2015Dataset(data_dir=data_dir
                    , dimension=3, dir_type=dir_type, mask_type=mask_type
                    , resampled=resampled
                    , debug=debug)

    x_shape_w = dataset.w_grid
    x_shape_h = dataset.h_grid
    x_shape_d = dataset.d_grid

    if len(transforms) == 0:
        transforms = [
                aug.NormalizeMinMaxSampler(min_val=config.HU_MIN, max_val=config.HU_MAX, x_shape=(x_shape_h, x_shape_w, x_shape_d,1))
                , aug.Rotate3D()
            ]
        dataset.transforms = transforms

    # return ZipDataset([dataset])
    return dataset

def get_dataset_han_miccai2015_3D_samplerfull(data_dir, mask_type, dir_type, transforms=[], resampled=False, debug=False):
    
    from medloader.dataloader.tensorflow.han_miccai2015_samplerfull import HaNMICCAI2015Dataset

    dataset = HaNMICCAI2015Dataset(data_dir=data_dir
                    , dimension=3, dir_type=dir_type, mask_type=mask_type
                    , resampled=resampled
                    , debug=debug)

    x_shape_w = dataset.w_grid
    x_shape_h = dataset.h_grid
    x_shape_d = dataset.d_grid

    if len(transforms) == 0:
        transforms = [
                aug.NormalizeMinMaxSampler(min_val=config.HU_MIN, max_val=config.HU_MAX, x_shape=(x_shape_h, x_shape_w, x_shape_d,1))
                , aug.Rotate3D()
            ]
        dataset.transforms = transforms

    return ZipDataset([dataset])
    # return dataset

def get_dataset_han_miccai2015_3D_samplerfull_filtered(data_dir, mask_type, dir_type, transforms=[], resampled=False
                                            , single_sample=False, patient_shuffle=True, filterFunc=False
                                            , debug=False):
    
    from medloader.dataloader.tensorflow.han_miccai2015_samplerfull import HaNMICCAI2015Dataset

    # Step 1 - Get dataset class
    dataset = HaNMICCAI2015Dataset(data_dir=data_dir
                    , dimension=3, dir_type=dir_type, mask_type=mask_type
                    , resampled=resampled, patient_shuffle=patient_shuffle
                    , debug=debug, single_sample=single_sample)

    # Step 2 - Transforms
    if len(transforms) == 0:
        x_shape_w = dataset.w_grid
        x_shape_h = dataset.h_grid
        x_shape_d = dataset.d_grid
        transforms = [
                aug.NormalizeMinMaxSampler(min_val=config.HU_MIN, max_val=config.HU_MAX, x_shape=(x_shape_h, x_shape_w, x_shape_d,1))
                , aug.Rotate3D()
            ]
        dataset.transforms = transforms

    # Step 3 - Filters (to eliminate/reduce background-only grids)
    if filterFunc:
        dataset.filterFunc = aug.FilterByMask(len(dataset.LABEL_MAP), dataset.SAMPLER_PERC).execute

    return ZipDataset([dataset])
    # return dataset

def get_dataset_han_miccai2015_3D(data_dir, mask_type, dir_type, transforms=[], resampled=False, debug=False):
    
    from medloader.dataloader.tensorflow.han_miccai2015 import HaNMICCAI2015Dataset

    dataset = HaNMICCAI2015Dataset(data_dir=data_dir
                    , dimension=3, dir_type=dir_type, mask_type=mask_type
                    , resampled=resampled
                    , debug=debug)

    dims_3d_miccai = getattr(config, dataset.name)['DIMS']['3D']
    x_shape_w = dims_3d_miccai['MIDPOINT_EXTENSION_W_LEFT'] + dims_3d_miccai['MIDPOINT_EXTENSION_W_RIGHT']
    x_shape_h = dims_3d_miccai['MIDPOINT_EXTENSION_H_BACK'] + dims_3d_miccai['MIDPOINT_EXTENSION_H_FRONT']
    x_shape_d = dims_3d_miccai['MIDPOINT_EXTENSION_D_TOP']  + dims_3d_miccai['MIDPOINT_EXTENSION_D_BOTTOM']

    if len(transforms) == 0:
        transforms = [
                aug.NormalizeMinMaxSampler(min_val=config.HU_MIN, max_val=config.HU_MAX, x_shape=(x_shape_h, x_shape_w, x_shape_d,1))
                # , aug.Rotate3D()
            ]
    dataset.transforms = transforms

    # return ZipDataset([dataset])
    return dataset

if __name__ == "__main__":

    MAIN_DIR = Path(__file__).parent.absolute().parent.absolute()
    data_dir = Path(MAIN_DIR).joinpath('data2')
    
    batchsize = 1

    if 0:
        dataset2D = get_dataset_han_miccai2015_2D(data_dir=data_dir, dir_type='train', debug=False, resampled=True)
        # dataset.get_voxel_stats(show=True)
        
        if 1:
            for (X,Y,meta1,meta2) in dataset2D.generator().batch(batchsize):
                print (X.shape, Y.shape, meta1.numpy())
                utils_viz.viz_slice_raw_batch(X,Y,meta2,dataset2D)
        
        elif 1:
            utils.benchmark(dataset2D.generator().batch(batchsize))
        
        elif 0:
            import tqdm
            with tqdm.tqdm(total=len(dataset2D), desc=dataset2D.name) as pbar:
                for (X,Y,meta1,meta2) in dataset2D.generator().batch(1):
                    if X.shape[1] != 320 or X.shape[2] != 320:
                        filepath = Path(dataset2D.paths_raw[meta1[0]])
                        print (' - X: ', X.shape,' || filename: ', filepath.parts[-1])
                    pbar.update(1)
    
    elif 0:
        dataset3D = get_dataset_han_miccai2015_3D(data_dir=data_dir
                    , dir_type='train' # [train, test_offsite]
                    , resampled=True
                    , mask_type=config.MASK_TYPE_COMBINED # [config.MASK_TYPE_ONEHOT, config.MASK_TYPE_COMBINED]
                    , debug=False)

        if 1:
            for _ in range(1):
                for (X,Y,meta1,meta2) in dataset3D.generator().batch(batchsize):
                    print (' - ', X.shape, Y.shape, meta2)

                    if Y.shape[-1] == 1:
                        utils_viz.viz_3d_mask(Y[:,:,:,:,0], dataset3D, meta1, meta2)

                    # utils_viz.viz_3d_slices(X, Y, dataset3D, meta1, meta2)
                    pdb.set_trace()
                
        elif 0:
            for _ in range(3):
                print (' ----------------------------------- ')
                utils.benchmark(dataset3D.generator().batch(batchsize))
    
    elif 0:
        dataset3D = get_dataset_han_miccai2015_3D_sampler(data_dir=data_dir
                    , dir_type='train' # [train, test_offsite]
                    , resampled=True
                    , mask_type=config.MASK_TYPE_ONEHOT # [config.MASK_TYPE_ONEHOT, config.MASK_TYPE_COMBINED]
                    , debug=False)
        
        if 1:
            for _ in range(1):
                for (X,Y,meta1,meta2) in dataset3D.generator().batch(batchsize):
                    print (' - ', X.shape, Y.shape, meta2.numpy(), meta1.numpy())
                    # if Y.shape[-1] == 1:
                    #     utils_viz.viz_3d_mask(Y[:,:,:,:,0], dataset3D, meta1, meta2)
                    
                    pdb.set_trace()
        elif 0:
            for _ in range(1):
                print (' ----------------------------------- ')
                utils.benchmark(dataset3D.generator().batch(batchsize))
    
    # Sampler Full
    elif 0:
        print (' ------------------ medloader.dataloader.tensorflow.han_miccai2015_samplerfull.py\n')
        dataset3D = get_dataset_han_miccai2015_3D_samplerfull(data_dir=data_dir
                    , dir_type='train' # [train, test_offsite]
                    , resampled=True
                    , mask_type=config.MASK_TYPE_ONEHOT # [config.MASK_TYPE_ONEHOT, config.MASK_TYPE_COMBINED]
                    , debug=False)
        
        if 1:
            for _ in range(1):
                count = 0
                for (X,Y,meta1,meta2) in dataset3D.generator().batch(batchsize):
                    # print (' - ', X.shape, Y.shape, meta2.numpy(), meta1.numpy())
                    print (meta1)
                    count += 1
                    # pdb.set_trace()
                print (' - total grids: ', count)
    
    # Sampler Full (with filter)
    elif 1:
        print (' ------------------ medloader.dataloader.tensorflow.han_miccai2015_samplerfull.py\n')
        dataset3D = get_dataset_han_miccai2015_3D_samplerfull_filtered(data_dir=data_dir
                    , dir_type='train' # [train, test_offsite]
                    , resampled=True, patient_shuffle=False, filterFunc=False
                    , mask_type=config.MASK_TYPE_ONEHOT # [config.MASK_TYPE_ONEHOT, config.MASK_TYPE_COMBINED]
                    , debug=False, single_sample=False)
        
        if 1:
            res = {'background':0, 'non-background':0, 'total':0}
            for _ in range(1):
                for (X,Y,meta1,meta2) in dataset3D.generator().batch(batchsize):
                    print (' - ', X.shape, Y.shape, meta2.numpy(), meta1.numpy())
                    
                    # print (meta1)

                    if 0:
                        meta1 = meta1[0]
                        res['total'] += 1
                        sum_nonbackground = np.sum(meta1[-9:])
                        if sum_nonbackground > 0: res['non-background'] += 1
                        else                    : res['background'] += 1                            
                        print (res, meta1[-10:].numpy(), sum_nonbackground)

                    pdb.set_trace()
                