from medloader.demo.tf_han_miccai2015_trainer import PROJECT_DIR
import pdb
from pathlib import Path

import medloader.dataloader.utils as utils
import medloader.dataloader.config as config
import medloader.dataloader.utils_viz as utils_viz
import medloader.dataloader.tensorflow.augmentations as aug
from medloader.dataloader.tensorflow.dataset import ZipDataset

PROJECT_DIR = Path(__file__).parent.absolute().parent.absolute()
data_dir = Path(PROJECT_DIR).joinpath('_data')

def get_dataset_han_miccai2015_3D_grid(data_dir, dir_type=['train']
                    , dimension=3, grid=True, resampled=True, mask_type=config.MASK_TYPE_ONEHOT
                    , transforms=[], filter_grid=True
                    , parallel_calls=None, deterministic=False
                    , patient_shuffle=True
                    , debug=False, single_sample=False):
    
    from medloader.dataloader.tensorflow.han_miccai2015 import HaNMICCAI2015Dataset

    datasets = []
    for dir_type_ in dir_type:
        # Step 1 - Get dataset class
        dataset = HaNMICCAI2015Dataset(data_dir=data_dir, dir_type=dir_type_
                        , dimension=dimension, grid=grid, resampled=resampled, mask_type=mask_type
                        , transforms=transforms, filter_grid=filter_grid
                        , parallel_calls=parallel_calls, deterministic=deterministic
                        , patient_shuffle=patient_shuffle
                        , debug=debug, single_sample=single_sample)

        # Step 2 - Transforms
        if transforms:
            x_shape_w = dataset.w_grid
            x_shape_h = dataset.h_grid
            x_shape_d = dataset.d_grid
            transforms = [
                    aug.NormalizeMinMaxSampler(min_val=config.HU_MIN, max_val=config.HU_MAX, x_shape=(x_shape_h, x_shape_w, x_shape_d,1))
                    , aug.Rotate3D()
                ]
            dataset.transforms = transforms

        # Step 3 - Filters (to eliminate/reduce background-only grids)
        if filter_grid:
            dataset.filter = aug.FilterByMask(len(dataset.LABEL_MAP), dataset.SAMPLER_PERC)

        datasets.append(dataset)

    # Step 4- Return
    return ZipDataset(datasets)

def get_dataset_han_miccai2015_3D_full(data_dir, dir_type='train'
                    , dimension=3, grid=False, resampled=True, mask_type=config.MASK_TYPE_ONEHOT
                    , transforms=[], filter_grid=False
                    , parallel_calls=None, deterministic=False
                    , patient_shuffle=True
                    , debug=False, single_sample=False):
    
    from medloader.dataloader.tensorflow.han_miccai2015 import HaNMICCAI2015Dataset

    # Step 1 - Get dataset class
    dataset = HaNMICCAI2015Dataset(data_dir=data_dir, dir_type=dir_type
                    , dimension=dimension, grid=grid, resampled=resampled, mask_type=mask_type
                    , transforms=transforms, filter_grid=filter_grid
                    , parallel_calls=parallel_calls, deterministic=deterministic
                    , patient_shuffle=patient_shuffle
                    , debug=debug, single_sample=single_sample)

    # Step 2 - Transforms
    if transforms:
        x_shape_w = dataset.w_grid
        x_shape_h = dataset.h_grid
        x_shape_d = dataset.d_grid
        transforms = [
                aug.NormalizeMinMaxSampler(min_val=config.HU_MIN, max_val=config.HU_MAX, x_shape=(x_shape_h, x_shape_w, x_shape_d,1))
            ]
        dataset.transforms = transforms

    # Step 3 - Return
    return ZipDataset([dataset])

if 1:
    dataset3D = get_dataset_han_miccai2015_3D_full(data_dir=data_dir, dir_type=['train']
                                                , mask_type=config.MASK_TYPE_COMBINED, resampled=False
                                                , transforms=False)

# # 1. Main - full volume extractor (for viewing purposes)
if 0:
    batchsize = 1
    dataset3D = get_dataset_han_miccai2015_3D_full(data_dir=data_dir, dir_type=['train']
                                                , mask_type=config.MASK_TYPE_COMBINED, resampled=False
                                                , transforms=False)

    for (X,Y,meta1,meta2) in dataset3D.generator().batch(batchsize):
        print (' - ', X.shape, Y.shape, meta1.numpy())
        if len(Y.shape) == 4:
            datasets_this = utils.get_dataset_from_zip(meta2, dataset3D)
            utils_viz.viz_3d_mask(Y, datasets_this, meta1, meta2)
            pdb.set_trace()

# # 2. Main - grid extractor (for ML purposes - a Tensorflow loader)
if 0:
    batchsize = 1
    dataset3D = get_dataset_han_miccai2015_3D_grid(data_dir=data_dir, dir_type=['train']
                                                , mask_type=config.MASK_TYPE_ONEHOT, resampled=True
                                                , filter=True)

    try:
        for (X,Y,meta1,meta2) in dataset3D.generator().batch(batchsize):
            if X.shape != (1,96,96,96,1) and Y.shape != (1,96,96,96,10):
                print (X.shape, Y.shape, meta1.numpy())
                pdb.set_trace()
    except:
        import traceback
        traceback.print_exc()

# # 3. Main - grid extractor (for ML benchmarking purposes)
if 0:
    import medloader.dataloader.utils as utils

    batchsize = 2
    dataset3D = get_dataset_han_miccai2015_3D_grid(data_dir=data_dir, dir_type=['train', 'train_additional']
                                                , mask_type=config.MASK_TYPE_ONEHOT, resampled=True
                                                , filter_grid=True, transforms=True
                                                , parallel_calls=2
                                                , patient_shuffle=False 
                                                , single_sample=True)
    print (' ---------------- ')
    utils.benchmark(dataset3D.generator().batch(batchsize).prefetch(5), model_time=0.1)

# # 4. Main - grid extractor (for viewing purposes)
if 0:
    batchsize = 1
    dataset3D = get_dataset_han_miccai2015_3D_grid(data_dir=data_dir, dir_type='train'
                                                , mask_type=config.MASK_TYPE_COMBINED, resampled=False
                                                , filter=True, transforms=False)

    for (X,Y,meta1,meta2) in dataset3D.generator().batch(batchsize):
        print (' - ', X.shape, Y.shape, meta1.numpy())
        if len(Y.shape) == 4:
            datasets_this = utils.get_dataset_from_zip(meta2, dataset3D)
            utils_viz.viz_3d_mask(Y, datasets_this, meta1, meta2)
            


