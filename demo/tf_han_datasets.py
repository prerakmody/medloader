
import medloader.dataloader.config as config
import medloader.dataloader.utils as utils
import medloader.dataloader.utils_viz as utils_viz
import medloader.dataloader.tensorflow.augmentations as aug

from medloader.dataloader.tensorflow.han_miccai2015 import HaNMICCAI2015Dataset
from medloader.dataloader.tensorflow.han_tciacetuximab import HaNTCIACetuximabDataset
from medloader.dataloader.tensorflow.dataset import ZipDataset

from pathlib import Path
import tensorflow as tf

def get_dataset_han_tciacetuximab(data_dir, transforms=[], batch_size=1, resampled=False, debug=False):
    
    if len(transforms) == 0:
        transforms = [
                aug.Crop2D(h_start=config.TCIACETUXIMAB_H_START, h_end=config.TCIACETUXIMAB_H_END
                        , w_start=config.TCIACETUXIMAB_W_START, w_end=config.TCIACETUXIMAB_W_END
                    )
                , aug.NormalizeMinMax()
                , aug.Rotate()
            ]

    dataset = HaNTCIACetuximabDataset(data_dir=data_dir
                ,transforms=transforms, resampled=resampled, batch_size=batch_size
                , debug=debug)

    return dataset

def get_dataset_han_micca2015(data_dir, dir_type, transforms=[], resampled=False, debug=False):
    
    if len(transforms) == 0:
        transforms = [
                aug.Crop2D(h_start=config.MICCAI2015_H_START, h_end=config.MICCAI2015_H_END, w_start=config.MICCAI2015_W_START, w_end=config.MICCAI2015_W_END)
                , aug.NormalizeMinMax()
                , aug.Rotate()
            ]

    dataset = HaNMICCAI2015Dataset(data_dir=data_dir
                    , dir_type=dir_type
                    , resampled=resampled, transforms=transforms
                    , debug=debug)

    return dataset

if __name__ == "__main__":

    MAIN_DIR = Path(__file__).parent.absolute().parent.absolute()
    data_dir = Path(MAIN_DIR).joinpath('data')

    batch_size = 2
    datasets = []
    datasets_generators = []

    if 1:
        dataset_han_miccai2015 = get_dataset_han_micca2015(data_dir=data_dir, dir_type='train', resampled=True)
        dataset_han_tciacetuximab = get_dataset_han_tciacetuximab(data_dir=data_dir, resampled=True)
        datasets = [dataset_han_miccai2015, dataset_han_tciacetuximab]

    dataset_global = ZipDataset(datasets)

    if 1:
        for (X,Y,meta1,meta2) in dataset_global.generator().batch(batch_size):
            utils_viz.viz_slice_raw_batch_datasets(X,Y,meta1,meta2,datasets)
    else:
        utils.benchmark(dataset_global, batch_size)
