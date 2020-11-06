import pdb
from pathlib import Path

import medloader.nnet.config as config
import medloader.nnet.tensorflow.train as train

MAIN_DIR = Path(__file__).parent.absolute().parent.absolute()
data_dir = Path(MAIN_DIR).joinpath('data')

params = {
    'MAIN_DIR': MAIN_DIR
    , 'random_seed': 42
    , 'exp_name': 'UNet3D_seed42'
    , 'dataloader': {
        'data_dir': data_dir
        , 'name': config.DATALOADER_MICCAI2015
        , 'resampled': True
        , 'single_sample': True
        , 'batch_size': 2
    }
    , 'model':{
        'name': config.MODEL_UNET3D
        , 'kernel_reg': False
        , 'optimizer': config.OPTIMIZER_ADAM
        , 'lr': 0.005
        , 'epochs': 2
        , 'epochs_save': 10
        , 'epochs_eval': 40
        , 'epochs_viz': 100
        , 'load_model': {'load': False, 'load_epoch':-1}
    }
    , 'metrics':{
        'metrics_loss': {'Dice': config.LOSS_DICE}
        , 'metrics_eval': {'Dice': config.LOSS_DICE}
        , 'loss_weighted': {'Dice': True}
    }
}

trainer = train.Trainer(params)
trainer.train()