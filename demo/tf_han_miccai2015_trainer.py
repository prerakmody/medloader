import pdb
from pathlib import Path

import medloader.nnet.config as config
import medloader.nnet.tensorflow.train as train

# Step 1 - Set PROJECT_DIR
PROJECT_DIR = Path(__file__).parent.absolute().parent.absolute()

# Step 2.1 - Set dataloader, model and metrics params (for training)
if 1:
    params = {
        'PROJECT_DIR': PROJECT_DIR
        , 'random_seed': 42
        , 'exp_name': 'UNet3DASPP_WeightedMaskedDice_LossVector_Softmax_seed42'
        , 'dataloader': {
            'data_dir': Path(PROJECT_DIR).joinpath('_data')
            , 'dir_type': [config.DATALOADER_MICCAI2015_TRAIN, config.DATALOADER_MICCAI2015_TRAIN_ADD]
            , 'name': config.DATALOADER_MICCAI2015
            , 'resampled': True
            , 'single_sample': False # !!!!!!!!!!!!!!!!!!!!!!!! [WATCH OUT!] !!!!!!!!!!!!!!!!!!!!!!!!
            , 'batch_size': 2
            , 'prefetch_batch': 5 # [1]
            , 'parallel_calls': 2 # None
            , 'random_grid': True
        }
        , 'model':{
            'name': config.MODEL_UNET3DASPP # [config.MODEL_UNET3D, config.MODEL_UNET3DSMALL, config.MODEL_UNET3DSHALLOW, config.MODEL_UNET3DASPP]
            , 'activation': 'softmax' # ['softmax', 'sigmoid']
            , 'kernel_reg': False
            , 'optimizer': config.OPTIMIZER_ADAM
            , 'grad_persistent': False
            , 'init_lr': 0.005
            , 'epochs':  1000
            , 'epochs_save': 20
            , 'epochs_eval': 60
            , 'epochs_viz': 300
            , 'load_model': {'load': False, 'load_epoch':-1} # [False, -1]
            , 'model_tboard': False
            , 'profiler': {
                    'profile': False
                    , 'epochs': [1,2]
                    , 'steps_per_epoch': 50
                    , 'starting_step': 2
                }
        }
        , 'metrics':{
            'metrics_eval': {'Dice': config.LOSS_DICE}
            , 'metrics_loss': {'Dice': config.LOSS_DICE} # , 'CE': config.LOSS_CE
            , 'loss_weighted': {'Dice': True} # , 'CE': True
            , 'loss_mask': {'Dice':True}
            , 'loss_type': {'Dice': config.LOSS_VECTOR}
            , 'loss_combo': []
        }
        , 'others': {
            'epochs_timer': 10
        }
    }

    # Call the trainer
    trainer = train.Trainer(params)
    trainer.train()

# Step 2.1 - Set dataloader, model and metrics params (for validation)
else:
    params = {
        'PROJECT_DIR': PROJECT_DIR
        , 'exp_name': 'UNet3DASPP_WeightedMaskedDice_LossVector_Softmax_seed42'
    }
