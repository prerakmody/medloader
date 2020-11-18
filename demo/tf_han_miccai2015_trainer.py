import pdb
from pathlib import Path

import medloader.nnet.config as config
import medloader.nnet.tensorflow.train as train

# Step 1 - Set PROJECT_DIR
PROJECT_DIR = Path(__file__).parent.absolute().parent.absolute()

# Step 2 - Set dataloader, model and metrics params
params = {
    'PROJECT_DIR': PROJECT_DIR
    , 'random_seed': 60
    , 'exp_name': 'UNet3D_DoubleTrainv1_seed60'
    , 'dataloader': {
        'data_dir': Path(PROJECT_DIR).joinpath('_data')
        , 'dir_type': [config.DATALOADER_MICCAI2015_TRAIN, config.DATALOADER_MICCAI2015_TRAIN_ADD]
        , 'name': config.DATALOADER_MICCAI2015
        , 'resampled': True
        , 'single_sample': False # [WATCH OUT!]
        , 'batch_size': 2
        , 'parallel_calls': 2
        , 'prefetch_batch': 5
    }
    , 'model':{
        'name': config.MODEL_UNET3D # [config.MODEL_UNET3D, config.MODEL_UNET3DSMALL, config.MODEL_UNET3DSHALLOW]
        , 'activation': 'softmax' # ['softmax', 'sigmoid']
        , 'kernel_reg': False
        , 'optimizer': config.OPTIMIZER_ADAM
        , 'init_lr': 0.005
        , 'epochs':  1000
        , 'epochs_save': 20
        , 'epochs_eval': 40
        , 'epochs_viz': 300
        , 'load_model': {'load': False, 'load_epoch':-1}
        , 'model_tboard': False
        , 'profiler': {
                'profile': False
                , 'epochs': [1,2]
                , 'steps_per_epoch': 50
                , 'starting_step': 2
            }
    }
    , 'metrics':{
        'metrics_loss': {'Dice': config.LOSS_DICE}
        , 'metrics_eval': {'Dice': config.LOSS_DICE}
        , 'loss_weighted': {'Dice': True}
        , 'loss_combo': []
    }
}

# Call the trainer
trainer = train.Trainer(params)
trainer.train()