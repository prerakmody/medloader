import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private" # to avoid large "Kernel Launch Time"

import tensorflow as tf
if len(tf.config.list_physical_devices('GPU')):
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

############################################################
#                    MODEL RELATED                         #
############################################################

MODEL_ATTENTIONUNET3D = 'AttentionUnet3D'
MODEL_UNET3D = 'UNet3D'
MODEL_UNET3DSMALL = 'UNet3DSmall'
MODEL_UNET3DSHALLOW = 'UNet3DShallow'
MODEL_UNET3DASPP = 'UNet3DASPP'

OPTIMIZER_ADAM = 'Adam'

MODEL_CHKPOINT_MAINFOLDER = '_models'
MODEL_CHKPOINT_NAME_FMT = 'ckpt_epoch{:03d}' 
MODEL_IMGCHKPOINT_NAME_FMT = 'img_ckpt_epoch{:03d}'
MODEL_LOGS_FOLDERNAME = 'logs' 
MODEL_IMGS_FOLDERNAME = 'images'

MODE_TRAIN = 'Train'
MODE_VAL = 'Val'
MODE_TEST = 'Test'

############################################################
#                      LOSSES RELATED                      #
############################################################
LOSS_DICE = 'Dice'
LOSS_FOCAL = 'Focal'
LOSS_CE = 'CE'
LOSS_NCC = 'NCC'
LOSS_SCALAR = 'scalar'
LOSS_VECTOR = 'vector'

############################################################
#                    DATALOADER                            #
############################################################
DATALOADER_MICCAI2015_TRAIN = 'train'
DATALOADER_MICCAI2015_TRAIN_ADD = 'train_additional'
DATALOADER_MICCAI2015 = 'HaN_MICCAI2015'