import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private" # to avoid large "Kernel Launch Time"


############################################################
#                    MODEL RELATED                         #
############################################################

MODEL_ATTENTIONUNET3D = 'AttentionUnet3D'
MODEL_UNET3D = 'UNet3D'
MODEL_UNET3DSMALL = 'UNet3DSmall'
MODEL_UNET3DSHALLOW = 'UNet3DShallow'

OPTIMIZER_ADAM = 'Adam'

MODEL_CHKPOINT_MAINFOLDER = '_models'
MODEL_CHKPOINT_NAME_FMT = 'ckpt_epoch{:03d}' 
MODEL_IMGCHKPOINT_NAME_FMT = 'img_ckpt_epoch{:03d}'
MODEL_LOGS_FOLDERNAME = 'logs' 
MODEL_IMGS_FOLDERNAME = 'images'

LOSS_DICE = 'Dice'
LOSS_CE = 'CE'

############################################################
#                    DATALOADER                            #
############################################################
DATALOADER_MICCAI2015 = 'HaN_MICCAI2015'