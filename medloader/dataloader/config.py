import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

try:
    cfg = get_ipython().config
    IPYTHON_FLAG = True
except:
    IPYTHON_FLAG = False

################################### BODY REGIONS ###################################
HEAD_AND_NECK = 'HaN'
PROSTATE = 'Prostrate'
THORACIC = 'Thoracic'

################################### DIRNAME/FILENAME ###################################
DIRNAME_MAIN = '_data'

DIRNAME_PROCESSED = 'processed'
DIRNAME_RAW = 'raw'
DIRNAME_SAVE_3D = 'data_3D'
DIRNAME_SAVE_2D = 'data_2D'
DIRNAME_IMG_2D = 'img'
DIRNAME_MASK_2D = 'mask'

DIRNAMENAME_CACHED_TF = 'cached_tflow'

FILENAME_IMG_3D = 'img.mha'
FILENAME_MASK_3D = 'mask.mha'
FILENAME_MASK_TUMOR_3D = 'mask_tumor.mha'
FILENAME_IMG_RESAMPLED_3D = 'img_resampled.mha'
FILENAME_MASK_RESAMPLED_3D = 'mask_resampled.mha'

FILENAME_VOXEL_INFO = 'voxelinfo.json'

FILENAME_IMGNPY_2D = '{}_slice{}_img.npy'
FILENAME_MASKNPY_2D = '{}_slice{}_mask.npy'
STR_ORIG_IMG_NPY_2D = 'img.npy'
STR_ORIG_MASK_NPY_2D = 'mask.npy'

FILENAME_IMG_RESAMPLED_NPY_2D = '{}_slice{}_resampled_img.npy'
FILENAME_MASK_RESAMPLED_NPY_2D = '{}_slice{}_resampled_mask.npy'
STR_REPLACE_IMG_NPY_2D = 'resampled_img.npy'
STR_REPLACE_MASK_NPY_2D = 'resampled_mask.npy'

FILENAME_JSON_IMG = 'img.json'
FILENAME_JSON_MASK = 'mask.json'
FILENAME_JSON_IMG_RESAMPLED = 'img_resampled.json'
FILENAME_JSON_MASK_RESAMPLED = 'mask_resampled.json'
FILENAME_CSV_IMG = 'img.csv'
FILENAME_CSV_MASK = 'mask.csv'
FILENAME_CSV_IMG_RESAMPLED = 'img_resampled.csv'
FILENAME_CSV_MASK_RESAMPLED = 'mask_resampled.csv'

MODALITYNAME_CT = 'CT'
MODALITYNAME_RTSTRUCT = 'RTSTRUCT'
MODALITYNAME_REGISTRATION = 'REG'
MODALITY_RTDOSE = 'RTDOSE'
MODALITY_RTPLAN = 'RTPLAN'

KEYNAME_PIXEL_SPACING = 'pixel_spacing'
KEYNAME_ORIGIN = 'origin'
KEYNAME_SHAPE = 'shape'
KEYNAME_INTERCEPT = 'intercept'
KEYNAME_SLOPE = 'slope'
KEYNAME_ZVALS = 'z_vals'
KEYNAME_MEAN_BRAINSTEAM = 'mean_brainstem'
KEYNAME_OTHERS = 'others'
KEYNAME_INTERPOLATOR = 'interpolator'
KEYNAME_INTERPOLATOR_IMG = 'interpolator_img'
KEYNAME_INTERPOLATOR_MASK = 'interpolator_mask'

FILE_EXTENSION_CSV = '.csv'
FILE_EXTENSION_JSON = '.json'

################################### PROCESSING ###################################

DATAEXTRACTOR_WORKERS = 8

import numpy as np
import tensorflow as tf

DATATYPE_VOXEL_IMG = np.int16
DATATYPE_VOXEL_MASK = np.uint8

DATATYPE_NP_INT32 = np.int32

DATATYPE_TF_STRING = tf.string
DATATYPE_TF_UINT8 = tf.uint8
DATATYPE_TF_INT16 = tf.int16
DATATYPE_TF_INT32 = tf.int32
DATATYPE_TF_FLOAT32 = tf.float32

# in mm
VOXEL_RESO =  (0.8,0.8,2.5) # [(0.8,0.8,2.5), () , (1,1,1), (1,1,2)] 

TYPE_VOXEL_ORIGSHAPE = 'orig'
TYPE_VOXEL_RESAMPLED = 'resampled'

MASK_TYPE_ONEHOT = 'one_hot'
MASK_TYPE_COMBINED = 'combined'

HU_MIN = -400
HU_MAX = 1000

MIDPOINT_EXTENSION_PX_2D_MICCAI = 160

################################### HaN - MICCAI 2015 ###################################
MICCAI2015_H_START, MICCAI2015_H_END = 110, 430
MICCAI2015_W_START, MICCAI2015_W_END = 40, 360 
MICCAI2015_IMG_H = MICCAI2015_H_END - MICCAI2015_H_START # target=320
MICCAI2015_IMG_W = MICCAI2015_W_END - MICCAI2015_W_START

HaN_MICCAI2015 = {
    'LABEL_MAP' : {
        'Background':0
        , 'BrainStem':1 , 'Chiasm':2, 'Mandible':3
        , 'OpticNerve_L':4, 'OpticNerve_R':5
        , 'Parotid_L':6,'Parotid_R':7
        ,'Submandibular_L':8, 'Submandibular_R':9
    }
    , 'IGNORE_LABELS' : [0]
    , 'LABELID_BACKGROUND' : 0
    , 'LABEL_COLORS' : {
        0: [255,255,255,10]
        , 1:[0,110,254,255], 2: [225,128,128,255], 3:[254,0,128,255]
        , 4:[191,50,191,255], 5:[254,128,254,255]
        , 6: [182, 74, 74,255], 7:[128,128,0,255]
        , 8:[50,105,161,255], 9:[46,194,194,255]
        , 
    }
    , 'GRID_3D' : {
        'SIZE' : [96,96,96] # [[60,60,60], [96,96,96]]   
        , 'OVERLAP' : [20,20,20] # [[10,10,10], [20,20,20]]
        , 'SAMPLER_PERC': 0.95
        , 'RANDOM_SHIFT_MAX': 20
        , 'RANDOM_SHIFT_PERC': 0.5
    } 
}

################################### VIZ PARAMS ###################################
FIGSIZE = (15,15)