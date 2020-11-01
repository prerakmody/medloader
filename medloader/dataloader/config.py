import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

################################### BODY REGIONS ###################################
HEAD_AND_NECK = 'HaN'
PROSTATE = 'Prostrate'
THORACIC = 'Thoracic'

################################### DIRNAME/FILENAME ###################################

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
FILENAME_IMG_REG_AFFINE_3D = 'img_reg_affine.mha'
FILENAME_MASK_REG_AFFINE_3D = 'mask_reg_affine.mha'

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
KEYNAME_FRAMEOFREFUID = 'FrameOfReferenceUID'
KEYNAME_REGISTRATION_MODALITY = 'REG'
KEYNAME_UNREGISTERED = 'unregistered'
KEYNAME_REGISTERED = 'registered'
KEYNAME_REGISTRATION_PARAMS = 'registration_params'
KEYNAME_REPEAT_CTS = 'repeat_cts'
KEYNAME_PLANNING_CT = 'planning_ct'

# KEYNAME_PLANNING = 'planning'
# KEYNAME_REPEAT = 'repeat' 
# KEYNAME_PLANNING_REG = 'planning_registered'
# KEYNAME_PLANNING_REGPARAMS = 'planning_registration_params'

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
DATATYPE_TF_INT16 = tf.int16
DATATYPE_TF_INT32 = tf.int32
DATATYPE_TF_FLOAT32 = tf.float32

# in mm
VOXEL_RESO =  () # [(0.8,0.8,2.5), () , (1,1,1), (1,1,2)] 

TYPE_VOXEL_ORIGSHAPE = 'orig'
TYPE_VOXEL_RESAMPLED = 'resampled'

MASK_TYPE_ONEHOT = 'one_hot'
MASK_TYPE_COMBINED = 'combined'

HU_MIN = -400
HU_MAX = 1000

MIDPOINT_EXTENSION_PX_2D_MICCAI = 160

################################### IMAGE REGISTRATION ###################################

REG_CONFIG = {
        'transform' : {
            'affine': 'AffineTransform'
            , 'bspline': 'BSplineTransform'
        }
    }

################################### HaN - ConcensusGuidelines ###################################
HaN_ConcensusGuidelines = {
    'LABEL_MAP2' : {
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
        , 10: [120,60,0,255], 11: [151,108,64,255]
        , 12: [130,70,10,255], 13: [161,118,74,255]
        , 14: [86,86,5,255], 15: [128,28,0,255]
        , 16: [128,0,255,255]
        , 17: [0,128,0,255]
        , 18: [64,0,64,255], 19:[240,234,175,255]
        , 20: [128,0,255,255], 21:[255,128,0,255]
        , 22: [255,255,91,255], 23:[255,255,212,255]
        , 24: [226,121,121,255], 25: [255,10,10,255]
        , 26: [64,0,128,255], 27:[147,87,208,255]
        , 28: [255,255,0,255], 29: [102,102,255,255]
        , 30: [51,25,255,255], 31: [153,0,76,255]
        , 32: [255,102,102,255], 33: [255,204,204]
        , 36: [51,51,255,255]
        , 34: [0,204,0,255], 35: [76,153,0,255]
        , 37: [225,120,120,255], 38: [120,0,240,255]
    }
    , 'LABEL_MAP' : {
        'Background':0
        , 'BrainStem':1 , 'Chiasm':2, 'Mandible':3
        , 'OpticNerve_L':4, 'OpticNerve_R':5
        , 'Parotid_L':6,'Parotid_R':7
        ,'Submandibular_L':8, 'Submandibular_R':9
        , 'Eye_L':10, 'Eye_R':11
        , 'Lens_L':12, 'Lens_R':13
        , 'Cochlea_L':14, 'Cochlea_R':15
        , 'SpinalCord':16
        , 'Cricopharyngeus':17
        , 'Esophagus':18, 'Larynx':19
        , 'Pituitary': 20, 'Thyroid': 21
        , 'Lacrimal_L': 22, 'Lacrimal_R': 23
        , 'Arytenoid_L': 24, 'Arytenoid_R': 25
        , 'Carotid_L':26, 'Carotid_R':27
        , 'OralCavitiy': 28, 'Lips':29
        , 'Pharynx':30, 'Glottic':31
        , 'BrachialPlexus_L':32, 'BrachialPlexus_R':33
        , 'BuccalMucosa_L':34, 'BuccalMucosa_R':35
        # , 'Brain':36
        # , 'Chiasm+5mm':37
        # , 'Pituitary+5mm': 38
    }
}

################################### HaN - HOLLAND PTC ###################################
HaN_HollandPTC = {
    'LABEL_COLORS' : {
        0:[255,255,255,10]
        , 1:[0,0,64,255], 2:[157,193,230,255], 3:[128,128,255,255]
        , 5:[120,60,0,255], 6:[151,108,64,255], 7:[255,0,128,255], 8:[255,128,255,255]
        , 9:[255,255,91,255], 10:[255,255,212,255], 11:[225,128,128,255], 12:[128,0,255,255]
        , 13:[86,86,5,255], 14:[128,28,0,255]
        , 15:[64,0,128,255], 16:[147,87,208,255]
        , 17:[0,128,0,255], 18:[64,0,64,255]
        , 19:[192,98,192,255], 20:[240,234,175,255]
        , 22:[0,128,255,255], 23:[178,208,48,255], 24:[255,0,255,255]
        , 25:[255,0,128,255], 26:[255,255,0,255]
        , 27:[182,74,74,255], 28:[128,128,0,255]
        , 29:[50,105,161,255], 30:[46,194,194,255]
        , 31:[83,83,0,255]
        , 100: [254,0,128,255]
    }
    , 'LABEL_MAP' : {
        'Background':0
        , 'Brain':1 , 'BrainStem':2, 'SpinalCord':3, 'Cerebellum': 4
        , 'Eye_L':5, 'Eye_R':6, 'OpticNerve_L':7,'OpticNerve_R':8
        , 'Lacrimal_L':9, 'Lacrimal_R':10, 'Chiasm':11, 'Pituitary':12
        , 'Cochlea_L':13, 'Cochlea_R':14
        , 'Carotid_L':15, 'Carotid_R':16
        , 'Cricopharyngeus':17, 'Esophagus':18
        , 'Glottic_Area':19, 'Larynx':20, 'Thyroid': 21
        , 'Musc_Constrict_S': 22, 'Musc_Constrict_M': 23, 'Musc_Constrict_I': 24
        , 'Mandible':25, 'Oral_Cavity': 26
        , 'Parotid_L':27,'Parotid_R':28
        , 'Submandibular_L':29, 'Submandibular_R':30
        , 'Lips': 31
        , 'Irradiated Lymph Node': 100, 'Irradiated Region': 101, 'GTVp': 102, 'GTVn left': 103,  
    }
    , 'LABELID_BACKGROUND' : 0
    , 'GRID_3D' : {
        'SIZE' : [96,96,96]   
        , 'OVERLAP' : [20,20,20]
        , 'SAMPLER_PERC': 0.95
    } 
}

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
    , 'DIMS2': {
        '3D': {
            'MIDPOINT_EXTENSION_W_LEFT'   : 112
            ,'MIDPOINT_EXTENSION_W_RIGHT' : 112 # 224
            ,'MIDPOINT_EXTENSION_H_BACK'  : 50
            ,'MIDPOINT_EXTENSION_H_FRONT' : 174 # 224
            ,'MIDPOINT_EXTENSION_D_TOP'   : 35
            ,'MIDPOINT_EXTENSION_D_BOTTOM': 61  # 96
        }
        , '3D_SAMPLER': { # sample*new_dim - (sample-1)*overlap = current_dim
            'MIDPOINT_EXTENSION_W_LEFT'   : 120 
            ,'MIDPOINT_EXTENSION_W_RIGHT' : 120 # 240
            ,'MIDPOINT_EXTENSION_H_BACK'  : 66
            ,'MIDPOINT_EXTENSION_H_FRONT' : 174 # 240
            ,'MIDPOINT_EXTENSION_D_TOP'   : 42
            ,'MIDPOINT_EXTENSION_D_BOTTOM': 66  # 108 
            ,'GRIDS_SAMPLE_W' : 3
            ,'GRIDS_SAMPLE_H' : 3
            ,'GRIDS_SAMPLE_D' : 2
            ,'GRIDS_OVERLAP_W': 24  # 3*w' - (2*24) = 240; w'=96
            ,'GRIDS_OVERLAP_H': 24  # 3*w' - (2*24) = 240; h'=96
            ,'GRIDS_OVERLAP_D': 20  # 2*w' - (2*20) = 108; d'=64
        }
    }
    , 'DIMS3': {
        '3D': {
            'MIDPOINT_EXTENSION_W_LEFT'   : 112
            ,'MIDPOINT_EXTENSION_W_RIGHT' : 112 # 224
            ,'MIDPOINT_EXTENSION_H_BACK'  : 50
            ,'MIDPOINT_EXTENSION_H_FRONT' : 174 # 224
            ,'MIDPOINT_EXTENSION_D_TOP'   : 20
            ,'MIDPOINT_EXTENSION_D_BOTTOM': 76  # 96
        }
        , '3D_SAMPLER': { # sample*new_dim - (sample-1)*overlap = current_dim
            'MIDPOINT_EXTENSION_W_LEFT'   : 120 
            ,'MIDPOINT_EXTENSION_W_RIGHT' : 120 # 240
            ,'MIDPOINT_EXTENSION_H_BACK'  : 66
            ,'MIDPOINT_EXTENSION_H_FRONT' : 174 # 240
            ,'MIDPOINT_EXTENSION_D_TOP'   : 20
            ,'MIDPOINT_EXTENSION_D_BOTTOM': 88  # 108 
            ,'GRIDS_SAMPLE_W' : 3
            ,'GRIDS_SAMPLE_H' : 3
            ,'GRIDS_SAMPLE_D' : 2
            ,'GRIDS_OVERLAP_W': 24  # 3*w' - (2*24) = 240; w'=96
            ,'GRIDS_OVERLAP_H': 24  # 3*w' - (2*24) = 240; h'=96
            ,'GRIDS_OVERLAP_D': 20  # 2*w' - (2*20) = 108; d'=64
        }
    } 
    , 'DIMS': {
        '3D': {
            'MIDPOINT_EXTENSION_W_LEFT'   : 112
            ,'MIDPOINT_EXTENSION_W_RIGHT' : 112 # 224
            ,'MIDPOINT_EXTENSION_H_BACK'  : 50
            ,'MIDPOINT_EXTENSION_H_FRONT' : 174 # 224
            ,'MIDPOINT_EXTENSION_D_TOP'   : 20
            ,'MIDPOINT_EXTENSION_D_BOTTOM': 76  # 96
        }
        , '3D_SAMPLER': { # sample*new_dim - (sample-1)*overlap = current_dim
            'MIDPOINT_EXTENSION_W_LEFT'   : 120 
            ,'MIDPOINT_EXTENSION_W_RIGHT' : 120 # 240
            ,'MIDPOINT_EXTENSION_H_BACK'  : 66
            ,'MIDPOINT_EXTENSION_H_FRONT' : 174 # 240
            ,'MIDPOINT_EXTENSION_D_TOP'   : 20
            ,'MIDPOINT_EXTENSION_D_BOTTOM': 76  # 96 
            ,'GRIDS_SAMPLE_W' : 3
            ,'GRIDS_SAMPLE_H' : 3
            ,'GRIDS_SAMPLE_D' : 1
            ,'GRIDS_OVERLAP_W': 24  # 3*w' - (2*24) = 240; w'=96
            ,'GRIDS_OVERLAP_H': 24  # 3*w' - (2*24) = 240; h'=96
            ,'GRIDS_OVERLAP_D': 0   # 1*w' - (0*0)  = 96 ; d'=96
        }
    }
    , 'GRID_3D' : {
        'SIZE' : [96,96,96]   
        , 'OVERLAP' : [20,20,20]
        , 'SAMPLER_PERC': 0.95
    } 
}

################################### HaN - TCIA-CETUXIMAB 2013 ###################################
TCIACETUXIMAB_H_START, TCIACETUXIMAB_H_END = 40, 360  #70, 390
TCIACETUXIMAB_W_START, TCIACETUXIMAB_W_END = 110, 430 #80, 400 
TCIACETUXIMAB_IMG_H = TCIACETUXIMAB_H_END - TCIACETUXIMAB_H_START # target=320
TCIACETUXIMAB5_IMG_W = TCIACETUXIMAB_W_END - TCIACETUXIMAB_W_START

HaN_TCIACetuximab = {
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
    }
    , 'LABEL_MAP2' : {
        'Background':0
        , 'BrainStem':1 , 'Chiasm':2, 'Mandible':3
        , 'OpticNerve_L':4, 'OpticNerve_R':5
        , 'Parotid_L':6,'Parotid_R':7
        ,'Submandibular_L':8, 'Submandibular_R':9
        , 'Eye_L':10, 'Eye_R':11
        , 'Lens_L':12, 'Lens_R':13
        , 'Cochlea_L':14, 'Cochlea_R':15
        , 'SpinalCord':16
        , 'Cricopharyngeus':17
        , 'Esophagus':18, 'Larynx':19  
    }
}


################################### VIZ PARAMS ###################################
FIGSIZE = (15,15)