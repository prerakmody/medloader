# Import external libraries
import pdb
import numpy as np
import tensorflow as tf

_EPSILON = tf.keras.backend.epsilon()

############################################################
#                           UTILS                          #
############################################################

_EPSILON = tf.keras.backend.epsilon()

@tf.function
def get_mask(mask_1D, Y):
    # mask_1D: [[1,0,0,0, ...., 1]] - [B,L] something like this
    # Y : [B,H,W,D,L] 
    if 0:
        mask = tf.expand_dims(tf.expand_dims(tf.expand_dims(mask_1D, axis=1),axis=1),axis=1) # mask.shape=[B,1,1,1,L]
        mask = tf.tile(mask, multiples=[1,Y.shape[1],Y.shape[2],Y.shape[3],1]) # mask.shape = [B,H,W,D,L]
        mask = tf.cast(mask, tf.float32)
        return mask
    else:
        return mask_1D

############################################################
#                         DICE                             #
############################################################

def loss_dice_numpy(y_true, y_pred):
    """
    :param y_true: [H, W, D, L]
    :param y_pred: [H, W, D, L] 
    - Ref: V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
    - Ref: https://gist.github.com/jeremyjordan/9ea3032a32909f71dd2ab35fe3bacc08#file-soft_dice_loss-py
    """
    loss_labels = []
    for label_id in range(y_pred.shape[-1]):
        
        y_true_label = y_true[:,:,:,label_id]
        y_pred_label = y_pred[:,:,:,label_id]

        # Calculate loss (over all pixels)
        if np.sum(y_true_label) > 0:
            num = 2*np.sum(y_true_label * y_pred_label)
            den = np.sum(y_true_label + y_pred_label)
            loss_label = 1.0 - np.mean(num/den)
        else:
            loss_label = 0.0

        loss_labels.append(loss_label)
    
    loss_labels = np.array(loss_labels)
    loss = np.mean(loss_labels[loss_labels>0])
    return loss, loss_labels 

@tf.function
def loss_dice_3d_tf_func(y_true, y_pred, label_mask, weighted=False):
    """
    Calculates soft-DICE loss

    :param y_true: [B, H, W, C, L]
    :param y_pred: [B, H, W, C, L] 
    :param label_mask: [B,L] 
    - Ref: V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
    - Ref: https://gist.github.com/jeremyjordan/9ea3032a32909f71dd2ab35fe3bacc08#file-soft_dice_loss-py
    """
    print (' - [loss_dice_3d_tf_func()] Normalized Weights')
    label_weights = tf.constant([0.01, 2, 5, 1, 5, 5, 2, 2, 3, 3], dtype=tf.float32) # [0.0003, 0.07, 0.17, 0.03, 0.17, 0.17, 0.07, 0.07, 0.10, 0.10]
    label_weights = label_weights / tf.math.reduce_sum(label_weights) # nomalized

    dice_labels = []
    for label_id in range(y_pred.shape[-1]):
        
        # Calculate loss (over all pixels)
        if tf.math.reduce_sum(label_mask[:,label_id]) > 0:
            y_true_label = y_true[:,:,:,:,label_id]
            y_pred_label = y_pred[:,:,:,:,label_id] 
            num = 2*tf.math.reduce_sum(y_true_label * y_pred_label, axis=[1,2,3]) # [B]
            den = tf.math.reduce_sum(y_true_label + y_pred_label, axis=[1,2,3]) # [B]
            dice_label = num/den
        else:
            dice_label = tf.convert_to_tensor(tf.zeros(y_pred.shape[0])) # [B]

        dice_labels.append(dice_label) # [B,L]
    
    dice_labels = tf.transpose(tf.convert_to_tensor(dice_labels)) # [B,L]

    # Step 2 - Return results (weighted/non-weighted)
    label_mask = tf.cast(label_mask, dtype=tf.float32)
    label_mask = tf.where(tf.math.greater(label_mask,0), label_mask, _EPSILON)
    dice_for_train = None
    dice_labels_for_train = None
    dice_labels_for_report = tf.math.reduce_sum(dice_labels,axis=0) / tf.math.reduce_sum(label_mask, axis=0)
    dice_for_report = tf.math.reduce_mean(tf.math.reduce_sum(dice_labels,axis=1) / tf.math.reduce_sum(label_mask, axis=1))
    
    if weighted:
        dice_labels_w = dice_labels * label_weights
        dice_labels_for_train = tf.math.reduce_sum(dice_labels_w,axis=0) / tf.math.reduce_sum(label_mask, axis=0) 
        dice_for_train = tf.math.reduce_mean(tf.math.reduce_sum(dice_labels_w,axis=1) / tf.math.reduce_sum(label_mask, axis=1))
    else:
        dice_labels_for_train = dice_labels_for_report
        dice_for_train = dice_for_report

    return 1.0 - dice_for_train, 1.0 - dice_labels_for_train, 1.0 - dice_for_report, 1.0 - dice_labels_for_report 

############################################################
#                      CROSS ENTROPY                        #
############################################################

@tf.function
def loss_ce_3d_tf_func(y_true, y_pred, label_mask, weighted=False):
    """
    Calculates Cross Entropy

    :param y_true: [B, H, W, C, L]
    :param y_pred: [B, H, W, C, L] 
    - Additional: https://github.com/tqkhai2705/edge-detection/blob/master/Canny-TensorFlow.py
    """

    print (' - [loss_ce_3d_tf_func()] Normalized Weights')
    label_weights = tf.constant([0.01, 2, 5, 1, 5, 5, 2, 2, 3, 3], dtype=tf.float32) # [0.0003, 0.07, 0.17, 0.03, 0.17, 0.17, 0.07, 0.07, 0.10, 0.10]
    label_weights = label_weights / tf.math.reduce_sum(label_weights) # nomalized

    loss_labels = []
    for label_id in range(y_pred.shape[-1]):

        # Calculate loss (over all pixels)
        if tf.math.reduce_sum(label_mask[:,label_id]) > 0:
            y_true_label = y_true[:,:,:,:,label_id] # [B,H,W,D]
            y_pred_label = y_pred[:,:,:,:,label_id] 
            # ce_pixel = -1.0*y_true_label*tf.math.log(tf.math.maximum(y_pred_label, _EPSILON)) # focuses only on GT pixels
            ce_pixel = -1.0*y_true_label*tf.math.log(tf.math.maximum(y_pred_label, 0.1))/tf.math.log(10.0) # focuses only on GT pixels
            loss_label = tf.math.reduce_sum(ce_pixel, axis=[1,2,3]) / tf.math.reduce_sum(y_true_label, axis=[1,2,3]) # [B] average of CE over all GT voxels
        else:
            loss_label = tf.convert_to_tensor(tf.zeros(y_pred.shape[0]))

        loss_labels.append(loss_label) # [L,B]
    
    loss_labels = tf.transpose(tf.convert_to_tensor(loss_labels)) # [B,L]
    
    # Return results (weighted/non-weighted)
    label_mask = tf.cast(label_mask, dtype=tf.float32)
    label_mask = tf.where(tf.math.greater(label_mask,0), label_mask, _EPSILON)
    loss_for_train = None
    loss_labels_for_train = None
    loss_labels_for_report = tf.math.reduce_sum(loss_labels,axis=0) / tf.math.reduce_sum(label_mask, axis=0)
    loss_for_report = tf.math.reduce_mean(tf.math.reduce_sum(loss_labels,axis=1) / tf.math.reduce_sum(label_mask, axis=1))
    
    if weighted:
        loss_labels_w = loss_labels * label_weights
        loss_labels_for_train = tf.math.reduce_sum(loss_labels_w,axis=0) / tf.math.reduce_sum(label_mask, axis=0) 
        loss_for_train = tf.math.reduce_mean(tf.math.reduce_sum(loss_labels_w,axis=1) / tf.math.reduce_sum(label_mask, axis=1))
    else:
        loss_labels_for_train = loss_labels_for_report
        loss_for_train = loss_for_report

    return loss_for_train, loss_labels_for_train, loss_for_report, loss_labels_for_report