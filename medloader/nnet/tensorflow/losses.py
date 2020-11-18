import pdb
import numpy as np
import tensorflow as tf

np.set_printoptions(suppress=True)
_EPSILON = tf.keras.backend.epsilon()

# label_weights = tf.constant([0.1,1,3,1,3,3,1,1,2,2], dtype=tf.float32)
# label_weights = tf.constant([0.01, 2, 5, 1, 5, 5, 2, 2, 3, 3], dtype=tf.float32) # [0.0003, 0.07, 0.17, 0.03, 0.17, 0.17, 0.07, 0.07, 0.10, 0.10]
# label_weights = label_weights / tf.math.reduce_sum(label_weights) # nomalized 
label_weights      = tf.constant([0.1,1,3,1,3,3,1,1,2,2], dtype=tf.float32)

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
    :param y_true: [B, H, W, C, L]
    :param y_pred: [B, H, W, C, L] 
    - Ref: V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
    - Ref: https://gist.github.com/jeremyjordan/9ea3032a32909f71dd2ab35fe3bacc08#file-soft_dice_loss-py
    """
    loss_labels = []
    for label_id in range(y_pred.shape[-1]):

        # Prepare hmaps (calculate loss only if GT is present)    
        y_pred = y_pred*label_mask # particularly useful when you want to mask out some batch samples for a particular class_id 
        
        # Calculate loss (over all pixels)
        y_true_label = y_true[:,:,:,:,label_id]
        y_pred_label = y_pred[:,:,:,:,label_id]
        if tf.math.reduce_sum(y_true_label) > 0:
            num = 2*tf.math.reduce_sum(y_true_label * y_pred_label)
            den = tf.math.reduce_sum(y_true_label + y_pred_label)
            loss_label = 1.0 - tf.reduce_mean(num/den) # NB: E[1-x] = 1 - E[x]
        else:
            loss_label = tf.convert_to_tensor(0.0)

        loss_labels.append(loss_label)
    
    loss_labels = tf.convert_to_tensor(loss_labels)

    # Return results (weighted/non-weighted)
    loss_for_train = None
    loss_labels_for_train = None
    loss_labels_for_report = loss_labels
    loss_for_report = tf.math.reduce_mean(loss_labels[loss_labels>0])
    
    if weighted:
        loss_labels_for_train = loss_labels_for_report * label_weights
        loss_for_train = tf.math.reduce_mean(loss_labels_for_train[loss_labels_for_train>0])
    else:
        loss_labels_for_train = loss_labels_for_report
        loss_for_train = loss_for_report

    return loss_for_train, loss_labels_for_train, loss_for_report, loss_labels_for_report 

@tf.function
def loss_dice_3d_tf_func_v3(y_true, y_pred, label_mask, weighted=False):
    """
    Calcultes soft-DICE loss

    :param y_true: [B, H, W, C, L]
    :param y_pred: [B, H, W, C, L] 
    :param label_mask: [B,L] 
    - Ref: V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
    - Ref: https://gist.github.com/jeremyjordan/9ea3032a32909f71dd2ab35fe3bacc08#file-soft_dice_loss-py
    """
    loss_labels = []
    for label_id in range(y_pred.shape[-1]):
        
        # Calculate loss (over all pixels)
        if tf.math.reduce_sum(label_mask[:,label_id]) > 0:
            y_true_label = y_true[:,:,:,:,label_id]
            y_pred_label = y_pred[:,:,:,:,label_id] 
            num = 2*tf.math.reduce_sum(y_true_label * y_pred_label, axis=[1,2,3]) # [B]
            den = tf.math.reduce_sum(y_true_label + y_pred_label, axis=[1,2,3]) # [B]
            loss_label = num/den
        else:
            loss_label = tf.convert_to_tensor(tf.zeros(y_pred.shape[0])) # [B]

        loss_labels.append(loss_label) # [B,L]
    
    loss_labels = tf.transpose(tf.convert_to_tensor(loss_labels)) # [B,L]

    # Step 2 - Return results (weighted/non-weighted)
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

    return 1.0 - loss_for_train, 1.0 - loss_labels_for_train, 1.0 - loss_for_report, 1.0 - loss_labels_for_report 

# @tf.function
def loss_dice_3d_tf_func_v2(y_true, y_pred, label_mask, weighted=False):
    """
    Calcultes soft-DICE loss
    :param y_true: [B, H, W, C, L]
    :param y_pred: [B, H, W, C, L]
    :param label_mask: [B,L] 
    - Ref: V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
    - Ref: https://gist.github.com/jeremyjordan/9ea3032a32909f71dd2ab35fe3bacc08#file-soft_dice_loss-py
    """

    # Step 1 - Calculate soft-DICE loss
    loss_labels = []
    for label_id in range(y_pred.shape[-1]):
    
        # Calculate loss (over all voxels)
        if tf.math.reduce_sum(label_mask[:,label_id]) > 0: # calculate loss only if GT is present (in any batch sample)
            y_true_label = y_true[:,:,:,:,label_id]
            y_pred_label = y_pred[:,:,:,:,label_id] # [B,H,W,D]

            num = 2*tf.math.reduce_sum(y_true_label * y_pred_label, axis=[1,2,3])
            den = tf.math.reduce_sum(y_true_label + y_pred_label, axis=[1,2,3])
            loss_label = num/den # [B,1]
        else:
            loss_label = tf.convert_to_tensor(tf.zeros(y_pred.shape[0]))

        loss_labels.append(loss_label)
    
    loss_labels = tf.transpose(tf.convert_to_tensor(loss_labels)) # [B,L]

    
    # Step 2 - Return results (weighted/non-weighted)
    label_mask = tf.cast(label_mask, dtype=tf.float32)
    label_mask = tf.where(tf.math.greater(label_mask,0), label_mask, _EPSILON)
    loss_for_train = None
    loss_labels_for_train = None
    loss_labels_for_report = tf.math.reduce_sum(loss_labels,axis=0) / tf.math.reduce_sum(label_mask, axis=0)
    loss_for_report = tf.math.reduce_mean(tf.math.reduce_sum(loss_labels,axis=1) / tf.math.reduce_sum(label_mask, axis=1))
    
    if weighted:
        # Prepare Weights for the grids of this batch
        vol_sum_list = tf.math.reduce_sum(y_true, axis=[1,2,3])
        vol_sum_list_inv = tf.where(tf.math.greater(vol_sum_list, 0.0), tf.math.divide(1, vol_sum_list), 0)
        ratios = tf.math.divide(vol_sum_list_inv, tf.math.reduce_sum(vol_sum_list_inv, axis=[1], keepdims=True)) 
        loss_labels_w = loss_labels * ratios
        loss_labels_for_train = tf.math.reduce_sum(loss_labels_w,axis=0) / tf.math.reduce_sum(label_mask, axis=0) 
        loss_for_train = tf.math.reduce_mean(tf.math.reduce_sum(loss_labels_w,axis=1) / tf.math.reduce_sum(label_mask, axis=1))
        
        # for batch_id in range(vol_sum_list.shape[0]):
        #     print (' - ', np.array([each for each in zip(vol_sum_list[batch_id].numpy(), ratios[batch_id].numpy())]))
        # pdb.set_trace()
    else:
        loss_labels_for_train = loss_labels_for_report
        loss_for_train = loss_for_report

    return 1.0 - loss_for_train, 1.0 - loss_labels_for_train, 1.0 - loss_for_report, 1.0 - loss_labels_for_report 