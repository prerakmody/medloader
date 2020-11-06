import numpy as np
import tensorflow as tf

_EPSILON = tf.keras.backend.epsilon()

label_weights = tf.constant([0.1,1,3,1,3,3,1,1,2,2], dtype=tf.float32)

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
        y_pred = y_pred*label_mask
        
        # Calculate loss (over all pixels)
        y_true_label = y_true[:,:,:,:,label_id]
        y_pred_label = y_pred[:,:,:,:,label_id]
        if tf.math.reduce_sum(y_true_label) > 0:
            num = 2*tf.math.reduce_sum(y_true_label * y_pred_label)
            den = tf.math.reduce_sum(y_true_label + y_pred_label)
            loss_label = 1.0 - tf.reduce_mean(num/den)
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