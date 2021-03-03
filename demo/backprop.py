# Ref: https://towardsdatascience.com/backpropagation-in-a-convolutional-layer-24c8d64d8509
## 1D conv with 2 channels

import os
import pdb
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
if len(tf.config.list_physical_devices('GPU')):
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)


with tf.GradientTape(persistent=True, watch_accessed_variables=True) as tape:
    x = tf.ones([4,2]) # 2 channels with 4 inputs each (think of a timeseries with values for 4 timestamps)
    weights = tf.random.normal([2,2]) # 2 weights for each of the 2 channels
    mask = tf.constant([1.0, 0.0]) # masking to show what happens when one does not consider a particular channel (due to lack of data)
    tape.watch(x)
    tape.watch(weights)
    tape.watch(mask)

    y11 = tf.tensordot(x[0:2,0], weights[:,0], axes=1)
    y21 = tf.tensordot(x[1:3,0], weights[:,0], axes=1)
    y31 = tf.tensordot(x[2:4,0], weights[:,0], axes=1)
    y12 = tf.tensordot(x[0:2,1], weights[:,1], axes=1)
    y22 = tf.tensordot(x[1:3,1], weights[:,1], axes=1)
    y32 = tf.tensordot(x[2:4,1], weights[:,1], axes=1)

    y = tf.convert_to_tensor([[y11,y12],[y21,y22],[y31,y32]]) # shape=[3,2]
    ytrue = tf.ones(([3,2])) # random truth value
    loss = (ytrue - y)*mask
    loss1 = tf.math.reduce_mean(loss[loss != 0.0]) # avg (full), shape=[1]
    loss2 = tf.math.reduce_mean(loss, axis=0) # avg (across labels only), shape = [1,2]

print (' - loss1: ', loss1)
print (' - loss2: ', loss2)

gradients1 = tape.gradient(loss1, [weights])
gradients2 = tape.gradient(loss2, [weights])
print ('=================')
print (' - gradients1: ', gradients1)
print (' - gradients2: ', gradients2)

print ('\n =============================================================== \n')

with tf.GradientTape(persistent=True, watch_accessed_variables=True) as tape:
    x = tf.ones([2,4,2]) # 2 channels with 4 inputs each (think of a timeseries with values for 4 timestamps)
    weights = tf.random.normal([2,2]) # 2 weights for each of the 2 channels
    mask = tf.constant([[1.0, 0.0], [0.0, 1.0]]) # masking to show what happens when one does not consider a particular channel (due to lack of data)
    tape.watch(x)
    tape.watch(weights)
    tape.watch(mask)

    
    y11 = tf.tensordot(x[:,0:2,0], weights[:,0], axes=1)
    y21 = tf.tensordot(x[:,1:3,0], weights[:,0], axes=1)
    y31 = tf.tensordot(x[:,2:4,0], weights[:,0], axes=1)
    y12 = tf.tensordot(x[:,0:2,1], weights[:,1], axes=1)
    y22 = tf.tensordot(x[:,1:3,1], weights[:,1], axes=1)
    y32 = tf.tensordot(x[:,2:4,1], weights[:,1], axes=1)

    y = tf.transpose(tf.convert_to_tensor([[y11,y12],[y21,y22],[y31,y32]]), [1,0,2]) # shape=[2,3,2]
    ytrue = tf.ones(([2,3,2])) # random truth value [batch, values, channel]
    mask = tf.expand_dims(mask, axis=1)
    loss = (ytrue - y)*mask
    loss1 = tf.math.reduce_mean(loss[loss != 0.0]) # avg (full), shape=[1]
    loss2 = tf.math.reduce_sum(loss, axis=[1,2]) / tf.math.reduce_sum(mask, axis=[1,2]) # avg (across labels only), shape = [1,2]

print (' - loss1: ', loss1)
print (' - loss2: ', loss2)
gradienty1 = tape.gradient(loss1, [y])
gradienty2 = tape.gradient(loss2, [y])
gradients1 = tape.gradient(loss1, [weights])
gradients2 = tape.gradient(loss2, [weights])
print (' - gradient_y: ', gradienty1)
print (' - gradient_y: ', gradienty2)
print (' - gradients1: ', gradients1)
print (' - gradients2: ', gradients2)