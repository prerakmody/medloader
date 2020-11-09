import pdb
import traceback
import tensorflow as tf

class NormalizeMinMaxSampler:

    def __init__(self, min_val, max_val, x_shape):
        self.name = 'NormalizeMinMaxSampler'
        self.min_val = tf.constant(min_val, dtype=tf.float32)
        self.max_val = tf.constant(max_val, dtype=tf.float32)
        self.x_shape = x_shape

    def execute(self, x, y, meta1, meta2):
        x_min = self.min_val
        x_max = self.max_val
        x = tf.math.maximum(x, tf.fill(dims=self.x_shape, value=x_min))
        x = tf.math.minimum(x, tf.fill(dims=self.x_shape, value=x_max))
        x = (x - x_min) / (x_max - x_min)
        return x,y,meta1,meta2

class Crop3D:

    def __init__(self, dims_3D):
        self.name = 'Crop3D'
        self.dims_3D = dims_3D
    
    def execute(self, x ,y ,meta1, meta2):
    
        midpoint_info = meta1[1:]
        w_start = midpoint_info[0] - self.dims_3D.MIDPOINT_EXTENSION_W_LEFT
        w_end = midpoint_info[0] + self.dims_3D.MIDPOINT_EXTENSION_W_RIGHT
        h_start = midpoint_info[1] - self.dims_3D.MIDPOINT_EXTENSION_H_FRONT
        h_end = midpoint_info[1] + self.dims_3D.MIDPOINT_EXTENSION_H_BACK
        d_start = midpoint_info[2] - self.dims_3D.MIDPOINT_EXTENSION_D_BOTTOM
        d_end = midpoint_info[2] + self.dims_3D.MIDPOINT_EXTENSION_D_TOP

        x = x[w_start:w_end, h_start:h_end, d_start:d_end]
        y = y[w_start:w_end, h_start:h_end, d_start:d_end]

        return x,y,meta1,meta2

class Rotate3D:

    def __init__(self):
        self.name = 'Rotate3D'

    def execute(self, x, y, meta1, meta2):
        """
        Rotates a 3D image along the z-axis by some random angle
        - Ref: https://www.tensorflow.org/api_docs/python/tf/image/rot90
        
        Parameters
        ----------
        x: tf.Tensor
            This is the 3D image of dtype=tf.int16 and shape=(H,W,C,1)
        y: tf.Tensor
            This is the 3D mask of dtype=tf.uint8 and shape=(H,W,C,Labels)
        meta1 = tf.Tensor
            This contains some indexing and meta info. Irrelevant to this function
        meta2 = tf.Tensor
            This contains some string information on patient identification. Irrelevant to this function
        """
        
        try:
            if tf.random.uniform([], minval=0, maxval=1, dtype=tf.dtypes.float32) <= 0.5:
                rotate_count =  tf.random.uniform([], minval=1, maxval=4, dtype=tf.dtypes.int32)
                
                # k=3 (rot=270) (anti-clockwise)
                if rotate_count == 3:
                    xtmp = tf.transpose(tf.reverse(x, [0]), [1,0,2,3])
                    ytmp = tf.transpose(tf.reverse(y, [0]), [1,0,2,3])

                # k = 1 (rot=90) (anti-clockwise)
                elif rotate_count == 1:
                    xtmp = tf.reverse(tf.transpose(x, [1,0,2,3]), [0])
                    ytmp = tf.reverse(tf.transpose(y, [1,0,2,3]), [0])

                # k=2 (rot=180) (clock-wise)
                elif rotate_count == 2:
                    xtmp = tf.reverse(x, [0,1])
                    ytmp = tf.reverse(y, [0,1])
                
                else:
                    xtmp = x
                    ytmp = y
                
                return xtmp, ytmp, meta1, meta2
                # return xtmp.read_value(), ytmp.read_value(), meta1, meta2
                
            else:
                return x, y, meta1, meta2
        except:
            traceback.print_exc()
            return x, y, meta1, meta2

class FilterByMask:

    def __init__(self, class_count, sampler_perc):
        self.class_count = class_count
        self.sampler_perc = sampler_perc

    def execute(self, x,y, meta1, meta2):
        # Step 1 - Calculate if non-background elements are present
        class_masks = meta1[-self.class_count:]
        non_background_presence = tf.math.greater(tf.math.reduce_sum(class_masks[1:]), 0)

        # Step 2.1 - if non-background elements are present, return True
        if tf.math.equal(non_background_presence, True):
            return True
        else:
            # Step 2.2.1 - With (self.sampler_perc)% ignore background-only grid  
            if tf.random.uniform([], minval=0, maxval=1, dtype=tf.dtypes.float32) < self.sampler_perc:
                return False
            else:
                return True

