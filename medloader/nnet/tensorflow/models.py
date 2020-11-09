import pdb
import traceback
import tensorflow as tf

import medloader.nnet.config as config 

############################################################
#                        3D MODELS                         #
############################################################

class ConvBlock3D(tf.keras.layers.Layer):

    def __init__(self, filters, pool, kernel_size=(3,3,3), dropout=None, trainable=False, name=''):
        super(ConvBlock3D, self).__init__(name='{}_ConvBlock3D'.format(name))

        self.filters = filters
        self.pool = pool
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.trainable = trainable

        self.conv_layer = tf.keras.Sequential()
        for filter_id, filter_count in enumerate(self.filters):
            self.conv_layer.add(
                tf.keras.layers.Conv3D(filter_count, kernel_size, padding='same'
                        , activation='relu'
                        , kernel_regularizer=tf.keras.regularizers.l2(0.01)
                        , name='Conv_{}'.format(filter_id))
            )
            self.conv_layer.add(tf.keras.layers.BatchNormalization(trainable=self.trainable))
            if filter_id == 0 and self.dropout is not None:
                self.conv_layer.add(tf.keras.layers.Dropout(rate=self.dropout, name=self.name + '_DropOut'))
        
        self.pool_layer = tf.keras.layers.MaxPooling3D((2,2,2), strides=(2,2,2), trainable=self.trainable, name='Pool_{}'.format(self.name))
    
    def call(self, x):
        
        x = self.conv_layer(x)
        
        if self.pool is False:
            return x
        else:
            x_pool = self.pool_layer(x)
            return x, x_pool

class UpConvBlock3D(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size=(2,2,2), strides=(2, 2, 2), padding='same', trainable=False, name=''):
        super(UpConvBlock3D, self).__init__(name='{}_UpConv3D'.format(name))
        
        self.upconv_layer = tf.keras.Sequential()
        self.upconv_layer.add(tf.keras.layers.Conv3DTranspose(filters, kernel_size, strides, padding=padding
                        , activation='relu'
                        , kernel_regularizer=tf.keras.regularizers.l2(0.01)
                        , name='UpConv_{}'.format(self.name))
                    )
        # self.upconv_layer.add(tf.keras.layers.BatchNormalization(trainable=trainable))
    
    def call(self, x):
        return self.upconv_layer(x)

class ModelUNet3D(tf.keras.Model):

    def __init__(self, class_count, activation='softmax', trainable=False, verbose=False):
        super(ModelUNet3D, self).__init__(name='ModelUNet3D')

        self.verbose = verbose
        
        # dropout = [0, 0, 0, 0.2, 0.3, 0.2, 0, 0, 0]
        dropout = [0.1, 0.1, 0.1, 0.2, 0.3, 0.2, 0.1, 0.1, 0.1]
        filters = [[8,8], [16,16], [32,32], [64,64], [128,128]]

        if 1:
            self.convblock1 = ConvBlock3D(filters=filters[0]  , pool=True , dropout=dropout[0], trainable=trainable, name='Block1') # Dim/2 (e.g. 96/2=48)
            self.convblock2 = ConvBlock3D(filters=filters[1]  , pool=True , dropout=dropout[1], trainable=trainable, name='Block2') # Dim/4 (e.g. 96/4=24)
            self.convblock3 = ConvBlock3D(filters=filters[2]  , pool=True , dropout=dropout[2], trainable=trainable, name='Block3') # Dim/8 (e.g. 96/8=12)
            self.convblock4 = ConvBlock3D(filters=filters[3]  , pool=True , dropout=dropout[3], trainable=trainable, name='Block4') # Dim/16 (e.g. 96/16=6)

            self.convblock5 = ConvBlock3D(filters=filters[4]  , pool=False, dropout=dropout[4], trainable=trainable, name='Block5') # Dim/16 (e.g. 96/16=6)

            self.upconvblock6 = UpConvBlock3D(filters=filters[3][0], trainable=trainable, name='Block6_1')
            self.convblock6   = ConvBlock3D(filters=filters[4], pool=False, dropout=dropout[5], trainable=trainable, name='Block6_2') 
            
            self.upconvblock7 = UpConvBlock3D(filters=filters[2][0], trainable=trainable, name='Block7_1')
            self.convblock7   = ConvBlock3D(filters=filters[3], pool=False, dropout=dropout[6], trainable=trainable, name='Block7_2')
            
            self.upconvblock8 = UpConvBlock3D(filters=filters[1][0], trainable=trainable, name='Block8_1')
            self.convblock8   = ConvBlock3D(filters=filters[2], pool=False, dropout=dropout[7], trainable=trainable, name='Block8_2')
            
            self.upconvblock9 = UpConvBlock3D(filters=filters[0][0], trainable=trainable, name='Block9_1')
            self.convblock9   = ConvBlock3D(filters=[class_count,class_count], pool=False, dropout=dropout[8], trainable=trainable, name='Block9_2')
            
            self.convblock10  = tf.keras.layers.Conv3D(filters=class_count, kernel_size=(1,1,1), padding='same'
                                , activation=activation
                                , name='Block10')
    
    def call(self,x):
        try:
            conv1, pool1 = self.convblock1(x)
            conv2, pool2 = self.convblock2(pool1)
            conv3, pool3 = self.convblock3(pool2)
            conv4, pool4 = self.convblock4(pool3)

            conv5        = self.convblock5(pool4)
            
            up6   = self.upconvblock6(conv5)
            conv6 = self.convblock6(tf.concat([conv4, up6], axis=-1))

            up7   = self.upconvblock7(conv6)
            conv7 = self.convblock7(tf.concat([conv3, up7], axis=-1))

            up8   = self.upconvblock8(conv7)
            conv8 = self.convblock8(tf.concat([conv2, up8], axis=-1))

            up9   = self.upconvblock9(conv8)
            conv9 = self.convblock9(tf.concat([conv1, up9], axis=-1)
            )

            conv10 = self.convblock10(conv9)
            
            if self.verbose:
                print (' - x:', x.shape)
                print (' - conv1: ', conv1.shape, ' || pool1: ', pool1.shape)                
                print (' - conv2: ', conv2.shape, ' || pool2: ', pool2.shape)                
                print (' - conv3: ', conv3.shape, ' || pool3: ', pool3.shape)                
                print (' - conv4: ', conv4.shape, ' || pool4: ', pool4.shape)                
                print (' - conv5: ', conv5.shape)
                print (' - conv6: ', conv6.shape)                
                print (' - conv7: ', conv7.shape)                
                print (' - conv8: ', conv8.shape)                
                print (' - conv9: ', conv9.shape)
                print (' - conv10: ', conv10.shape)
                            
            return conv10

        except:
            traceback.print_exc()
            pdb.set_trace()

class ModelUNet3DSmall(tf.keras.Model):

    def __init__(self, class_count, activation='softmax', trainable=False, verbose=False):
        super(ModelUNet3DSmall, self).__init__(name='ModelUNet3DSmall')

        self.verbose = verbose
        
        # dropout = [0, 0, 0, 0.2, 0.3, 0.2, 0, 0, 0]
        dropout = [0.1, 0.1, 0.2, 0.1, 0.1]
        filters = [[8],[16],[32]]

        if 1:
            self.convblock1 = ConvBlock3D(filters=filters[0]  , pool=True , dropout=dropout[0], trainable=trainable, name='Block1') # Dim/2 (e.g. 96/2=48)
            self.convblock2 = ConvBlock3D(filters=filters[1]  , pool=True , dropout=dropout[1], trainable=trainable, name='Block2') # Dim/4 (e.g. 96/4=24)
            
            self.convblock3 = ConvBlock3D(filters=filters[2]  , pool=False , dropout=dropout[2], trainable=trainable, name='Block3') # Dim/8 (e.g. 96/8=12)
            
            self.upconvblock4 = UpConvBlock3D(filters=filters[1][0], trainable=trainable, name='Block4')
            self.convblock4   = ConvBlock3D(filters=filters[1], pool=False, dropout=dropout[3], trainable=trainable, name='Block4') 
            
            self.upconvblock5 = UpConvBlock3D(filters=filters[0][0], trainable=trainable, name='Block5')
            self.convblock5   = ConvBlock3D(filters=filters[0], pool=False, dropout=dropout[4], trainable=trainable, name='Block5')
            
            self.convblock6  = tf.keras.layers.Conv3D(filters=class_count, kernel_size=(1,1,1), padding='same'
                                , activation=activation
                                , name='Block6')
    
    def call(self,x):
        conv1, pool1 = self.convblock1(x)
        conv2, pool2 = self.convblock2(pool1)

        conv3 = self.convblock3(pool2)
        
        up4   = self.upconvblock4(conv3)
        conv4 = self.convblock4(tf.concat([conv2, up4], axis=-1))

        up5   = self.upconvblock5(conv4)
        conv5 = self.convblock5(tf.concat([conv1, up5], axis=-1))

        conv6 = self.convblock6(conv5)

        return conv6

class AttentionBlock3D(tf.keras.Model):

    def __init__(self, filters, kernel_size=(1,1,1), strides=(1,1,1), padding='same', trainable=False, name=''):
        super(AttentionBlock3D, self).__init__(name='{}_AttnBlock3D'.format(name))

        self.convblock_gate = tf.keras.Sequential()
        self.convblock_gate.add(tf.keras.layers.Conv3D(filters, kernel_size, strides, padding))
        self.convblock_gate.add(tf.keras.layers.BatchNormalization(trainable=trainable))

        self.convblock_query = tf.keras.Sequential()
        self.convblock_query.add(tf.keras.layers.Conv3D(filters, kernel_size, strides, padding))
        self.convblock_query.add(tf.keras.layers.BatchNormalization(trainable=trainable))

        self.psi = tf.keras.Sequential()
        self.psi.add(tf.keras.layers.ReLU())
        self.psi.add(tf.keras.layers.Conv3D(1, kernel_size, strides, padding))
        self.psi.add(tf.keras.layers.BatchNormalization(trainable=self.trainable))
        self.psi.add(tf.keras.layers.Activation(tf.keras.activations.sigmoid))
    
    def call(self, x_gate, x):
        g1 = self.convblock_gate(x_gate)
        x1 = self.convblock_query(x)
        psi = self.psi(g1 + x1)
        return x*psi 

class AttentionUnet3D(tf.keras.Model):
    """
     - Ref: https://github.com/LeeJunHyun/Image_Segmentation/blob/db34de21767859e035aee143c59954fa0d94bbcd/network.py#L276
    """

    def __init__(self, class_count, activation='softmax', trainable=False, verbose=False):
        super(AttentionUnet3D, self).__init__(name = 'AttentionUnet3D')

        self.verbose = verbose
        # dropout = [0, 0, 0, 0.2, 0.3, 0.2, 0, 0, 0]
        dropout = [0.1, 0.1, 0.1, 0.2, 0.3, 0.2, 0.1, 0.1, 0.1]
        filters = [[8,8], [16,16], [32,32], [64,64], [128,128]]

        self.attn_style = 'top2' # [all, top2, top1, bottom2]

        if 1:
            self.convblock1 = ConvBlock3D(filters=filters[0] , pool=True , dropout=dropout[0], trainable=trainable, name='Block1')
            self.convblock2 = ConvBlock3D(filters=filters[1] , pool=True , dropout=dropout[1], trainable=trainable, name='Block2')
            self.convblock3 = ConvBlock3D(filters=filters[2] , pool=True , dropout=dropout[2], trainable=trainable, name='Block3')
            self.convblock4 = ConvBlock3D(filters=filters[3] , pool=True , dropout=dropout[3], trainable=trainable, name='Block4')
            
            self.convblock5 = ConvBlock3D(filters=filters[4] , pool=False, dropout=dropout[4], trainable=trainable, name='Block5')

            self.upconvblock6 = UpConvBlock3D(filters=filters[3][0], name='Block6_1')
            if self.attn_style == 'all':
                self.attnblock6 = AttentionBlock3D(filters=filters[3][0], trainable=trainable, name='Block6_2')
            self.convblock6 = ConvBlock3D(filters=filters[3], pool=False, dropout=dropout[5], trainable=trainable, name='Block6_3')

            self.upconvblock7 = UpConvBlock3D(filters=filters[2][0], name='Block7_1')
            if self.attn_style == 'all':
                self.attnblock7 = AttentionBlock3D(filters=filters[2][0], trainable=trainable, name='Block7_2')
            self.convblock7 = ConvBlock3D(filters=filters[2], pool=False, dropout=dropout[6], trainable=trainable, name='Block7_3')

            self.upconvblock8 = UpConvBlock3D(filters=filters[1][0], name='Block8_1')
            if self.attn_style in ['top2']:
                self.attnblock8 = AttentionBlock3D(filters=filters[1][0], trainable=trainable, name='Block8_2')
            self.convblock8 = ConvBlock3D(filters=filters[1], pool=False, dropout=dropout[7], trainable=trainable, name='Block8_3')

            self.upconvblock9 = UpConvBlock3D(filters=class_count, name='Block9_1')
            if self.attn_style in ['top2', 'top1']:
                self.attnblock9 = AttentionBlock3D(filters=class_count, trainable=trainable, name='Block9_2')
            self.convblock9 = ConvBlock3D(filters=[class_count,class_count], pool=False, dropout=dropout[8], trainable=trainable, name='Block9_3')

            self.convblock10  = tf.keras.layers.Conv3D(filters=class_count, kernel_size=(1,1,1), padding='same'
                                , activation=activation , trainable=trainable
                                , name='Block10')
    
    def call(self,x):
            
        conv1, pool1 = self.convblock1(x)
        conv2, pool2 = self.convblock2(pool1)
        conv3, pool3 = self.convblock3(pool2)
        conv4, pool4 = self.convblock4(pool3)

        conv5        = self.convblock5(pool4)
        
        if self.attn_style == 'all':
            up6   = self.upconvblock6(conv5)
            attn6 = self.attnblock6(up6, conv4)
            conv6 = self.convblock6(tf.concat([attn6, up6], axis=-1))
            
            up7 = self.upconvblock7(conv6)
            attn7 = self.attnblock7(up7, conv3)
            conv7 = self.convblock7(tf.concat([attn7, up7], axis=-1))
            
            up8 = self.upconvblock8(conv7)
            attn8 = self.attnblock8(up8, conv2)
            conv8 = self.convblock8(tf.concat([attn8, up8], axis=-1))

            up9 = self.upconvblock9(conv8)
            attn9 = self.attnblock9(up9, conv1)
            conv9 = self.convblock9(tf.concat([attn9, up9], axis=-1))

        elif self.attn_style in ['top2', 'top1']:
            up6 = self.upconvblock6(conv5)
            conv6 = self.convblock6(tf.concat([conv4, up6], axis=-1))

            up7 = self.upconvblock7(conv6)
            conv7 = self.convblock7(tf.concat([conv3, up7], axis=-1))

            if self.attn_style in ['top2']:
                up8 = self.upconvblock8(conv7)
                attn8 = self.attnblock8(up8, conv2)
                conv8 = self.convblock8(tf.concat([attn8, up8], axis=-1))
            else:
                up8 = self.upconvblock8(conv7)
                conv8 = self.convblock8(tf.concat([conv2, up8], axis=-1))

            up9 = self.upconvblock9(conv8)
            attn9 = self.attnblock9(up9, conv1)
            conv9 = self.convblock9(tf.concat([attn9, up9], axis=-1))


        conv10 = self.convblock10(conv9)

        if self.verbose:
            print (' - x:', x.shape)
            print (' - conv1: ', conv1.shape, ' || pool1: ', pool1.shape)                
            print (' - conv2: ', conv2.shape, ' || pool2: ', pool2.shape)                
            print (' - conv3: ', conv3.shape, ' || pool3: ', pool3.shape)                
            print (' - conv4: ', conv4.shape, ' || pool4: ', pool4.shape)                
            print (' - conv5: ', conv5.shape)
            print (' - conv6: ', conv6.shape)                
            print (' - conv7: ', conv7.shape)                
            print (' - conv8: ', conv8.shape)                
            print (' - conv9: ', conv9.shape)
            print (' - conv10: ', conv10.shape)
                        
        return conv10
    