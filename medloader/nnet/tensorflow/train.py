import os
import sys
import gc
import pdb
import time
import tqdm
import pynvml
import datetime
import traceback
import numpy as np
import tensorflow as tf
from pathlib import Path

import medloader.nnet.config as config
import medloader.nnet.tensorflow.utils as utils
import medloader.nnet.tensorflow.models as models


class Trainer:

    def __init__(self, params):

        # Init
        self.params = params

        # Some vars
        self.mode_train = 'Train'
        self.mode_test = 'Test'

        # Random Seeds
        self.set_seed()

        # Set the dataloaders
        self.set_dataloaders()

        # Set the model
        self.set_model()

        # Set Metrics
        self.set_metrics()

        # Other flags
        self.write_flag_model = True

        # Print vars
        self.train_preprint()
        
    def set_seed(self):
        np.random.seed(self.params['random_seed'])
        tf.random.set_seed(self.params['random_seed'])
    
    def set_dataloaders(self):
        data_dir = self.params['dataloader']['data_dir']
        resampled = self.params['dataloader']['resampled']
        single_sample = self.params['dataloader']['single_sample']

        self.dataset_train = utils.get_dataloader_3D_train(data_dir, resampled=resampled, single_sample=single_sample)
        self.dataset_train_eval = utils.get_dataloader_3D_train_eval(data_dir, resampled=resampled, single_sample=single_sample)
        self.dataset_test_eval = utils.get_dataloader_3D_test_eval(data_dir, resampled=resampled, single_sample=single_sample)

    def set_model(self):

        # Step 1 - Get class ids
        self.label_ids = self.dataset_train.datasets[0].LABEL_MAP.values()
        self.params['label_ids'] = self.label_ids
        class_count = len(self.label_ids)

        # Step 2 - Get model arch
        if self.params['model']['name'] == config.MODEL_UNET3D:
            self.model = models.ModelUNet3D(class_count=class_count, trainable=True)
        elif self.params['model']['name'] == config.MODEL_UNET3DSMALL:
            self.model = models.ModelUNet3DSmall(class_count=class_count, trainable=True)
        elif self.params['model']['name'] == config.MODEL_ATTENTIONUNET3D:
            self.model = models.AttentionUnet3D(class_count=class_count, trainable=True)
        
        # Step 3 - Get optimizer
        if self.params['model']['optimizer'] == config.OPTIMIZER_ADAM:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.params['model']['lr'])

        # Step 4 - Load model if needed
        load_epoch = -1
        epochs = self.params['model']['epochs']
        self.epoch_range = range(1,epochs+1)
        if self.params['model']['load_model']['load']:
            load_epoch = self.params['model']['load_model']['load_epoch']
            self.epoch_range = range(load_epoch+1, epochs)
            print ('')
            print (' - [Trainer] Loading model from epoch={} and training till epoch={}'.format(load_epoch, epochs))

            utils.load_model(self.params['exp_name'], self.model, load_epoch, {'optimizer':self.optimizer}, load_type=self.mode_train)
            print (' - [train.py][train()] Model Loaded at epoch-{} !'.format(load_epoch))

    def set_metrics(self):
        self.metrics = {}
        self.metrics[self.mode_train] = utils.ModelMetrics(metric_type=self.mode_train, params=self.params)
        self.metrics[self.mode_test] = utils.ModelMetrics(metric_type=self.mode_test, params=self.params)
        
    def train_preprint(self):
        print ('')
        print (' -------------- {}    ({})'.format(self.params['exp_name'], str(datetime.datetime.now())))
        print (' - Loss: ', self.params['metrics']['metrics_loss'])
        print (' - Eval: ', self.params['metrics']['metrics_eval'])
        print (' - Model: ', str(self.model))
        print (' -- Kernel Reg: ', self.params['model']['kernel_reg'])
        print ('')
        print (' - Batch Size: ', self.params['dataloader']['batch_size'])
        print ('')
        print (' - Eval3D every {} epochs: '.format(self.params['model']['epochs_eval']))
        print (' - Viz3D every {} epochs: '.format(self.params['model']['epochs_viz']))
        print ('')
        print (' - OS-PID: ', os.getpid())
        print ('')
        if self.params['dataloader']['single_sample']:
            print (' !!!!!!!!!!!!!!!!!!! SINGLE SAMPLE !!!!!!!!!!!!!!!!!!!')
            print ('') 

    @tf.function
    def backprop(self, vars, gradients, optimizer):
        optimizer.apply_gradients(zip(gradients, vars))

    def train(self):
        
        # PARAMS
        exp_name = self.params['exp_name']

        metrics_loss = self.params['metrics']['metrics_loss']
        loss_weighted = self.params['metrics']['loss_weighted']
        
        epochs_save = self.params['model']['epochs_save']
        epochs_viz = self.params['model']['epochs_viz']
        epochs_eval = self.params['model']['epochs_eval']

        batch_size = self.params['dataloader']['batch_size']

        # VARS
        trainMetrics = self.metrics[self.mode_train]

        for epoch in self.epoch_range:
            try:
                
                # Epoch starter code
                trainMetrics.reset_metrics(self.params)
                utils.set_lr(epoch, self.optimizer)

                # Pretty print
                print ('')
                print (' ================== EPOCH:{} (LR={:3f}) =================='.format(epoch, self.optimizer.lr.numpy()))

                # Profiling
                if epoch == 1 and self.params['model']['profile']:
                    self.logdir = Path(config.MODEL_CHKPOINT_MAINFOLDER).joinpath(exp_name, config.MODEL_LOGS_FOLDERNAME, 'profiler')
                    tf.profiler.experimental.start(str(self.logdir))
                    print (' - tf.profiler.experimental.start(logdir)')

                with tqdm.tqdm(total=len(self.dataset_train), desc='') as pbar:

                    t1 = time.time()
                    loss_val = 0

                    for (X,Y,meta1,meta2) in self.dataset_train.generator().batch(batch_size):
                        t1_ = time.time()
                        
                        if self.write_flag_model and self.params['model']['tboard_arch']:
                            self.write_flag_model = False 
                            utils.write_model_tboard(self.model, X, self.params)

                        # Step 1 - Calculate loss and gradients from them
                        loss_vals = 0
                        with tf.GradientTape() as tape:
                                
                            t2 = time.time()
                            y_predict = self.model(X, training=True)
                            t2_ = time.time()
                            
                            # Step 1.1 - Get and add individual losses
                            
                            for metric_str in metrics_loss:
                                mask = utils.get_mask(meta1[:,-len(self.label_ids):], Y)
                                
                                weighted = False
                                if loss_weighted[metric_str]:
                                    weighted = True
                                if metrics_loss[metric_str] in [config.LOSS_DICE, config.LOSS_CE]:   
                                    loss_val_train, loss_labellist_train, loss_val_report, loss_labellist_report = trainMetrics.losses_obj[metric_str](Y, y_predict, mask, weighted=weighted)
                                    trainMetrics.update_metric_loss_labels(metric_str, loss_labellist_report, do_average=True)
                                    loss_vals += loss_val_train
                        
                            if len(self.model.losses) and self.params['model']['kernel_reg']:
                                loss_vals = tf.math.add(loss_vals, tf.math.add_n(self.model.losses))
                        
                        # Step 1.2 - Get gradients
                        t3 = time.time()
                        all_vars = self.model.trainable_variables
                        gradients = tape.gradient(loss_vals, all_vars) # dL/dW
                        self.backprop(all_vars, gradients, self.optimizer)
                        t3_ = time.time()

                        # Step 2.1 - Update metrics (time + eval + plots)
                        time_dataloader = t1_ - t1
                        time_predict = t2_ - t2
                        time_loss = t3 - t2_
                        time_backprop = t3_ - t3
                        trainMetrics.update_metrics_time(time_dataloader, time_predict, time_loss, time_backprop)
                                
                        # Step 3 - Update pbar
                        pbar.update(batch_size)
                        trainMetrics.update_pbar(pbar)

                        # Step 4 - End loop
                        t1 = time.time() # reset dataloader time calculator

                    # Profiling
                    if epoch == 1 and self.params['model']['profile']:
                        tf.profiler.experimental.stop()
                        print (' - tf.profiler.experimental.stop()')
                        print (' - Run the command `tensorboard --logdir={}`'.format(self.logdir))

                    # Step 5.1 - Model save
                    if epoch % epochs_save == 0:
                        utils.save_model(exp_name, self.model, epoch, {'optimizer':self.optimizer})
                    
                    
            except:
                utils.save_model(exp_name, self.model, epoch, {'optimizer':self.optimizer, 'MAIN_DIR': self.params['MAIN_DIR']})
                traceback.print_exc()
                pdb.set_trace()
