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

if len(tf.config.list_physical_devices('GPU')):
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

import medloader.nnet.config as config
import medloader.nnet.tensorflow.utils as utils
import medloader.nnet.tensorflow.models as models


class Trainer:

    def __init__(self, params):

        # Init
        self.params = params

        # Print
        self._train_preprint()

        # Random Seeds
        self._set_seed()

        # Set the dataloaders
        self._set_dataloaders()

        # Set the model
        self._set_model()

        # Set Metrics
        self._set_metrics()

        # Other flags
        self.write_model_done = False
    
    def _train_preprint(self):
        print ('')
        print (' -------------- {}    ({})'.format(self.params['exp_name'], str(datetime.datetime.now())))
        
        print ('')
        print (' DATALOADER ')
        print (' ---------- ')
        print (' - dir_type: ', self.params['dataloader']['dir_type'])
        print (' -- batch_size: ', self.params['dataloader']['batch_size'])
        print (' -- single_sample: ', self.params['dataloader']['single_sample'])
        if self.params['dataloader']['single_sample']:
            print (' !!!!!!!!!!!!!!!!!!! SINGLE SAMPLE !!!!!!!!!!!!!!!!!!!')
            print ('')
        print ('  -- parallel_calls: ', self.params['dataloader']['parallel_calls'])

        print ('')
        print (' MODEL ')
        print (' ----- ')
        print (' - Model: ', str(self.params['model']['name']))
        print (' -- Activation: ', self.params['model']['activation'])
        print (' -- Kernel Reg: ', self.params['model']['kernel_reg'])
        print (' -- Model TBoard: ', self.params['model']['model_tboard'])
        print (' -- Profiler: ', self.params['model']['profiler']['profile'])
        if self.params['model']['profiler']['profile']:
            print (' ---- Profiler Epochs: ', self.params['model']['profiler']['epochs'])
            print (' ---- Step Per Epochs: ', self.params['model']['profiler']['steps_per_epoch'])
        print (' - Optimizer: ', str(self.params['model']['optimizer']))
        print (' -- Init LR: ', self.params['model']['init_lr'])
        print (' - Epochs: ', self.params['model']['epochs'])
        print (' -- Save: every {} epochs'.format(self.params['model']['epochs_save']))
        print (' -- Eval3D: every {} epochs '.format(self.params['model']['epochs_eval']))
        print (' -- Viz3D: every {} epochs '.format(self.params['model']['epochs_viz']))

        print ('')
        print (' METRICS ')
        print (' ------- ') 
        print (' - Eval: ', self.params['metrics']['metrics_eval'])
        print (' - Loss: ', self.params['metrics']['metrics_loss'])
        print (' -- Weighted Loss: ', self.params['metrics']['loss_weighted'])
        print (' -- Combo: ', self.params['metrics']['loss_combo'])

        print ('')
        print (' DEVOPS ')
        print (' ------ ')
        print (' - OS-PID: ', os.getpid())

        print ('')

    def _set_seed(self):
        np.random.seed(self.params['random_seed'])
        tf.random.set_seed(self.params['random_seed'])
    
    def _set_dataloaders(self):
        data_dir = self.params['dataloader']['data_dir']
        dir_type = self.params['dataloader']['dir_type']
        resampled = self.params['dataloader']['resampled']
        single_sample = self.params['dataloader']['single_sample']

        self.dataset_train = utils.get_dataloader_3D_train(data_dir, dir_type=dir_type
                                        , resampled=resampled, single_sample=single_sample)
        self.label_map = self.dataset_train.get_label_map()

    def _set_model(self):

        # Step 1 - Get class ids
        self.label_ids = self.dataset_train.datasets[0].LABEL_MAP.values()
        self.params['label_ids'] = self.label_ids
        class_count = len(self.label_ids)

        # Step 2 - Get model arch
        if self.params['model']['name'] == config.MODEL_UNET3D:
            self.model = models.ModelUNet3D(class_count=class_count, trainable=True, activation=self.params['model']['activation'])
        elif self.params['model']['name'] == config.MODEL_UNET3DSHALLOW:
            self.model = models.ModelUNet3DShallow(class_count=class_count, trainable=True, activation=self.params['model']['activation'])
        elif self.params['model']['name'] == config.MODEL_UNET3DSMALL:
            self.model = models.ModelUNet3DSmall(class_count=class_count, trainable=True, activation=self.params['model']['activation'])
        elif self.params['model']['name'] == config.MODEL_ATTENTIONUNET3D:
            self.model = models.AttentionUnet3D(class_count=class_count, trainable=True, activation=self.params['model']['activation'])
        
        # Step 3 - Get optimizer
        if self.params['model']['optimizer'] == config.OPTIMIZER_ADAM:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.params['model']['init_lr'])

        # Step 4 - Load model if needed
        load_epoch = -1
        epochs = self.params['model']['epochs']
        self.epoch_range = range(1,epochs+1)
        if self.params['model']['load_model']['load']:
            load_epoch = self.params['model']['load_model']['load_epoch']
            self.epoch_range = range(load_epoch+1, epochs)
            print ('')
            print (' - [Trainer] Loading model from epoch={} and training till epoch={}'.format(load_epoch, epochs))

            load_model_params = {'PROJECT_DIR': self.params['PROJECT_DIR'], 'exp_name': self.params['exp_name'], 'load_epoch': load_epoch
                                , 'optimizer':self.optimizer}
            utils.load_model(self.model, load_type=config.MODE_TRAIN, params=load_model_params)
            print (' - [train.py][train()] Model Loaded at epoch-{} !'.format(load_epoch))

    def _set_metrics(self):
        self.metrics = {}
        self.metrics[config.MODE_TRAIN] = utils.ModelMetrics(metric_type=config.MODE_TRAIN, params=self.params)
        self.metrics[config.MODE_TEST] = utils.ModelMetrics(metric_type=config.MODE_TEST, params=self.params)
    
    def _set_profiler(self, epoch, epoch_step):
        exp_name = self.params['exp_name']

        if self.params['model']['profiler']['profile']:
            if epoch in self.params['model']['profiler']['epochs']:
                if epoch_step == self.params['model']['profiler']['starting_step']:
                    self.logdir = Path(config.MODEL_CHKPOINT_MAINFOLDER).joinpath(exp_name, config.MODEL_LOGS_FOLDERNAME, 'profiler', str(epoch))
                    tf.profiler.experimental.start(str(self.logdir))
                    print (' - tf.profiler.experimental.start(logdir)')
                    print ('')
                elif epoch_step == self.params['model']['profiler']['starting_step'] + self.params['model']['profiler']['steps_per_epoch']:
                    print (' - tf.profiler.experimental.stop()')
                    tf.profiler.experimental.stop()
                    print ('')

    @tf.function
    def _train_loss(self, Y, y_predict, meta1, metrics_loss, loss_weighted, trainMetrics, label_ids):

        loss_vals = tf.constant(0.0, dtype=tf.float32)
        for metric_str in metrics_loss:
            mask = utils.get_mask(meta1[:,-len(label_ids):], Y)

            weighted = False
            if loss_weighted[metric_str]:
                weighted = True
            if metrics_loss[metric_str] in [config.LOSS_DICE, config.LOSS_CE, config.LOSS_FOCAL]:   
                loss_val_train, loss_labellist_train, loss_val_report, loss_labellist_report = trainMetrics.losses_obj[metric_str](Y, y_predict, mask, weighted=weighted)

                if metrics_loss[metric_str] in [config.LOSS_DICE]:
                    trainMetrics.update_metric_loss_labels(metric_str, loss_labellist_report)
                
                # loss_vals = tf.math.add(loss_vals, loss_val_train) # Averaged loss
                loss_vals = tf.math.add(loss_vals, loss_labellist_train) # Averaged loss by label

        return loss_vals

    @tf.function
    def _train_step(self, model, optimizer, X, Y, meta1, metrics_loss, loss_weighted, trainMetrics, label_ids):

        try:
            # Step 1 - Calculate loss and gradients
            with tf.GradientTape() as tape:
                t2 = tf.timestamp()
                y_predict = model(X, training=True)
                t2_ = tf.timestamp()

                loss_vals = self._train_loss(Y, y_predict, meta1, metrics_loss, loss_weighted, trainMetrics, label_ids)

            # tf.print(' - loss_Vals: ', loss_vals)
            t3 = tf.timestamp()
            all_vars = model.trainable_variables
            gradients = tape.gradient(loss_vals, all_vars) # dL/dW
            optimizer.apply_gradients(zip(gradients, all_vars))
            t3_ = tf.timestamp()

            return t2_-t2, t3-t2_, t3_-t3

        except:
            traceback.print_exc()
            return None, None, None
        
    def train(self):
        
        # PARAMS
        exp_name = self.params['exp_name']

        data_dir = self.params['dataloader']['data_dir']
        dir_type = self.params['dataloader']['dir_type']
        batch_size = self.params['dataloader']['batch_size']
        resampled = self.params['dataloader']['resampled']
        single_sample = self.params['dataloader']['single_sample']
        parallel_calls = self.params['dataloader']['parallel_calls']
        prefetch_batch = self.params['dataloader']['prefetch_batch']

        epochs_save = self.params['model']['epochs_save']
        epochs_viz = self.params['model']['epochs_viz']
        epochs_eval = self.params['model']['epochs_eval']

        metrics_loss = self.params['metrics']['metrics_loss']
        metrics_eval = self.params['metrics']['metrics_eval']
        loss_weighted = self.params['metrics']['loss_weighted']
        
        # VARS
        trainMetrics = self.metrics[config.MODE_TRAIN]
        params_save_model = {'PROJECT_DIR': self.params['PROJECT_DIR'], 'exp_name': exp_name, 'optimizer':self.optimizer}
        params_eval = {'PROJECT_DIR': self.params['PROJECT_DIR'], 'exp_name': exp_name
                            , 'eval_type': config.MODE_TRAIN, 'batch_size': batch_size}
        t_start_time = time.time()
        
        for epoch in self.epoch_range:

            try:
                
                # Epoch starter code
                trainMetrics.reset_metrics(self.params)
                utils.set_lr(epoch, self.optimizer)

                # Pretty print
                print ('')
                print (' ================== EPOCH:{} (LR={:3f}) =================='.format(epoch, self.optimizer.lr.numpy()))

                self.dataset_train = utils.get_dataloader_3D_train(data_dir, dir_type=dir_type
                                        , resampled=resampled, single_sample=single_sample
                                        , parallel_calls=parallel_calls)
                dataset_train_gen = self.dataset_train.generator().batch(batch_size).prefetch(prefetch_batch)
                epoch_step = 0
                with tqdm.tqdm(total=len(self.dataset_train), desc='') as pbar:

                    t1 = time.time()
                    for (X,Y,meta1,meta2) in dataset_train_gen:
                        t1_ = time.time()
                        
                        # Model Writing to tensorboard
                        if self.params['model']['model_tboard'] and self.write_model_done is False :
                            self.write_model_done = True 
                            utils.write_model_tboard(self.model, X, self.params)
                        
                        # Start/Stop Profiling (after dataloader is kicked off)
                        self._set_profiler(epoch, epoch_step)

                        # Calculate loss and gradients from them
                        time_predict, time_loss, time_backprop = self._train_step(self.model, self.optimizer, X, Y, meta1, metrics_loss, loss_weighted, trainMetrics, self.label_ids)

                        # Update metrics (time + eval + plots)
                        time_dataloader = t1_ - t1
                        trainMetrics.update_metrics_time(time_dataloader, time_predict, time_loss, time_backprop)
                                
                        # Update pbar
                        pbar.update(batch_size)
                        trainMetrics.update_pbar(pbar)

                        # End loop
                        epoch_step += batch_size
                        t1 = time.time() # reset dataloader time calculator

                # Model save
                if epoch % epochs_save == 0:
                    params_save_model['epoch'] = epoch
                    utils.save_model(self.model, params_save_model)
                
                # Eval on full 3D
                if epoch % epochs_eval == 0:
                    self.params['epoch'] = epoch
                    save=False
                    if epoch > 0 and epoch % epochs_viz == 0:
                        save=True

                    for metric_str in metrics_eval:
                        if metrics_eval[metric_str] in [config.LOSS_DICE]:
                            self.dataset_train_eval = utils.get_dataloader_3D_train_eval(data_dir, resampled=resampled, single_sample=single_sample)
                            params_eval['epoch'] = epoch
                            eval_avg, eval_labels_avg = utils.eval_3D(self.model, self.dataset_train_eval, params_eval, save=save)
                            trainMetrics.update_metric_eval_labels(metric_str, eval_labels_avg, do_average=True)
                
                # Test
                if epoch % epochs_eval == 0:
                    self.params['epoch'] = epoch
                    self._test()

                # Step 5.5 - Epochs summary
                eval_condition = epoch % epochs_eval == 0
                trainMetrics.write_epoch_summary(epoch, self.label_map, {'optimizer':self.optimizer}, eval_condition)
                if epoch > 0 and epoch % self.params['others']['epochs_timer'] == 0:
                        elapsed_seconds =  time.time() - t_start_time
                        print (' - Total time elapsed : {}'.format( str(datetime.timedelta(seconds=elapsed_seconds)) ))
                    
            except:
                utils.print_exp_name(exp_name + '-' + config.MODE_TEST, epoch)
                params_save_model['epoch'] = epoch
                utils.save_model(self.model, params_save_model)
                traceback.print_exc()
                pdb.set_trace()

    def _test(self):

        try:

            # Step 1.1 - Params
            exp_name = self.params['exp_name']
            
            data_dir = self.params['dataloader']['data_dir']
            resampled = self.params['dataloader']['resampled']
            single_sample = self.params['dataloader']['single_sample']

            epoch = self.params['epoch']

            metrics_eval = self.params['metrics']['metrics_eval']
            epochs_viz = self.params['model']['epochs_viz']
            batch_size = self.params['dataloader']['batch_size']
            
            # vars
            testMetrics = self.metrics[config.MODE_TEST]
            testMetrics.reset_metrics(self.params)
            params_eval = {'PROJECT_DIR': self.params['PROJECT_DIR'], 'exp_name': exp_name
                            , 'eval_type': config.MODE_TEST, 'batch_size': batch_size
                            , 'epoch':epoch}
                
            # Step 2 - Eval on full 3D
            save=False
            if epoch > 0 and epoch % epochs_viz == 0:
                save=True
            for metric_str in metrics_eval:
                if metrics_eval[metric_str] in [config.LOSS_DICE]:
                    self.dataset_test_eval = utils.get_dataloader_3D_test_eval(data_dir, resampled=resampled, single_sample=single_sample)
                    params_eval = {'eval_type': self.mode_test, 'epoch':epoch, 'batch_size': batch_size, 'exp_name': exp_name, 'MAIN_DIR': self.params['MAIN_DIR']}
                    eval_avg, eval_labels_avg = utils.eval_3D(self.model, self.dataset_test_eval, params_eval, save=save)
                    testMetrics.update_metric_eval_labels(metric_str, eval_labels_avg, do_average=True)

            testMetrics.write_epoch_summary(epoch, self.label_map, {}, True)

        except:
            traceback.print_exc()
            pdb.set_trace()
