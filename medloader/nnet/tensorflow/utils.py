import os
import gc
import pdb
import time
import tqdm
import traceback
import numpy as np
from pathlib import Path
import tensorflow as tf

import medloader.dataloader.config as medconfig
import medloader.dataloader.tensorflow.augmentations as aug 
from medloader.dataloader.tensorflow.dataset import ZipDataset
from medloader.dataloader.tensorflow.han_miccai2015 import HaNMICCAI2015Dataset

import medloader.nnet.config as config
import medloader.nnet.tensorflow.losses as losses


############################################################
#                     DATA RELATED                         #
############################################################

def get_dataloader_3D_train(data_dir, dir_type=['train']
                    , dimension=3, grid=True, resampled=True, mask_type=medconfig.MASK_TYPE_ONEHOT
                    , transforms=[], filter_grid=True
                    , parallel_calls=None, deterministic=False
                    , patient_shuffle=True
                    , debug=False, single_sample=False):

    datasets = []
    for dir_type_ in dir_type:
        # Step 1 - Get dataset class
        dataset = HaNMICCAI2015Dataset(data_dir=data_dir, dir_type=dir_type_
                        , dimension=dimension, grid=grid, resampled=resampled, mask_type=mask_type
                        , transforms=transforms, filter_grid=filter_grid
                        , parallel_calls=parallel_calls, deterministic=deterministic
                        , patient_shuffle=patient_shuffle
                        , debug=debug, single_sample=single_sample)

        # Step 2 - Transforms
        x_shape_w = dataset.w_grid
        x_shape_h = dataset.h_grid
        x_shape_d = dataset.d_grid
        transforms = [
                aug.NormalizeMinMaxSampler(min_val=medconfig.HU_MIN, max_val=medconfig.HU_MAX, x_shape=(x_shape_h, x_shape_w, x_shape_d,1))
                , aug.Rotate3D()
            ]
        dataset.transforms = transforms

        # Step 3 - Filters (to eliminate/reduce background-only grids)
        if filter_grid:
            dataset.filter = aug.FilterByMask(len(dataset.LABEL_MAP), dataset.SAMPLER_PERC)

        datasets.append(dataset)

    # Step 4- Return
    return ZipDataset(datasets)
    
def get_dataloader_3D_train_eval(data_dir, dir_type='train'
                    , dimension=3, grid=True, resampled=True, mask_type=medconfig.MASK_TYPE_ONEHOT
                    , transforms=[], filter_grid=False
                    , parallel_calls=None, deterministic=True
                    , patient_shuffle=False
                    , debug=False, single_sample=False):
    
    datasets = []
    
    # Dataset 1
    if 1:

        # Step 1 - Get dataset class
        dir_type = 'train'
        dataset_han_miccai2015 = HaNMICCAI2015Dataset(data_dir=data_dir, dir_type=dir_type
                    , dimension=dimension, grid=grid, resampled=resampled, mask_type=mask_type
                    , transforms=transforms, filter_grid=filter_grid
                    , parallel_calls=parallel_calls, deterministic=deterministic
                    , patient_shuffle=patient_shuffle
                    , debug=debug, single_sample=single_sample)

        # Step 2 - Training transforms
        x_shape_w = dataset_han_miccai2015.w_grid
        x_shape_h = dataset_han_miccai2015.h_grid
        x_shape_d = dataset_han_miccai2015.d_grid
        transforms = [
                    aug.NormalizeMinMaxSampler(min_val=medconfig.HU_MIN, max_val=medconfig.HU_MAX, x_shape=(x_shape_h, x_shape_w, x_shape_d,1))
                ]
        dataset_han_miccai2015.transforms = transforms 

        # Step 3 - Training filters for background-only grids
        # None

        # Step 4 - Append to list
        datasets.append(dataset_han_miccai2015)
    
    dataset = ZipDataset(datasets)
    return dataset

def get_dataloader_3D_test_eval(data_dir, dir_type='test_offsite'
                    , dimension=3, grid=True, resampled=True, mask_type=medconfig.MASK_TYPE_ONEHOT
                    , transforms=[], filter_grid=False
                    , parallel_calls=None, deterministic=True
                    , patient_shuffle=False
                    , debug=False, single_sample=False):
                    
    datasets = []

    # Dataset 1
    if 1:

        # Step 1 - Get dataset class
        dir_type = 'test_offsite'
        dataset_han_miccai2015 = HaNMICCAI2015Dataset(data_dir=data_dir, dir_type=dir_type
                    , dimension=dimension, grid=grid, resampled=resampled, mask_type=mask_type
                    , transforms=transforms, filter_grid=filter_grid
                    , parallel_calls=parallel_calls, deterministic=deterministic
                    , patient_shuffle=patient_shuffle
                    , debug=debug, single_sample=single_sample)

        # Step 2 - Testing transforms
        x_shape_w = dataset_han_miccai2015.w_grid
        x_shape_h = dataset_han_miccai2015.h_grid
        x_shape_d = dataset_han_miccai2015.d_grid
        transforms = [
                    aug.NormalizeMinMaxSampler(min_val=medconfig.HU_MIN, max_val=medconfig.HU_MAX, x_shape=(x_shape_h, x_shape_w, x_shape_d,1))
                ]
        dataset_han_miccai2015.transforms = transforms

        # Step 3 - Testing filters for background-only grids
        # None

        # Step 4 - Append to list
        datasets.append(dataset_han_miccai2015)

    dataset = ZipDataset(datasets)
    return dataset


############################################################
#                    MODEL RELATED                         #
############################################################
def save_model(model, params):
    """
     - Ref: https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint
    """
    try:
        PROJECT_DIR = params['PROJECT_DIR']
        exp_name = params['exp_name']
        epoch = params['epoch']

        folder_name = config.MODEL_CHKPOINT_NAME_FMT.format(epoch)
        model_folder = Path(PROJECT_DIR).joinpath(config.MODEL_CHKPOINT_MAINFOLDER, exp_name, folder_name)
        model_folder.mkdir(parents=True, exist_ok=True)
        model_path = Path(model_folder).joinpath(folder_name)
        
        optimizer = params['optimizer']
        ckpt_obj = tf.train.Checkpoint(optimizer=optimizer, model=model)
        
        ckpt_obj.save(file_prefix=model_path)

    except:
        traceback.print_exc()
        pdb.set_trace()

def load_model(exp_name, model, epoch, params, load_type):
    """
     - Ref: https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint
    """
    try:
        PROJECT_DIR = params['PROJECT_DIR']
        exp_name = params['exp_name']
        load_epoch = params['load_epoch']

        folder_name = config.MODEL_CHKPOINT_NAME_FMT.format(load_epoch)
        model_folder = Path(PROJECT_DIR).joinpath(config.MODEL_CHKPOINT_MAINFOLDER, exp_name, folder_name)
        
        if load_type == config.MODE_TRAIN:
            if 'optimizer' in params:
                ckpt_obj = tf.train.Checkpoint(optimizer=params['optimizer'], model=model)
                ckpt_obj.restore(save_path=tf.train.latest_checkpoint(str(model_folder))).assert_existing_objects_matched()
                # ckpt_obj.restore(save_path=tf.train.latest_checkpoint(str(model_folder))).assert_consumed()
            else:
                print (' - [ERROR][utils.load_model] Optimizer not passed !')
                pdb.set_trace()

        elif load_type in [config.MODE_VAL, config.MODE_TEST]:
            ckpt_obj = tf.train.Checkpoint(model=model)
            ckpt_obj.restore(save_path=tf.train.latest_checkpoint(str(model_folder))).expect_partial()

        else:
            print (' - [ERROR][utils.load_model] It should not be here!')
            pdb.set_trace()
            # tf.keras.Model.load_weights
            # tf.train.list_variables(tf.train.latest_checkpoint(str(model_folder)))

    except:
        traceback.print_exc()
        pdb.set_trace()

def get_tensorboard_writer(exp_name, suffix):
    try:
        import tensorflow as tf

        logdir = Path(config.MODEL_CHKPOINT_MAINFOLDER).joinpath(exp_name, config.MODEL_LOGS_FOLDERNAME, suffix)
        writer = tf.summary.create_file_writer(str(logdir))
        return writer

    except:
        traceback.print_exc()
        pdb.set_trace()

def make_summary(fieldname, epoch, writer1=None, value1=None, writer2=None, value2=None):
    try:
        import tensorflow as tf

        if writer1 is not None and value1 is not None:
            with writer1.as_default():
                tf.summary.scalar(fieldname, value1, epoch)
                writer1.flush()
        if writer2 is not None and value2 is not None:
            with writer2.as_default():
                tf.summary.scalar(fieldname, value2, epoch)
                writer2.flush()
    except:
        traceback.print_exc()
        pdb.set_trace()

def write_model_tboard(model, X, params, suffix='model'):
    """
     - Ref:
        - https://www.tensorflow.org/api_docs/python/tf/summary/trace_on 
        - https://www.tensorflow.org/api_docs/python/tf/summary/trace_export
        - https://www.tensorflow.org/api_docs/python/tf/keras/utils/plot_model
        - https://stackoverflow.com/questions/56690089/how-to-graph-tf-keras-model-in-tensorflow-2-0
    """

    # Step 1 - Start trace
    tf.summary.trace_on(graph=True, profiler=False)

    # Step 2 - Perform operation
    _ = write_model_trace(model, X)

    # Step 3 - Export trace
    writer = get_tensorboard_writer(params['exp_name'], suffix)
    with writer.as_default():
        tf.summary.trace_export(name=model.name, step=0, profiler_outdir=None)

    # Step 4 - Save as .png
    # tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, expand_nested=True) # only works for the functional API. :(

@tf.function
def write_model_trace(model, X):
    return model(X)

def set_lr(epoch, optimizer):
    if epoch == 200:
        optimizer.lr.assign(0.0001)

@tf.function
def get_mask(mask, Y):
    mask = tf.expand_dims(tf.expand_dims(tf.expand_dims(mask, axis=1),axis=1),axis=1)
    mask = tf.tile(mask, multiples=[1,Y.shape[1],Y.shape[2],Y.shape[3],1])
    mask = tf.cast(mask, tf.float32)
    return mask

class ModelMetrics():
    
    def __init__(self, metric_type, params):
        self.label_ids = params['label_ids']
        self.metric_type = metric_type

        self.losses_obj = self.get_losses_obj(params)

        self.init_metrics(params)
        self.init_tboard_writers(params)
        self.reset_metrics(params)
    
    def get_losses_obj(self, params):
        losses_obj = {} 
        metrics_loss = params['metrics']['metrics_loss']
        for loss_key in metrics_loss:
            if config.LOSS_DICE == metrics_loss[loss_key]:
                losses_obj[loss_key] = losses.loss_dice_3d_tf_func
        
        return losses_obj
    
    def init_metrics(self, params):
        """
        These are metrics derived from tensorflows library
        """
        # Metrics for losses (during training for smaller grids)
        self.metrics_loss_obj = {}
        metrics_loss = params['metrics']['metrics_loss']
        for metric_key in metrics_loss:
            self.metrics_loss_obj[metric_key] = {}
            self.metrics_loss_obj[metric_key]['total'] = tf.keras.metrics.Mean(name='Avg{}-{}'.format(metric_key, self.metric_type))
            if metrics_loss[metric_key] in [config.LOSS_DICE, config.LOSS_CE]:
                for label_id in self.label_ids:
                    self.metrics_loss_obj[metric_key][label_id] = tf.keras.metrics.Mean(name='Avg{}-Label-{}-{}'.format(metric_key, label_id, self.metric_type))
        
        # Metrics for eval (for full 3D volume)
        self.metrics_eval_obj = {}
        metrics_eval = params['metrics']['metrics_eval']
        for metric_key in metrics_eval:
            self.metrics_eval_obj[metric_key] = {}
            self.metrics_eval_obj[metric_key]['total'] = tf.keras.metrics.Mean(name='Avg{}-{}'.format(metric_key, self.metric_type))
            if metrics_eval[metric_key] in [config.LOSS_DICE]:
                for label_id in self.label_ids:
                    self.metrics_eval_obj[metric_key][label_id] = tf.keras.metrics.Mean(name='Avg{}-Label-{}-{}'.format(metric_key, label_id, self.metric_type))

        # Time Metrics
        self.metric_time_dataloader     = tf.keras.metrics.Mean(name='AvgTime-Dataloader-{}'.format(self.metric_type))
        self.metric_time_model_predict  = tf.keras.metrics.Mean(name='AvgTime-ModelPredict-{}'.format(self.metric_type))
        self.metric_time_model_loss     = tf.keras.metrics.Mean(name='AvgTime-ModelLoss-{}'.format(self.metric_type))
        self.metric_time_model_backprop = tf.keras.metrics.Mean(name='AvgTime-ModelBackProp-{}'.format(self.metric_type))        
    
    def reset_metrics(self, params):

        # Metrics for losses (during training for smaller grids)
        metrics_loss = params['metrics']['metrics_loss']
        for metric_key in metrics_loss:
            self.metrics_loss_obj[metric_key]['total'].reset_states()
            if metrics_loss[metric_key] in [config.LOSS_DICE, config.LOSS_CE]:
                for label_id in self.label_ids:
                    self.metrics_loss_obj[metric_key][label_id].reset_states()

        # Metrics for eval (for full 3D volume)
        metrics_eval = params['metrics']['metrics_eval']
        for metric_key in metrics_eval:
            self.metrics_eval_obj[metric_key]['total'].reset_states()
            if metrics_eval[metric_key] in [config.LOSS_DICE]:
                for label_id in self.label_ids:
                    self.metrics_eval_obj[metric_key][label_id].reset_states()

        # Time Metrics
        self.metric_time_dataloader.reset_states()
        self.metric_time_model_predict.reset_states()
        self.metric_time_model_loss.reset_states()
        self.metric_time_model_backprop.reset_states()
    
    def init_tboard_writers(self, params):
        """
        These are tensorboard writer
        """
        # Writers for loss (during training for smaller grids)
        self.writers_loss_obj = {}
        metrics_loss = params['metrics']['metrics_loss']
        for metric_key in metrics_loss:
            self.writers_loss_obj[metric_key] = {}
            self.writers_loss_obj[metric_key]['total'] = get_tensorboard_writer(params['exp_name'], suffix=self.metric_type + '-Loss')
            if metrics_loss[metric_key] in [config.LOSS_DICE, config.LOSS_CE]:
                for label_id in self.label_ids:
                    self.writers_loss_obj[metric_key][label_id] = get_tensorboard_writer(params['exp_name'], suffix=self.metric_type + '-Loss-' + str(label_id))
        
        # Writers for eval (for full 3D volume)
        self.writers_eval_obj = {}
        metrics_eval = params['metrics']['metrics_eval']
        for metric_key in metrics_eval:
            self.writers_eval_obj[metric_key] = {}
            self.writers_eval_obj[metric_key]['total'] = get_tensorboard_writer(params['exp_name'], suffix=self.metric_type + '-Eval')
            if metrics_eval[metric_key] in [config.LOSS_DICE]:
                for label_id in self.label_ids:
                    self.writers_eval_obj[metric_key][label_id] = get_tensorboard_writer(params['exp_name'], suffix=self.metric_type + '-Eval-' + str(label_id))

        # Time and other writers
        self.writer_lr                  = get_tensorboard_writer(params['exp_name'], suffix='LR')
        self.writer_time_dataloader     = get_tensorboard_writer(params['exp_name'], suffix='Time-Dataloader')
        self.writer_time_model_predict  = get_tensorboard_writer(params['exp_name'], suffix='Time-Model-Predict')
        self.writer_time_model_loss     = get_tensorboard_writer(params['exp_name'], suffix='Time-Model-Loss')
        self.writer_time_model_backprop = get_tensorboard_writer(params['exp_name'], suffix='Time-Model-Backprop')

    def update_metrics_time(self, time_dataloader, time_predict, time_loss, time_backprop):
        self.metric_time_dataloader.update_state(time_dataloader)
        self.metric_time_model_predict.update_state(time_predict)
        self.metric_time_model_loss.update_state(time_loss)
        self.metric_time_model_backprop.update_state(time_backprop)  

    def update_metric_loss(self, metric_str, metric_val):
        # Metrics for losses (during training for smaller grids)
        self.metrics_loss_obj[metric_str]['total'].update_state(metric_val)
    
    @tf.function
    def update_metric_loss_labels(self, metric_str, metric_vals_labels):
        # Metrics for losses (during training for smaller grids)

        for label_id in self.label_ids:
            if metric_vals_labels[label_id] > 0:
                self.metrics_loss_obj[metric_str][label_id].update_state(metric_vals_labels[label_id])

    def update_metric_eval(self, metric_str, metric_val):
        # Metrics for eval (for full 3D volume)
        self.metrics_eval_obj[metric_str]['total'].update_state(metric_val)
        
    def update_metric_eval_labels(self, metric_str, metric_vals_labels, do_average=False):
        # Metrics for eval (for full 3D volume)

        metric_avg = []
        for label_id in self.label_ids:
            if metric_vals_labels[label_id] > 0:
                self.metrics_eval_obj[metric_str][label_id].update_state(metric_vals_labels[label_id])
                if do_average:
                    if label_id > 0:
                        metric_avg.append(metric_vals_labels[label_id])
        
        if do_average:
            if len(metric_avg):
                self.metrics_eval_obj[metric_str]['total'].update_state(np.mean(metric_avg))

    def write_epoch_summary(self, epoch, label_map, params=None, eval_condition=False):

        # Metrics for losses (during training for smaller grids)
        for metric_str in self.metrics_loss_obj:
            make_summary('Loss/{}'.format(metric_str), epoch, writer1=self.writers_loss_obj[metric_str]['total'], value1=self.metrics_loss_obj[metric_str]['total'].result())
            if len(self.metrics_loss_obj[metric_str]) > 1: # i.e. has label ids
                for label_id in self.label_ids:
                    label_name, _ = get_info_from_label_id(label_id, label_map)
                    make_summary('Loss/{}/{}'.format(metric_str, label_name), epoch, writer1=self.writers_loss_obj[metric_str][label_id], value1=self.metrics_loss_obj[metric_str][label_id].result())
        
        # Metrics for eval (for full 3D volume)
        if eval_condition:
            for metric_str in self.metrics_eval_obj:
                make_summary('Eval3D/{}'.format(metric_str), epoch, writer1=self.writers_eval_obj[metric_str]['total'], value1=self.metrics_eval_obj[metric_str]['total'].result())
                if len(self.metrics_eval_obj[metric_str]) > 1: # i.e. has label ids
                    for label_id in self.label_ids:
                        label_name, _ = get_info_from_label_id(label_id, label_map)
                        make_summary('Eval3D/{}/{}'.format(metric_str, label_name), epoch, writer1=self.writers_eval_obj[metric_str][label_id], value1=self.metrics_eval_obj[metric_str][label_id].result())

        # Time Metrics
        make_summary('Info/Time/Dataloader'   , epoch, writer1=self.writer_time_dataloader    , value1=self.metric_time_dataloader.result())
        make_summary('Info/Time/ModelPredict' , epoch, writer1=self.writer_time_model_predict , value1=self.metric_time_model_predict.result())
        make_summary('Info/Time/ModelLoss'    , epoch, writer1=self.writer_time_model_loss    , value1=self.metric_time_model_loss.result())
        make_summary('Info/Time/ModelBackProp', epoch, writer1=self.writer_time_model_backprop, value1=self.metric_time_model_backprop.result())
        
        # Learning Rate
        if params is not None:
            if 'optimizer' in params:
                make_summary('Info/LR', epoch, writer1=self.writer_lr, value1=params['optimizer'].lr)

    def update_pbar(self, pbar):
        desc_str = ''

        # Metrics for losses (during training for smaller grids)
        for metric_str in self.metrics_loss_obj:
            if len(desc_str): desc_str += ',' 

            metric_avg = []
            for label_id in self.label_ids:
                if label_id > 0:
                    metric_avg.append(self.metrics_loss_obj[metric_str][label_id].result().numpy())
            loss_text = '{}Loss:{:2f}'.format(metric_str, np.mean(metric_avg))
            desc_str += loss_text
        
        pbar.set_description(desc=desc_str, refresh=True)

def get_info_from_label_id(label_id, label_map, label_colors=None):
    """
    The label_id param has to be greater than 0
    """
    
    label_name = [label for label in label_map if label_map[label] == label_id]
    if len(label_name):
        label_name = label_name[0]
    else:
        label_name = None

    if label_colors is not None:
        label_color = np.array(label_colors[label_id])
        if np.any(label_color > 0):
            label_color = label_color/255.0
    else:
        label_color = None

    return label_name, label_color

def eval_3D_create_folder(epoch, params):
    folder_name = config.MODEL_CHKPOINT_NAME_FMT.format(epoch)
    model_folder_epoch_save = Path(params['PROJECT_DIR']).joinpath(config.MODEL_CHKPOINT_MAINFOLDER, params['exp_name'], folder_name, config.MODEL_IMGS_FOLDERNAME, params['eval_type'])
    model_folder_epoch_patches = Path(model_folder_epoch_save).joinpath('patches')
    model_folder_epoch_imgs = Path(model_folder_epoch_save).joinpath('imgs')
    Path(model_folder_epoch_patches).mkdir(parents=True, exist_ok=True)
    Path(model_folder_epoch_imgs).mkdir(parents=True, exist_ok=True)

    return model_folder_epoch_patches, model_folder_epoch_imgs

def eval_3D_finalize(patient_img, patient_gt, patient_pred
                        , patient_id_curr
                        , model_folder_epoch_imgs, model_folder_epoch_patches 
                        , spacing
                        , show=False, save=False):

    # Step 3.1.2 - Vizualize
    if show:
        pass

    # Step 3.1.3 - Save 3D grid to visualize in 3D Slicer (drag-and-drop mechanism)
    if save:
        import medloader.dataloader.utils as medutils
        medutils.write_nrrd(str(Path(model_folder_epoch_patches).joinpath('nrrd_' + patient_id_curr)) + '_img.nrrd' , patient_img[:,:,:,0], spacing)
        medutils.write_nrrd(str(Path(model_folder_epoch_patches).joinpath('nrrd_' + patient_id_curr)) + '_mask.nrrd', np.argmax(patient_gt, axis=3),spacing)
        medutils.write_nrrd(str(Path(model_folder_epoch_patches).joinpath('nrrd_' + patient_id_curr)) + '_maskpred.nrrd', np.argmax(patient_pred, axis=3),spacing)

def eval_3D(model, dataset_eval, params, show=False, save=False, verbose=False):
    
    try:

        # Step 0.1 - Extract params
        exp_name = params['exp_name']
        PROJECT_DIR = params['PROJECT_DIR']
        eval_type = params['eval_type']

        batch_size = params['batch_size']
        batch_size = 2
        epoch = params['epoch']
        
        # Step 0.2 - Init results array
        loss_list = []
        loss_labels_list = []
        patient_grid_count = {}
        if verbose: print (''); print (' --------------------- eval_3D({}) ---------------------'.format(eval_type))

        # Step 0.3 - Init temp variables
        patient_id_curr = None
        w_grid, h_grid, d_grid = None, None, None
        meta1_batch = None
        patient_gt = None
        patient_img = None
        patient_pred_overlap = None
        patient_pred_vals = None
        model_folder_epoch_patches = None
        model_folder_epoch_imgs = None
        if save:
            model_folder_epoch_patches, model_folder_epoch_imgs = eval_3D_create_folder(epoch, params)

        # Step 0.4 - Debug vars
        filename = Path(__file__).parts[-1]
        t0, t99 = None, None
        import psutil
        import humanize
        process = psutil.Process(os.getpid())

        # Step 1 - Loop over dataset_eval (which provides patients & grids in an ordered manner)
        pbar_desc_prefix = 'Eval3D_{}'.format(eval_type)
        with tqdm.tqdm(total=len(dataset_eval), desc=pbar_desc_prefix, leave=False) as pbar_eval:
            for (X,Y,meta1,meta2) in dataset_eval.generator().batch(batch_size):
                y_predict = model(X, training=False) # training=False sets dropout rate to 0.0 
                for batch_id in range(X.shape[0]):

                    # Step 2 - Get grid info
                    patient_id_running = meta2[batch_id].numpy().decode('utf-8')
                    if patient_id_running in patient_grid_count: patient_grid_count[patient_id_running] += 1
                    else: patient_grid_count[patient_id_running] = 1

                    meta1_batch = meta1[batch_id].numpy()
                    w_start, h_start, d_start = meta1_batch[1], meta1_batch[2], meta1_batch[3]
                    
                    # Step 3 - Check if its a new patient
                    if patient_id_running != patient_id_curr:

                        # Step 3.1 - Sort out old patient (patient_id_curr)
                        if patient_id_curr != None:
                            
                            # TESTING
                            # import matplotlib.pyplot as plt
                            # f, axarr = plt.subplots(1,3)
                            # axarr[0].imshow(patient_pred_overlap[:,:,0])
                            # axarr[1].imshow(patient_pred_overlap[:,:,40])
                            # axarr[2].imshow(patient_pred_overlap[:,:,76])
                            # plt.show()
                            # pdb.set_trace()

                            # Step 3.1.1 - Get stitched patient grid
                            if verbose: t0 = time.time()
                            patient_pred_overlap = np.expand_dims(patient_pred_overlap, -1)
                            patient_pred = patient_pred_vals/patient_pred_overlap
                            del patient_pred_vals
                            del patient_pred_overlap
                            gc.collect()
                            if verbose: print (' - [eval_3D()] Post-Process time: ', time.time() - t0,'s')

                            # Step 3.1.2 - Save/Visualize
                            if verbose: t0 = time.time()
                            spacing = np.array([meta1_batch[4], meta1_batch[5], meta1_batch[6]])/100.0
                            eval_3D_finalize(patient_img, patient_gt, patient_pred
                                , patient_id_curr
                                , model_folder_epoch_imgs, model_folder_epoch_patches 
                                , spacing
                                , show=show, save=save)
                            if verbose: print (' - [eval_3D()] Save as .nrrd time: ', time.time() - t0,'s')
                            
                            # Step 3.1.3 - Loss Calculation
                            if verbose: t0 = time.time()
                            loss_avg_val, loss_labels_val = losses.loss_dice_numpy(patient_gt, patient_pred)
                            if loss_avg_val != -1 and len(loss_labels_val):
                                loss_list.append(loss_avg_val)
                                loss_labels_list.append(loss_labels_val)
                            else:
                                print (' - [ERROR][eval_3D()] patient_id: ', patient_id_curr)
                            if verbose: print (' - [eval_3D()] Loss calculation time: ', time.time() - t0,'s')
                            if verbose: print (' - [eval_3D()] Total patient time: ', time.time() - t99,'s')
                            
                        # Step 3.2 - Create variables for new patient
                        if verbose: t99 = time.time()
                        patient_id_curr = patient_id_running
                        patient_scan_size = meta1_batch[7:10]
                        dataset_name = patient_id_curr.split('-')[0]
                        dataset_this = dataset_eval.get_subdataset(param_name=dataset_name)
                        w_grid, h_grid, d_grid = dataset_this.w_grid, dataset_this.h_grid, dataset_this.d_grid
                        patient_pred_size = list(patient_scan_size) + [len(dataset_this.LABEL_MAP)]
                        patient_pred_overlap = np.zeros(patient_scan_size, dtype=np.uint8)
                        patient_pred_vals = np.zeros(patient_pred_size, dtype=np.float32)
                        patient_gt = np.zeros(patient_pred_size, dtype=np.float32)
                        if save:
                            patient_img = np.zeros(list(patient_scan_size) + [1], dtype=np.int16)

                    # Step 4 - If not new patient anymore, fill up data
                    patient_pred_vals[w_start:w_start + w_grid, h_start:h_start+h_grid, d_start:d_start+d_grid] += y_predict[batch_id]
                    patient_pred_overlap[w_start:w_start + w_grid, h_start:h_start+h_grid, d_start:d_start+d_grid] += np.ones(y_predict[batch_id].shape[:-1], dtype=np.uint8)
                    patient_gt[w_start:w_start+w_grid, h_start:h_start+h_grid, d_start:d_start+d_grid] = Y[batch_id]
                    if save:
                        data_vol = X[batch_id]*(medconfig.HU_MAX - medconfig.HU_MIN) + medconfig.HU_MIN
                        patient_img[w_start:w_start+w_grid, h_start:h_start+h_grid, d_start:d_start+d_grid] = data_vol
        
                pbar_eval.update(batch_size)
                memory = pbar_desc_prefix + '[' + humanize.naturalsize(process.memory_info().rss) + ']'
                pbar_eval.set_description(desc=memory, refresh=True)

        # Step 3 - For last patient
        # Step 3.1.1 - Get stitched patient grid
        if verbose: t0 = time.time()
        patient_pred_overlap = np.expand_dims(patient_pred_overlap, -1)
        patient_pred = patient_pred_vals/patient_pred_overlap
        del patient_pred_vals
        del patient_pred_overlap
        gc.collect()
        if verbose: print (' - [eval_3D()] Post-Process time: ', time.time() - t0,'s')

        # Step 3.1.2 - Save/Visualize
        if verbose: t0 = time.time()
        spacing = np.array([meta1_batch[4], meta1_batch[5], meta1_batch[6]])/100.0
        eval_3D_finalize(patient_img, patient_gt, patient_pred
            , patient_id_curr
            , model_folder_epoch_imgs, model_folder_epoch_patches 
            , spacing
            , show=show, save=save)
        if verbose: print (' - [eval_3D()] Save as .nrrd time: ', time.time() - t0,'s')
        
        # Step 3.1.3 - Loss Calculation
        if verbose: t0 = time.time()
        loss_avg_val, loss_labels_val = losses.loss_dice_numpy(patient_gt, patient_pred)
        if loss_avg_val != -1 and len(loss_labels_val):
            loss_list.append(loss_avg_val)
            loss_labels_list.append(loss_labels_val)
        else:
            print (' - [ERROR][eval_3D()] patient_id: ', patient_id_curr)
        if verbose: print (' - [eval_3D()] Loss calculation time: ', time.time() - t0,'s')
        if verbose: print (' - [eval_3D()] Total patient time: ', time.time() - t99,'s')

        # Step 5 - Summarize
        loss_labels_avg = []
        loss_labels_list = np.array(loss_labels_list)
        for label_id in range(loss_labels_list.shape[1]): 
            tmp_vals = loss_labels_list[:,label_id]
            loss_labels_avg.append(np.mean(tmp_vals[tmp_vals > 0]))
        
        loss_avg = np.mean(loss_list)
        print (' - eval_type: ', eval_type)
        print (' - loss_labels_3D: ', ['%.4f' % each for each in loss_labels_avg])
        print (' - loss_3D: %.4f' % loss_avg)
        print (' - loss_3D (w/o bgd): %.4f' %  np.mean(loss_labels_avg[1:]))

        return loss_avg, {i:loss_labels_avg[i] for i in range(len(loss_labels_avg))}

    except:
        traceback.print_exc()
        pdb.set_trace()
        return -1, {} 

############################################################
#                      DEBUG RELATED                       #
############################################################
def print_exp_name(exp_name, epoch):
    print ('')
    print (' [ERROR] ========================= {}(epoch={}) ========================='.format(exp_name, epoch))
    print ('')