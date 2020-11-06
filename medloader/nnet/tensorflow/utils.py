import pdb
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

def get_dataloader_3D_train(data_dir, resampled=False, single_sample=False):
    
    debug = False
    datasets = []
    
    # Dataset 1
    if 1:

        # Step 1 - Get dataset class
        dir_type = 'train'
        dataset_han_miccai2015 = HaNMICCAI2015Dataset(data_dir=data_dir, dir_type=dir_type
                    , dimension=3, mask_type=medconfig.MASK_TYPE_ONEHOT, resampled=resampled
                    , patient_shuffle=False
                    , debug=debug, single_sample=single_sample)

        # Step 2 - Training transforms
        x_shape_w = dataset_han_miccai2015.w_grid
        x_shape_h = dataset_han_miccai2015.h_grid
        x_shape_d = dataset_han_miccai2015.d_grid
        transforms = [
                    aug.NormalizeMinMaxSampler(min_val=medconfig.HU_MIN, max_val=medconfig.HU_MAX, x_shape=(x_shape_h, x_shape_w, x_shape_d,1))
                    , aug.Rotate3D()
                ]
        dataset_han_miccai2015.transforms = transforms 

        # Step 3 - Training filters for background-only grids
        dataset_han_miccai2015.filter = aug.FilterByMask(len(dataset_han_miccai2015.LABEL_MAP), dataset_han_miccai2015.SAMPLER_PERC).execute

        # Step 4 - Append to list
        datasets.append(dataset_han_miccai2015)
    
    dataset = ZipDataset(datasets)
    return dataset

def get_dataloader_3D_train_eval(data_dir, resampled=False, single_sample=False, debug=False):

    datasets = []
    
    # Dataset 1
    if 1:

        # Step 1 - Get dataset class
        dir_type = 'train'
        dataset_han_miccai2015 = HaNMICCAI2015Dataset(data_dir=data_dir, dir_type=dir_type
                    , dimension=3, mask_type=medconfig.MASK_TYPE_ONEHOT, resampled=resampled
                    , patient_shuffle=False
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

def get_dataloader_3D_test_eval(data_dir, resampled=True, debug=False, single_sample=False):

    datasets = []

    # Dataset 1
    if 1:

        # Step 1 - Get dataset class
        dir_type = 'test_offsite'
        dataset_han_miccai2015 = HaNMICCAI2015Dataset(data_dir=data_dir, dir_type=dir_type
                        , dimension=3, mask_type=medconfig.MASK_TYPE_ONEHOT, resampled=resampled
                        , patient_shuffle=False
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
def save_model(exp_name, model, epoch, params):
    """
     - Ref: https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint
    """
    try:
        import tensorflow as tf

        folder_name = config.MODEL_CHKPOINT_NAME_FMT.format(epoch)
        model_folder = Path(params['MAIN_DIR']).joinpath(config.MODEL_CHKPOINT_MAINFOLDER, exp_name, folder_name)
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
        import tensorflow as tf

        folder_name = config.MODEL_CHKPOINT_NAME_FMT.format(epoch)
        model_folder = Path(params['MAIN_DIR']).joinpath(config.MODEL_CHKPOINT_MAINFOLDER, exp_name, folder_name)
        
        if len(params):
            optimizer = params['optimizer']
            ckpt_obj = tf.train.Checkpoint(optimizer=optimizer, model=model)
            if load_type == 'train':
                ckpt_obj.restore(save_path=tf.train.latest_checkpoint(str(model_folder))).assert_existing_objects_matched()
                # ckpt_obj.restore(save_path=tf.train.latest_checkpoint(str(model_folder))).assert_consumed()
            elif load_type in ['test', 'val']:
                ckpt_obj.restore(save_path=tf.train.latest_checkpoint(str(model_folder))).expect_partial()
        else:
            pass
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
    
    def update_metric_loss_labels(self, metric_str, metric_vals_labels, do_average=False):
        # Metrics for losses (during training for smaller grids)

        metric_avg = []
        for label_id in self.label_ids:
            if metric_vals_labels[label_id] > 0:
                self.metrics_loss_obj[metric_str][label_id].update_state(metric_vals_labels[label_id])
                if do_average:
                    if label_id > 0:
                        metric_avg.append(metric_vals_labels[label_id])
        
        if do_average:
            if len(metric_avg):
                self.metrics_loss_obj[metric_str]['total'].update_state(np.mean(metric_avg))

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

    def write_epoch_summary(self, epoch, label_map, params=None):

        # Metrics for losses (during training for smaller grids)
        for metric_str in self.metrics_loss_obj:
            make_summary('Loss/{}'.format(metric_str), epoch, writer1=self.writers_loss_obj[metric_str]['total'], value1=self.metrics_loss_obj[metric_str]['total'].result())
            if len(self.metrics_loss_obj[metric_str]) > 1: # i.e. has label ids
                for label_id in self.label_ids:
                    label_name, _ = get_info_from_label_id(label_id, label_map)
                    make_summary('Loss/{}/{}'.format(metric_str, label_name), epoch, writer1=self.writers_loss_obj[metric_str][label_id], value1=self.metrics_loss_obj[metric_str][label_id].result())
        
        # Metrics for eval (for full 3D volume)
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
            loss_text = '{}Loss:{:2f}'.format(metric_str, self.metrics_loss_obj[metric_str]['total'].result())
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