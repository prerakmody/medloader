import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pdb
import time
import traceback

import tensorflow as tf

class Transform:

    def __init__(self):
        self.name = 'transform'
    
    def executeDouble(self, x1,x2):
        time.sleep(0.01)
        return x1,x2
    
    def execute(self, x1):
        time.sleep(0.01)
        return x1
    
class SampleDataset:
    """
    A sample dataset to test tf.data.Dataset 
    """

    def __init__(self, double=False):
        self.double = double

    def generator(self, transforms, batch_size=1):
        if self.double:
            dataset = tf.data.Dataset.from_generator(self._generator2DDouble, output_types=(tf.float32, tf.float32), args=())
            # dataset = dataset.interleave(lambda x1,x2: dataset, block_length=1, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        else:
            dataset = tf.data.Dataset.from_generator(self._generator2D, output_types=(tf.float32), args=())
            # dataset = dataset.interleave(lambda x1: dataset, cycle_length=1, block_length=1) # [Works, but slow]
            # dataset = dataset.interleave(lambda x1: dataset, cycle_length=2, block_length=1) # [Produce duplicates]
            # dataset = dataset.interleave(lambda x1: dataset, cycle_length=3, block_length=1) # [Produce triplets]
            # dataset = dataset.interleave(lambda x1: dataset, cycle_length=1, block_length=2) # [Works, but slow]
            # dataset = dataset.interleave(lambda x1: dataset, cycle_length=1, block_length=8)

            # dataset = dataset.interleave(lambda x: dataset, cycle_length=2, block_length=16,num_parallel_calls=tf.data.experimental.AUTOTUNE)
            # dataset = dataset.interleave(lambda x: dataset, cycle_length=1, block_length=16,num_parallel_calls=tf.data.experimental.AUTOTUNE)
            # dataset = dataset.interleave(lambda x: dataset, cycle_length=1, block_length=32, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            # dataset = dataset.interleave(lambda x: dataset, cycle_length=2, block_length=32, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = dataset.map(lambda x: x, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        for transform in transforms:
            if self.double:
                dataset = dataset.map(transform.executeDouble, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            else:   
                dataset = dataset.map(transform.execute, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        # dataset = dataset.prefetch(1000)
        return dataset

    def _generator2D(self):
        for i in range(1000):
            time.sleep(0.04) # e.g. 0.04x32 = 1.28seconds
            yield i
    
    def _generator2DDouble(self):
        for i in range(1000):
            time.sleep(0.04) # e.g. 0.04x32 = 1.28seconds
            yield i,i

def get_sample_generator(transforms=[], batch_size=1, double=False):
    
    dataset = SampleDataset(double=double)

    if len(transforms) == 0:
        transforms = [
                Transform()
            ]
    dataset_generator = dataset.generator(
                                transforms=transforms
                                , batch_size=batch_size)

    return dataset, dataset_generator

def benchmark_model(X):
    time.sleep(0.1)

def benchmark(dataset, dataset_generator):

    print (' - [benchmark()]')
    t0 = time.time()
    iter_ = 1
    for X in dataset_generator:
        t1 = time.time()
        benchmark_model(X)
        t2 = time.time()
        print (' - iter_: ', iter_, 'Data Time: ', round(t1 - t0,5),'s || Model Time: ', round(t2-t1,2),'s')
        iter_ += 1
        t0 = time.time()

def benchmark_multiple(datasets_generator, dataset, batch_size=1): # avg=0.3sec
    print (' - [benchmark_multiple()]')
    t0 = time.time()
    for X in tf.data.experimental.sample_from_datasets(datasets_generator).batch(batch_size):
        t1 = time.time()
        benchmark_model(X)
        t2 = time.time()
        print (' - Data Time: ', round(t1 - t0,5),'s || Model Time: ', round(t2-t1,2),'s')
        t0 = time.time()

def benchmark_with_profiler(dataset, dataset_generator):
    """
     - Ref: https://www.tensorflow.org/guide/profiler#profiling_apis
    """
    if tf.__version__ in ['2.3.0']:
        with tf.profiler.experimental.Profile('logdir'):
            print (' - [benchmark()]')
            t0 = time.time()
            for i,X in enumerate(dataset_generator):
                t1 = time.time()
                benchmark_model(X)
                t2 = time.time()
                print (' - Data Time: ', round(t1 - t0,5),'s || Model Time: ', round(t2-t1,2),'s', '(',X.shape,')')
                # print (X.shape, Y.shape)
                if i > 10:
                    break
                t0 = time.time()

def main():
    dataset, dataset_generator = get_sample_generator()
    batchsize = 32

    # benchmark(dataset, dataset_generator.batch(batchsize))
    benchmark_with_profiler(dataset, dataset_generator.batch(32))
    # benchmark_multiple([dataset_generator], dataset, batch_size=32)
    # benchmark_multiple([dataset_generator, dataset_generator], dataset, batch_size=batchsize)