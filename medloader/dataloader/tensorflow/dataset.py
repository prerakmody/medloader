import abc
import six

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

class ZipDataset:

    def __init__(self, datasets):
        self.datasets = datasets
        self.datasets_generators = []
    
    def __len__(self):
        length = 0
        for dataset in self.datasets:
            length += len(dataset)
        
        return length

    def generator(self):
        for dataset in self.datasets:
            self.datasets_generators.append(dataset.generator())
        return tf.data.experimental.sample_from_datasets(self.datasets_generators) #<_DirectedInterleaveDataset shapes: (<unknown>, <unknown>, <unknown>, <unknown>), types: (tf.float32, tf.float32, tf.int16, tf.string)>

    def get_subdataset(self, param_name):
        if type(param_name) == str:
            for dataset in self.datasets:
                if dataset.name == param_name:
                    return dataset
        else:
            print (' - [ERROR][ZipDataset] param_name needs to a str')
        
        return None 
    
    def get_label_map(self):
        return self.datasets[0].LABEL_MAP
