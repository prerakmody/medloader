import abc
import six

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

@six.add_metaclass(abc.ABCMeta)
class MedicalDataset:

    def __init__(self, name, data_dir):
        
        # Params
        self.name = name
        self.data_dir = data_dir

@six.add_metaclass(abc.ABCMeta)
class MedicalDatasetGenerator(MedicalDataset):

    def __init__(self, name, data_dir):
        super(MedicalDatasetGenerator, self).__init__(name=name, data_dir=data_dir)
        self.data = {}
        self.name = name
        self.data_dir = data_dir

    @abc.abstractmethod
    def generator(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _download(self):
        raise NotImplementedError

class ZipDataset:

    def __init__(self, datasets):
        self.datasets = datasets
        self.datasets_generators = []
        for dataset in self.datasets:
            self.datasets_generators.append(dataset.generator())
    
    def __len__(self):
        length = 0
        for dataset in self.datasets:
            length += len(dataset)
        
        return length

    def generator(self):
        return tf.data.experimental.sample_from_datasets(self.datasets_generators) #<_DirectedInterleaveDataset shapes: (<unknown>, <unknown>, <unknown>, <unknown>), types: (tf.float32, tf.float32, tf.int16, tf.string)>

    def get_subdataset(self, param_name):
        if type(param_name) == str:
            for dataset in self.datasets:
                if dataset.name == param_name:
                    return dataset
        else:
            print (' - [ERROR][ZipDataset] param_name needs to a str')
        
        return None 

class SampleDataset(tf.data.Dataset):

    def __new__(cls, num_samples):
        print ('__new__!!')
        return tf.data.Dataset.from_generator(cls._generator, output_types=tf.float32, args=(num_samples,))

    def __init__(self, num_samples):
        print ('__init__')

    def _generator(num_samples):
        for i in range(num_samples):
            yield (i)
    
if __name__ == "__main__":
    for each in SampleDataset(num_samples=10):
        print (each)    

