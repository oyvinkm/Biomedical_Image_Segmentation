
from builtins import object
from abc import ABCMeta, abstractmethod
import warnings
from collections import OrderedDict
from warnings import warn


class DataLoaderBase(object):
    def __init__(self, data, BATCH_SIZE, num_batches, number_of_threads_in_multithreaded=None):
        """
        Code is copied from https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/training/dataloading/dataset_loading.py
        

        Slim version of DataLoaderBase (which is now deprecated). Only provides very simple functionality.
        You must derive from this class to implement your own DataLoader. You must overrive self.generate_train_batch()
        If you use our MultiThreadedAugmenter you will need to also set and use number_of_threads_in_multithreaded. See
        multithreaded_dataloading in examples!
        :param data: will be stored in self._data. You can use it to generate your batches in self.generate_train_batch()
        :param batch_size: will be stored in self.batch_size for use in self.generate_train_batch()
        :param number_of_threads_in_multithreaded: will be stored in self.number_of_threads_in_multithreaded.
        None per default. If you wish to iterate over all your training data only once per epoch, you must coordinate
        your Dataloaders and you will need this information
        """
        __metaclass__ = ABCMeta
        self.number_of_threads_in_multithreaded = number_of_threads_in_multithreaded
        self._data = data
        self._num_batches = num_batches
        self.BATCH_SIZE = BATCH_SIZE
        self.thread_id = 0
        self._batches_generated = 0

    def set_thread_id(self, thread_id):
        self.thread_id = thread_id

    def __iter__(self):
        return self

    def __next__(self):
        
        return self.generate_train_batch()

    @abstractmethod
    def generate_train_batch(self):
        '''override this
        Generate your batch from self._data .Make sure you generate the correct batch size (self.BATCH_SIZE)
        '''
        pass