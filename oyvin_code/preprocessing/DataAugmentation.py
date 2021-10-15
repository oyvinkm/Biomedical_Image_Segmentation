import abc
from warnings import warn
from typing import Union, Tuple
import numpy as np


class AbstractTransform(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self, **data_dict):
        raise NotImplementedError("Abstract, so implement")

    def __repr__(self):
        ret_str = str(type(self).__name__) + "( " + ", ".join(
            [key + " = " + repr(val) for key, val in self.__dict__.items()]) + " )"
        return ret_str

class AddGaussianNoise(AbstractTransform):
    def __init__(self, noise_variance=(0, 0.1), p_per_sample=1, p_per_channel : float = 1, per_channel:bool=False, data_key='data'):
        self.noise_variance = noise_variance
        self.p_per_sample = p_per_sample
        self.p_per_channnel = p_per_channel
        self.per_channel = per_channel
        self.data_key = data_key

        def __call__(self, **data_dict):
            for b in range(len(data_dict[self.data_key])):
                if np.random.uniform() < self.p_per_sample:
                    data_dict[self.data_key][b] = augment_gaussian_noise(data_dict[self.data_key][b], self.noise_variance, self.p_per_channel, self.per_channel)
                    
            return data_dict




def augment_gaussian_noise(data_sample: np.ndarray, noise_variance: Tuple[float, float] = (0, 0.1),
                           p_per_channel: float = 1, per_channel: bool = False) -> np.ndarray:
    if not per_channel:
        variance = noise_variance[0] if noise_variance[0] == noise_variance[1] else \
            np.random.uniform(noise_variance[0], noise_variance[1])
    else:
        variance = None
    for c in range(data_sample.shape[0]):
        if np.random.uniform() < p_per_channel:
            variance_here = variance if variance is not None else \
                noise_variance[0] if noise_variance[0] == noise_variance[1] else \
                    np.random.uniform(noise_variance[0], noise_variance[1])
            data_sample[c] = data_sample[c] + np.random.normal(0.0, variance_here, size=data_sample[c].shape)
    return data_sample
