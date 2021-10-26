import abc
from warnings import warn
from typing import Optional, Sequence, Union, Tuple
import numpy as np
from preprocessing.augmentation_funcs import augment_gaussian_noise, augment_rician_noise, map_spatial_axes
import random
from scipy.ndimage import rotate



class AddGaussianNoise(object):
    def __init__(self, noise_variance=(0, 0.1), p_per_sample=1, p_per_channel : float = 1, per_channel:bool=False, data_key='data'):
        self.noise_variance = noise_variance
        self.p_per_sample = p_per_sample
        self.p_per_channel = p_per_channel
        self.per_channel = per_channel
        self.data_key = data_key

    def __call__(self, data_dict):
        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.p_per_sample:
                data_dict[self.data_key][b] = augment_gaussian_noise(data_dict[self.data_key][b], self.noise_variance, self.p_per_channel, self.per_channel)
        return data_dict


class AddRicianNoise(object):
    def __init__(self, noise_variance=(0, 0.1), p_per_sample=1, data_key='data'):
        self.noise_variance = noise_variance
        self.p_per_sample = p_per_sample
        self.data_key = data_key

    def __call__(self, data_dict):
        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.p_per_sample:
                data_dict[self.data_key][b] = augment_rician_noise(data_dict[self.data_key][b], self.noise_variance)
        return data_dict


class RotateRandomTransform(object):
    def __init__(self, angle = None, reshape = False, data_key='data', seg_key = 'seg', output=None):
        self.data_key = data_key
        self.seg_key = data_key
        self.output = output
        self.angle = angle
        self.reshape = reshape

    def __call__(self, sample):
        if self.angle is None:
            self.angle = np.random.uniform(1, 89)
        d_shape, s_shape = sample[self.data_key].shape, sample[self.seg_key].shape
        for c in range(d_shape[0]):
            sample[self.data_key][c] = rotate(sample[self.data_key][c], self.angle, reshape = self.reshape, order = 4)
        for d in range(s_shape[0]):
            sample[self.seg_key][d] = rotate(sample[self.seg_key][d], self.angle, reshape=self.reshape, order = 4)
        return sample

class FlipTransform(object):
    def __init__(self, spatial_axis: Optional[Union[Sequence[int], int]] = None, data_key = 'data', seg_key = 'seg') -> None:
        self.spatial_axis = spatial_axis
        self.data_key = data_key
        self.seg_key = seg_key

    def __call__(self, sample):
        d_shape, s_shape = sample[self.data_key].shape, sample[self.seg_key].shape
        for c in range(d_shape[0]):
            sample[self.data_key][c] = np.ascontiguousarray(np.flip(sample[self.data_key][c], map_spatial_axes(sample[self.data_key][c].ndim, self.spatial_axis))) 
        for d in range(s_shape[0]):
            sample[self.seg_key][d] = np.ascontiguousarray(np.flip(sample[self.seg_key][d], map_spatial_axes(sample[self.seg_key][d].ndim, self.spatial_axis)))
        return sample