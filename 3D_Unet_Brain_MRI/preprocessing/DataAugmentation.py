import abc
from warnings import warn
from typing import Optional, Sequence, Union, Tuple
import numpy as np
from numpy.lib.type_check import real
from preprocessing.augmentation_funcs import (  augment_gaussian_noise, 
                                                augment_rician_noise, 
                                                map_spatial_axes, 
                                                convert_to_tensor, 
                                                do_rotation, 
                                                interpolate_img)
import random
from scipy.ndimage import rotate
from scipy.spatial.transform import Rotation as R
import torch


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
    def __init__(self, angle = None, reshape = False, data_key='data', seg_key = 'seg', output=None, prob : float = 1.):
        self.data_key = data_key
        self.seg_key = seg_key
        self.output = output
        self.angle = angle
        self.reshape = reshape
        self.prob = prob

    def __call__(self, sample):
        if np.random.uniform() < self.prob:
            axis = np.random.choice(2,2,replace=False)
            if self.angle is None:
                self.angle = np.random.uniform(1, 89)
            d_shape, s_shape = sample[self.data_key].shape, sample[self.seg_key].shape
            for c in range(d_shape[0]):
                sample[self.data_key][c] = rotate(sample[self.data_key][c],  self.angle, axes=(axis[0], axis[1]), reshape = self.reshape, order = 4)
            for d in range(s_shape[0]):
                sample[self.seg_key][d] = rotate(sample[self.seg_key][d], self.angle,axes=(axis[0], axis[1]), reshape=self.reshape, order = 4)
            return sample

class FlipTransform(object):
    def __init__(self, spatial_axis: Optional[Union[Sequence[int], int]] = None, data_key = 'data', seg_key = 'seg', prob : float = 1.) -> None:
        self.spatial_axis = spatial_axis
        self.data_key = data_key
        self.seg_key = seg_key
        self.prob = prob

    def __call__(self, sample):
        if np.random.uniform() < self.prob:
            d_shape, s_shape = sample[self.data_key].shape, sample[self.seg_key].shape
            for c in range(d_shape[0]):
                sample[self.data_key][c] = np.ascontiguousarray(np.flip(sample[self.data_key][c], map_spatial_axes(sample[self.data_key][c].ndim, self.spatial_axis))) 
            for d in range(s_shape[0]):
                sample[self.seg_key][d] = np.ascontiguousarray(np.flip(sample[self.seg_key][d], map_spatial_axes(sample[self.seg_key][d].ndim, self.spatial_axis)))
        return sample

class ToTensor(object):

    def __init__(self, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None) -> None:
        super().__init__()
        self.dtype = dtype
        self.device = device

    def __call__(self, img) -> torch.Tensor:
        """
        Apply the transform to `img` and make it contiguous.
        """
        return convert_to_tensor(img, dtype=self.dtype, device=self.device, wrap_sequence=True)

class SpatialTransformsRotate(object):
    def __init__(self, data_key = 'data', seg_key = 'seg', order = 3, border_mode_data = 'nearest',
                border_mode_seg = 'nearest', border_cval_seg = 0, border_cval_data = 0, 
                p_per_channel :float = 1, p_per_axis : float = 1) -> None:
        self.data_key = data_key
        self.seg_key = seg_key
        self.order = order
        self.border_mode_data = border_mode_data
        self.border_cval_data = border_cval_data
        self.border_mode_seg = border_mode_seg
        self.border_cval_seg = border_cval_seg
        self.p_per_channel = p_per_channel
        self.p_per_axis = p_per_axis

    def calc(self, sample):
        data = sample[self.data_key]
        seg = sample[self.seg_key]
        d_shape = data.shape
        s_shape = seg.shape
        seg_result = np.zeros(s_shape, dtype=np.float32)
        data_result = np.zeros(d_shape, dtype=np.float32)
        img_shape = ((d_shape[1], d_shape[2], d_shape[3]))

        coords = do_rotation(shape = img_shape)
        for channel_id in range(data.shape[0]):
            data_result[channel_id] = interpolate_img(data[channel_id], 
                                                    coords, self.order,
                                                    self.border_mode_data, 
                                                    cval=self.border_cval_data)
        for channel_id in range(seg.shape[0]):
            seg_result[channel_id] = interpolate_img(seg[channel_id], 
                                                    coords, self.order,
                                                    self.border_mode_seg, 
                                                    cval=self.border_cval_seg)
        return data_result, seg_result
    def __call__(self,data_dict):
        data_dict[self.data_key], data_dict[self.seg_key] = self.calc(data_dict)
        return data_dict