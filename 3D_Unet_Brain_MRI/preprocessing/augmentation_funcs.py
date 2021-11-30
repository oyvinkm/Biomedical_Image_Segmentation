
from typing import Any, List, Optional, Sequence, Union, Tuple
import numpy as np
import random
from scipy.ndimage.interpolation import map_coordinates
import torch
from scipy.ndimage import map_coordinates
#Text
def augment_gaussian_noise(data_sample: np.ndarray, noise_variance: Tuple[float, float] = (0, 0.1),
                           p_per_channel: float = 1, per_channel: bool = False) -> np.ndarray:
    if not per_channel:
        variance = noise_variance[0] if noise_variance[0] == noise_variance[1] else \
            random.uniform(noise_variance[0], noise_variance[1])
    else:
        variance = None
    for c in range(data_sample.shape[0]):
        if np.random.uniform() < p_per_channel:
            variance_here = variance if variance is not None else \
                noise_variance[0] if noise_variance[0] == noise_variance[1] else \
                    random.uniform(noise_variance[0], noise_variance[1])
            data_sample[c] = data_sample[c] + np.random.normal(0.0, variance_here, size=data_sample[c].shape)
    return data_sample

def augment_rician_noise(data_sample, noise_variance : Tuple[float, float] = (0, 0.1)):
    variance = random.uniform(noise_variance[0], noise_variance[1])
    data_sample = np.sqrt(
        (data_sample + np.random.normal(0.0, variance, size=data_sample.shape)) ** 2 +
        np.random.normal(0.0, variance, size=data_sample.shape) ** 2) * np.sign(data_sample)
    return data_sample



def map_spatial_axes(
    img_ndim: int, spatial_axes: Optional[Union[Sequence[int], int]] = None, channel_first: bool = True
) -> List[int]:
    """
    Code from Monai
    Utility to map the spatial axes to real axes in channel first/last shape.


    Args:
        img_ndim: dimension number of the target image.
        spatial_axes: spatial axes to be converted, default is None.
            The default `None` will convert to all the spatial axes of the image.
            If axis is negative it counts from the last to the first axis.
            If axis is a tuple of ints.
        channel_first: the image data is channel first or channel last, default to channel first.

    """
    if spatial_axes is None:
        spatial_axes_ = list(range(1, img_ndim) if channel_first else range(img_ndim - 1))

    else:
        spatial_axes_ = []
        for a in spatial_axes:
            if channel_first:
                spatial_axes_.append(a if a < 0 else a + 1)
            else:
                spatial_axes_.append(a - 1 if a < 0 else a)

    return spatial_axes_

def convert_to_tensor(data,dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None):
    if data.ndim > 0:
        data = np.ascontiguousarray(data)
    return torch.as_tensor(data, dtype=dtype, device=device)


def create_matrix_rotation_x_3d(angle, matrix=None):
    """
    Code copied from nnUnet: 
    https://github.com/MIC-DKFZ/batchgenerators/blob/master/batchgenerators/augmentations/utils.py
    """
    rotation_x = np.array([[1, 0, 0],
                           [0, np.cos(angle), -np.sin(angle)],
                           [0, np.sin(angle), np.cos(angle)]])
    if matrix is None:
        return rotation_x

    return np.dot(matrix, rotation_x)

def create_matrix_rotation_y_3d(angle, matrix=None):
    """
    Code copied from nnUnet: 
    https://github.com/MIC-DKFZ/batchgenerators/blob/master/batchgenerators/augmentations/utils.py
    """
    rotation_y = np.array([[np.cos(angle), 0, np.sin(angle)],
                           [0, 1, 0],
                           [-np.sin(angle), 0, np.cos(angle)]])
    if matrix is None:
        return rotation_y

    return np.dot(matrix, rotation_y)


def create_matrix_rotation_z_3d(angle, matrix=None):
    """
    Code copied from nnUnet: 
    https://github.com/MIC-DKFZ/batchgenerators/blob/master/batchgenerators/augmentations/utils.py
    """
    rotation_z = np.array([[np.cos(angle), -np.sin(angle), 0],
                           [np.sin(angle), np.cos(angle), 0],
                           [0, 0, 1]])
    if matrix is None:
        return rotation_z

    return np.dot(matrix, rotation_z)

def rotate_coords_3d(coords, angle_x, angle_y, angle_z):
    """
    Code copied from nnUnet: 
    https://github.com/MIC-DKFZ/batchgenerators/blob/master/batchgenerators/augmentations/utils.py
    """
    rot_matrix = np.identity(len(coords))
    rot_matrix = create_matrix_rotation_x_3d(angle_x, rot_matrix)
    rot_matrix = create_matrix_rotation_y_3d(angle_y, rot_matrix)
    rot_matrix = create_matrix_rotation_z_3d(angle_z, rot_matrix)
    coords = np.dot(coords.reshape(len(coords), -1).transpose(), rot_matrix).transpose().reshape(coords.shape)
    return coords


def do_rotation(angle_x = (0,2 * np.pi), angle_y = (0,2 * np.pi), 
                angle_z = (0,2*np.pi), shape : Tuple = (128,128,128), p_rot_per_channel: float = 1, p_rot_per_axis : float = 1):
    coords = create_zero_centered_coordinate_mesh(shape)
    if np.random.uniform() < p_rot_per_channel:
        if np.random.uniform() <= p_rot_per_axis:
            a_x = np.random.uniform(angle_x[0], angle_x[1])
        else: 
            a_x = 0
        if np.random.uniform() <= p_rot_per_axis:
            a_y = np.random.uniform(angle_y[0], angle_y[1])
        else:
            a_y = 0
        if np.random.uniform() <= p_rot_per_axis:
            a_z = np.random.uniform(angle_z[0], angle_z[1])
        else:
            a_z = 0
            coords = rotate_coords_3d(coords, a_x, a_y, a_z)
    return coords

def interpolate_img(img, coords, order=3, mode='nearest', cval=0.0, is_seg=False):
    """
    Code copied from nnUnet: 
    https://github.com/MIC-DKFZ/batchgenerators/blob/master/batchgenerators/augmentations/utils.py
    """
    if is_seg and order != 0:
        unique_labels = np.unique(img)
        result = np.zeros(coords.shape[1:], img.dtype)
        for i, c in enumerate(unique_labels):
            res_new = map_coordinates((img == c).astype(float), coords, order=order, mode=mode, cval=cval)
            result[res_new >= 0.5] = c
        return result
    else:
        return map_coordinates(img.astype(float), coords, order=order, mode=mode, cval=cval).astype(img.dtype)

def create_zero_centered_coordinate_mesh(shape):
    """
    Code copied from nnUnet: 
    https://github.com/MIC-DKFZ/batchgenerators/blob/master/batchgenerators/augmentations/utils.py
    """
    tmp = tuple([np.arange(i) for i in shape])
    coords = np.array(np.meshgrid(*tmp, indexing='ij')).astype(float)
    for d in range(len(shape)):
        coords[d] -= ((np.array(shape).astype(float) - 1) / 2.)[d]
    return coords