import abc
from warnings import warn
from typing import Any, List, Optional, Sequence, Union, Tuple
import numpy as np
import random

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
def ensure_tuple(vals: Any) -> Tuple[Any, ...]:
    """
    Code from monai
    Returns a tuple of `vals`.
    """

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