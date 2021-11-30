from numpy.ma.core import empty
import scipy
from torch._C import device
from preprocessing.DataLoaderModule import DataLoaderBase
from collections import OrderedDict
import numpy as np
from scipy.ndimage import binary_fill_holes
from scipy import ndimage
from typing import Any, List, Optional, Sequence, Union, Tuple
import random
import torch
import torchio as tio


class DataLoader3D(DataLoaderBase):

    def __init__(self, data, BATCH_SIZE,  patch_size = None, is_test = False, pad_mode = 'edge',  
                    pad_kwargs_data = None, to_tensor = True, dialate : bool = False, iterations = 1, alpha : float = 0.6
                    ):
        
        super(DataLoader3D, self).__init__(data, BATCH_SIZE, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict
        if is_test:
            self.test_index = [i for i in range(BATCH_SIZE)]
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.is_test = is_test
        self.patch_size = patch_size
        if not self.is_test:
            self.data_shape, self.seg_shape = self.determine_shapes()
        self.data = data
        self.iterations = iterations
        self.data_len = self.get_data_length()
        self.to_tensor = to_tensor
        self.device = device
        self.dialate = dialate
        self.alpha = alpha
        self.dialeteshape = ndimage.generate_binary_structure(rank=4, connectivity=1)

    def get_seg_position(self, indx):
        seg_pos = np.where(self._data[indx]['seg'][0] != 0)
        if (len(seg_pos[0]) == 0 and len(seg_pos[1]) == 0 and len(seg_pos[2]) == 0):
            seg_pos = None
        return seg_pos

    def ToggleDialate(self, dialate : bool):
        self.dialate = dialate

    def GetDialate(self):
        return self.dialate

    def UpdateDialPad(self, iterations: int):
        if self.iterations > 1:
            if np.random.uniform() > self.alpha:
                self.iterations += iterations

    def get_keys(self):
        '''Don't use, it's super slow'''
        return {(self._data[i]['key']) : i for i in range(self.data_len)}

    def determine_shapes(self):
        num_seg = 1
        num_channels = 3
        data_shape = (self.BATCH_SIZE, num_channels, *self.patch_size)
        seg_shape = (self.BATCH_SIZE, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def get_data_length(self):
        return len(self._data)

    def get_random_center(self, shape, x : Tuple[int, int], y : Tuple[int,int], z : Tuple[int,int]):
        c_x = x[1] - (x[1] - x[0])
        c_y = y[1] - (y[1] - y[0])
        c_z = z[1] - (z[1] - z[0])
        center = np.array([c_x,c_y,c_z])
        rand_vec = np.random.randint(32, size=3)
        direction = 1 if random.random() < 0.5 else -1
        new_center = center + direction * rand_vec
        slicer = []
        for i in range(3):
            slicer.append(self.get_bbox_random(shape[i], new_center[i], i))
        resizer_data = (slice(0,3), slicer[0], slicer[1], slicer[2])
        resizer_seg = (slice(0,1), slicer[0], slicer[1], slicer[2])
        return resizer_data, resizer_seg

    def get_bbox_random(self, shape, center_point : int, axis : int):
        new_min = center_point - self.patch_size[axis] // 2
        new_max = center_point + self.patch_size[axis] // 2
        cut_min = new_min < 0
        cut_max = new_max > shape
        if cut_min:
            lb = 0
            ub = new_max + np.abs(new_min)
            assert ub - lb == self.patch_size[axis], ('Diff has to be shape 128, ub: ', ub, ' lb: ', lb)
        elif cut_max:
            lb = new_min - (new_max - shape)
            ub = shape
        else: 
            lb = new_min
            ub = new_max
        return slice(lb, ub)
    #Text
    def generate_train_batch(self):
        selected_index = np.random.choice(self.data_len, self.BATCH_SIZE, True, None)
        selected_keys = [self._data[k]['key'] for k in selected_index]
        if not self.is_test:
            "create patches if it's not testing"
            data = np.zeros(self.data_shape, dtype=np.float32)
            seg = np.zeros(self.seg_shape, dtype=np.float32)
            # i = batch number
            # j = index
            for i, j in enumerate(selected_index):
                seg_pos = self.get_seg_position(j)
                data_shape = self._data[j]['data'][0].shape
                if seg_pos is not None:
                    min_x = int(np.min(seg_pos[0]))
                    max_x = int(np.max(seg_pos[0]))
                    x = (min_x, max_x)
                    min_y = int(np.min(seg_pos[1]))
                    max_y = int(np.max(seg_pos[1]))
                    y = (min_y, max_y)
                    min_z = int(np.min(seg_pos[2])) 
                    max_z = int(np.max(seg_pos[2]))
                    z = (min_z, max_z)
                    
                    #If true different between lacunes along x/y/z axis is greater than patch size
                    x_tresh_gt = max_x - min_x > self.patch_size[0]
                    y_tresh_gt = max_y - min_y > self.patch_size[1]
                    z_tresh_gt = max_z - min_z > self.patch_size[2]

                    if (x_tresh_gt or y_tresh_gt or z_tresh_gt):
                        # If we need to choose on of two lacune regions
                        crop_choice = np.random.choice(['min', 'max'],1)
                        if crop_choice == 'min':
                            x = (min_x, min_x + np.random.randint(5))
                            y = (min_y, min_y + np.random.randint(5))
                            z = (min_z, min_z + np.random.randint(5))
                            resizer_data, resizer_seg = self.get_random_center(data_shape, x, y, z)

                        elif crop_choice == 'max':
                            x = (max_x, max_x + np.random.randint(5))
                            y = (max_y, max_y + np.random.randint(5))
                            z = (max_z, max_z + np.random.randint(5))
                            resizer_data, resizer_seg = self.get_random_center(data_shape, x, y, z)
                    
                    else:
                    #The lacune regions fits in the patch
                        resizer_data, resizer_seg = self.get_random_center(data_shape, x, y, z)
                    cropped_data = self._data[j]['data'][resizer_data]
                    data[i] = cropped_data
                    cropped_seg = self._data[j]['seg'][resizer_seg]
                    seg[i] = cropped_seg
                    if self.dialate:
                            dialated = ndimage.binary_dilation(seg[i], self.dialeteshape, iterations=self.iterations)
                            seg[i] = dialated
                            
                else: 
                    #Does not contain segmentation, so just doing a random crop
                    lb_x = np.random.randint(0, data_shape[0] - self.patch_size[0])
                    lb_y = np.random.randint(0, data_shape[1] - self.patch_size[1])
                    lb_z = np.random.randint(0, data_shape[2] - self.patch_size[2])
                    ub_x = lb_x + self.patch_size[0]
                    ub_y = lb_y + self.patch_size[1]
                    ub_z = lb_z + self.patch_size[2]
                    resizer_data = (slice(0,3),slice(lb_x, ub_x), slice(lb_y, ub_y), slice(lb_z, ub_z))
                    resizer_seg = (slice(0,1),slice(lb_x, ub_x), slice(lb_y, ub_y), slice(lb_z, ub_z))
                    cropped_data = self._data[j]['data'][resizer_data]
                    data[i] = cropped_data
                    cropped_seg = self._data[j]['seg'][resizer_seg]
                    seg[i] = cropped_seg
        else:
            "In case it's the testing set, don't create patches"
            max_x, max_y, max_z = 0, 0, 0
            data_list = []
            for i, j in enumerate(self.test_index):
                "iterating through the batch to get the max x, y and z position"
                data = self._data[j]
                data_shape = data['data'][0].shape
                max_x = max(max_x, data_shape[0])
                max_y = max(max_y, data_shape[1])
                max_z = max(max_z, data_shape[2])
                data_list.append(data)
            "ensures that the value can go 3 layers deep in a neural network"
            max_x = max_x + (8 - (max_x % 8))
            max_z = max_z + (8 - (max_z % 8))
            max_y = max_y + (8 - (max_y % 8))
            self.patch_size = (max_x, max_y, max_z)
            self.data_shape, self.seg_shape = self.determine_shapes()
            "Creates the dictionary"
            data = np.zeros(self.data_shape, dtype=np.float32)
            seg = np.zeros(self.seg_shape, dtype=np.float32)
            transform = tio.CropOrPad(self.patch_size)
            for i in range(len(data_list)):
                data[i] = transform(data_list[i]['data'])
                seg[i] = transform(data_list[i]['seg'])
            self.test_index = [i + self.BATCH_SIZE for i in self.test_index]
        if self.to_tensor:
            data = torch.from_numpy(data)
            seg = torch.from_numpy(seg)
        return {'data' : data, 'seg' : seg, 'keys' : selected_keys}

    
    
