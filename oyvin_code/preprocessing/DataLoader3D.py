from numpy.ma.core import empty
import scipy
from torch._C import device
from preprocessing.DataLoaderModule import DataLoaderBase
from collections import OrderedDict
import numpy as np
from scipy.ndimage import binary_fill_holes
from scipy import ndimage

import torch


class DataLoader3D(DataLoaderBase):

    def __init__(self, data, patch_size, BATCH_SIZE, device, pad_mode = 'edge',  
                    pad_kwargs_data = None, to_tensor = None, dialate : bool = False, iterations = 1):
        
        super(DataLoader3D, self).__init__(data, BATCH_SIZE, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.patch_size = patch_size
        self.data = data
        self.iterations = iterations
        self.data_len = self.get_data_length()
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.to_tensor = to_tensor
        self.device = device
        self.dialate = dialate
        self.dialeteshape = ndimage.generate_binary_structure(rank=4, connectivity=1)

    def get_seg_position(self, indx):
        seg_pos = np.where(self._data[indx]['seg'][0] != 0)
        if (len(seg_pos[0]) == 0 and len(seg_pos[1]) == 0 and len(seg_pos[2]) == 0):
            seg_pos = None
        return seg_pos

    def EnableDialate(self):
        if self.dialate:
            self.dialate = False
        else: 
            self.dialate = True

    def GetDialate(self):
        return self.dialate

    def UpdateDialPad(self, iterations: int):
        self.iterations += iterations

    def get_keys(self):
        '''Don't use, it's super slow'''
        print('getting keys')
        return {(self._data[i]['key']) : i for i in range(self.data_len)}

    def determine_shapes(self):
        num_seg = 1
        num_channels = 3
        data_shape = (self.BATCH_SIZE, num_channels, *self.patch_size)
        seg_shape = (self.BATCH_SIZE, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def get_data_length(self):
        return len(self._data)

    def get_bbox_axis(self, min, max, shape, axis : int, partial_patch = None): 
        if partial_patch is not None:
            assert partial_patch == 'min' or partial_patch == 'max', 'partial patch has to be either min or max'
            if partial_patch == 'min':
                new_min = min - self.patch_size[axis]//2
                cut_min = new_min < 0
                lb = new_min if not cut_min else 0
                ub = min + self.patch_size[axis]//2 if not cut_min else min + self.patch_size[axis]//2 + np.abs(new_min)
                assert ub - lb == self.patch_size[axis], ('Diff has to be shape 128, ub: ', ub, ' lb: ', lb) 
                return lb, ub
            else:
                new_max = max + self.patch_size[axis]//2
                cut_max = new_max > shape[axis]
                ub = new_max if not cut_max else shape[axis]
                lb = max - self.patch_size[axis]//2 if not cut_max \
                        else max - self.patch_size[axis]//2 - (new_max - shape[axis])
                assert ub - lb == self.patch_size[axis], ('Diff has to be shape 128, ub: ', ub, ' lb: ', lb) 
                return lb, ub
        else:
            padder = (self.patch_size[axis] - (max - min))//2
            new_min, new_max = min - padder, max + padder
            cut_min, cut_max = new_min < 0, new_max > shape[axis]
            if new_max - new_min != self.patch_size[axis]:
                new_max += self.patch_size[axis] - (new_max - new_min)
            if cut_min:
                lb = 0
                ub = new_max + np.abs(min - padder)
                assert ub - lb == self.patch_size[axis], ('Diff has to be shape 128, ub: ', ub, ' lb: ', lb) 
                return lb, ub
            elif cut_max:
                lb = new_min - (new_max - shape[axis])
                ub = shape[axis]
                assert ub - lb == self.patch_size[axis], ('Diff has to be shape 128, ub: ', ub, ' lb: ', lb)  
                return lb, ub
            else:
                lb = new_min
                ub = new_max
                assert ub - lb == self.patch_size[axis], ('Diff has to be shape 128, ub: ', ub, ' lb: ', lb) 
                return lb, ub
        


    def generate_train_batch(self):
        print('generating train batch')
        selected_index = np.random.choice(self.data_len, self.BATCH_SIZE, True, None)
        selected_keys = [self._data[k]['key'] for k in selected_index]
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
                min_y = int(np.min(seg_pos[1]))
                max_y = int(np.max(seg_pos[1]))
                min_z = int(np.min(seg_pos[2])) 
                max_z = int(np.max(seg_pos[2]))
                
                #If true different between lacunes along x/y/z axis is greater than patch size
                x_tresh_gt = max_x - min_x > self.patch_size[0]
                y_tresh_gt = max_y - min_y > self.patch_size[1]
                z_tresh_gt = max_z - min_z > self.patch_size[2]

                # If we need to choose on of two lacunes
                if (x_tresh_gt or y_tresh_gt or z_tresh_gt):
                    crop_choice = np.random.choice(['min', 'max'],1)
                    print(crop_choice)
                    lb_x, ub_x = self.get_bbox_axis(min_x , max_x, data_shape, 0, partial_patch=crop_choice)
                    lb_y, ub_y = self.get_bbox_axis(min_y , max_y, data_shape, 1, partial_patch=crop_choice)
                    lb_z, ub_z = self.get_bbox_axis(min_z , max_z, data_shape, 2, partial_patch=crop_choice)

                else:
                    lb_x, ub_x = self.get_bbox_axis(min_x , max_x, data_shape, 0)
                    lb_y, ub_y = self.get_bbox_axis(min_y , max_y, data_shape, 1)
                    lb_z, ub_z = self.get_bbox_axis(min_z , max_z, data_shape, 2)
                resizer_data = (slice(0,3),slice(lb_x, ub_x), slice(lb_y, ub_y), slice(lb_z, ub_z))
                resizer_seg = (slice(0,1),slice(lb_x, ub_x), slice(lb_y, ub_y), slice(lb_z, ub_z))
                cropped_data = self._data[j]['data'][resizer_data]
                data[i] = cropped_data
                cropped_seg = self._data[j]['seg'][resizer_seg]
                seg[i] = cropped_seg
                if self.dialate:
                        dialated = ndimage.binary_dilation(seg[i], self.dialeteshape, iterations=self.iterations)
                        seg[i] = dialated
                        
            else: 
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
        if self.to_tensor:
            data = torch.from_numpy(data)
            seg = torch.from_numpy(data)
        return {'data' : data, 'seg' : seg, 'keys' : selected_keys}

    
    