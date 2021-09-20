import os
import numpy as np
from nibabel.testing import data_path
import nibabel as nib
import torch
from numpy import random as rand
import torchio as tio
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from typing import Union, Tuple, List
import matplotlib.pyplot as plt
import PIL
import PIL.Image as Image



"Need to specify the local path on computer"
dir_path = '../Task3/sub-101/'
imgs = []

"Loads images from the folder sub-101"
for img in os.listdir(dir_path):
    if (img == '.DS_Store'):
        continue
    else:
        tmp = nib.load(dir_path + img)  
        tmp_arr = tmp.get_fdata()
        x = np.expand_dims(tmp_arr, axis=0)
        print(x[0][100][100][100])
        print(x.shape)
        imgs.append(x)
loader = DataLoader(imgs, batch_size=2)  



def show_slices(slices):
   """ Function to display row of image slices """
   fig, axes = plt.subplots(1, len(slices))
   for i, slice in enumerate(slices):
       axes[i].imshow(slice.T, cmap="gray", origin="lower")




"Simple neural network with one convolution and activation"
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = self._conv_layer_set(1, 32)
        self.conv2 = self._conv_layer_set(32,64)
        
    def _conv_layer_set(self, in_c, out_c):
            conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, 1),
            nn.LeakyReLU(),
            )
            return conv_layer

    def forward(self, x):
        x = x.float()
        print('1st convolution')
        out = self.conv1(x)
        print(out.shape)
        out=self.conv2(out)
        return out



model = CNN()
print(len(loader))
dataiter = iter(loader)
test = dataiter.next()
test.float()
print('Test shape:', test.shape)   


out = model(test)
print(out.shape)



slice_0 = test[0][0][50, :, :]
slice_1 = test[0][0][:, 89, :]
slice_2 = test[0][0][:, :, 130]
show_slices([slice_0, slice_1, slice_2])
plt.suptitle("Center slices for EPI image") 
plt.show()
