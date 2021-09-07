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


np.set_printoptions(precision=2, suppress=True)
affine_transform = tio.RandomAffine()

"Need to specify the local path on computer"
dir_path = 'Task3/sub-101/'
affine = None
imgs = []

"Loads images from the folder sub-101"
for img in os.listdir(dir_path):
    if (img == '.DS_Store'):
        continue
    else:
        img = tio.ScalarImage(dir_path + img)
        affine = img.affine
        print(type(img.data))
        imgs.append(img)
loader = DataLoader(imgs, batch_size=2)   


"Simple neural network with one convolution and activation"
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = self._conv_layer_set(1, 32)

    def _conv_layer_set(self, in_c, out_c):
            conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, 1),
            nn.LeakyReLU(),
            )
            return conv_layer

    def forward(self, x):
        print('1st convolution')
        out = self.conv1(x)
        return out

model = CNN()
print(len(loader))
test = next(iter(loader))
print(test['data'].shape)
out = model(test['data'])
print(out[0][0].shape)
out_img = out[0][0].detach().numpy()

ni_img = nib.Nifti1Image(out_img, affine)
nib.save(ni_img, 'out.nii.gz')
