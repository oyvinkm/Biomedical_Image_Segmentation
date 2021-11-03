
from preprocessing.DataAugmentation import AddGaussianNoise
from preprocessing.datasetModule import Set
from preprocessing.DataLoader3D import DataLoader3D
import os
from pathlib import Path
try:
    from preprocessing.DataAugmentation import (AddGaussianNoise, 
                                                RotateRandomTransform, 
                                                FlipTransform, 
                                                AddRicianNoise,
                                                SpatialTransformsRotate)
except:
    print('Path not good enough')
from torchvision import transforms
from torch import nn
from DeepSupervision import DeepSupervisionLoss
from Model import CNN
from loss import BinaryFocalLoss
import torch
import numpy as np
from Image_Functions import slicing
maxpool = nn.MaxPool3d

dir_path = os.path.join(Path(Path.cwd()), 'Cropped_Task3/Segmentations')
sub_dir = 'crop_sub-2'
data_folders = sorted([folder for folder  in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, folder)) and sub_dir in folder])
#data = sorted([name for name in os.listdir(path) if os.path.isfile(os.path.join(path,name)) and 'masked' in name])
#seg = sorted([name for name in os.listdir(path) if os.path.isfile(os.path.join(path,name)) and 'Lacunes' in name])
print(data_folders[:1])

training = Set(dir_path, data_folders[:1], transform=transforms.Compose([FlipTransform(prob = .5)]))
load = DataLoader3D(training, (128,128,128), BATCH_SIZE=1, device = 'cpu')
for i, img in enumerate(load):
    slicing(img['seg'][0][0], 64,64,64)
    slicing(img['data'][0][0], 64,64,64)




 

model = CNN(3,8, deep_supervision=False)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_func_2 = BinaryFocalLoss()



noise = AddGaussianNoise()(load)


exit()
def CreateSeg(seg, x : int):
    segs = [None for _ in range(x)]
    segs[-1] = seg
    for i in range(x-2, -1, -1):
        max = maxpool(2)
        segs[i] = max(segs[i+1])
    return segs 
    






