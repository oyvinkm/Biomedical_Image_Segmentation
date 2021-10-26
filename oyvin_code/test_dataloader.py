
from oyvin_code.preprocessing.DataAugmentation import AddGaussianNoise
from preprocessing.datasetModule import Set
from preprocessing.DataLoader3D import DataLoader3D
import os
from pathlib import Path
#import numpy as np
from sklearn.model_selection import train_test_split
from torch import nn
#from Image_Functions import slicing
from DeepSupervision import DeepSupervisionLoss
from Model import CNN
#from Dice_Loss import DiceLoss
import torch
import numpy as np
#from Image_Functions import crop_to_size
#from ExtraCrossEntropy import ExtraCrossEntropy
maxpool = nn.MaxPool3d

data_path = os.path.join(Path(Path.cwd()), 'Cropped_Task3/Segmentations')
folder = 'crop_sub-2'
path = os.path.join(data_path,folder)
#data = sorted([name for name in os.listdir(path) if os.path.isfile(os.path.join(path,name)) and 'masked' in name])
#seg = sorted([name for name in os.listdir(path) if os.path.isfile(os.path.join(path,name)) and 'Lacunes' in name])

training, test = train_test_split(Set(data_path, sub_dir=folder), test_size=0.1, random_state=25, shuffle=True)
def CreateSeg(seg, x : int):
    segs = [None for _ in range(x)]
    segs[-1] = seg
    for i in range(x-2, -1, -1):
        max = maxpool(2)
        segs[i] = max(segs[i+1])
    return segs  

model = CNN(3,8, deep_supervision=True)
loaded = DataLoader3D(training, (128,128,128), 1, 10)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_func_2 = nn.BCEWithLogitsLoss()
deep_loss = DeepSupervisionLoss(loss_func_2, weigths_factors=np.random.rand(3))
load = next(iter(loaded))

noise = AddGaussianNoise()(load)


    






