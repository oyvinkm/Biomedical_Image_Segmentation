from operator import contains
from datasetModule import Set
from preprocessing.DataLoader3D import DataLoader3D
import os
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from Image_Functions import slicing
from Model import CNN
from Dice_Loss import DiceLoss
import torch
from Image_Functions import crop_to_size
from ExtraCrossEntropy import ExtraCrossEntropy
from matplotlib import pyplot as plt
from preprocessing.DataAugmentation import AddGaussianNoise

data_path = os.path.join(Path(Path.cwd()), 'Cropped_Task3')

print(data_path)
folder = 'crop_sub-23'
path = os.path.join(data_path,folder)
#data = sorted([name for name in os.listdir(path) if os.path.isfile(os.path.join(path,name)) and 'masked' in name])
#seg = sorted([name for name in os.listdir(path) if os.path.isfile(os.path.join(path,name)) and 'Lacunes' in name])

training, test = train_test_split(Set(data_path, sub_dir=folder), test_size=0.1, random_state=25)

#print(training.shape)
num_batches = 5

loaded = DataLoader3D(training, (128,128,128), (128,128,128), 2, 10)
for epoch in range(5):
    print('epoch: ', epoch)
    for i, set in enumerate(loaded):
        print(i)
        print(set['keys'])
        print(set['data'][0].shape)
        if i >= num_batches:
            break


#print(test_img.keys())
#noise_test = 
 
#print(test_img['data'][0].shape)
#print(noise_test['data'][0].shape)
#slicing(test_img['data'][0])

#slicing(noise_test['data'][0])






