
from preprocessing.datasetModule import Set
from preprocessing.DataLoader3D import DataLoader3D
import os
from pathlib import Path
#import numpy as np
from sklearn.model_selection import train_test_split
#from Image_Functions import slicing
#from Model import CNN
#from Dice_Loss import DiceLoss
#import torch
#from Image_Functions import crop_to_size
#from ExtraCrossEntropy import ExtraCrossEntropy


data_path = os.path.join(Path(Path.cwd()), 'Cropped_Task3')
folder = 'crop_sub-1'
path = os.path.join(data_path,folder)
#data = sorted([name for name in os.listdir(path) if os.path.isfile(os.path.join(path,name)) and 'masked' in name])
#seg = sorted([name for name in os.listdir(path) if os.path.isfile(os.path.join(path,name)) and 'Lacunes' in name])

training, test = train_test_split(Set(data_path, sub_dir=folder), test_size=0.1, random_state=25, shuffle=True)

print(training[0]['data'].shape)

loaded = DataLoader3D(training, (128,128,128), (128,128,128), 2, 10)

for i, image in enumerate(loaded):
    print(image['keys'])
    print(i)

    






