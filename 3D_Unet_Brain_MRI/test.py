
from torch.optim import optimizer
from preprocessing.datasetModule import Set
from preprocessing.DataLoader3D import DataLoader3D
import os
import numpy as np
import csv
import torch
import nibabel as nib
import torch.optim.lr_scheduler as lr_s
from Dynamic_Unet import Dynamic_3DUnet, ConvDropoutNonlinNorm, ConvDropoutNormNonlin
def_lr = 0.01
epoch = 20
base_features = 8
device = 'cpu'
loss_func = torch.nn.BCELoss()
sub_dir = 'crop_sub'
dir_path = os.path.join(os.getcwd(), 'Cropped_Task3/Segmentations')
data_folders = sorted([folder for folder  in os.listdir(dir_path) if 
                        os.path.isdir(os.path.join(dir_path, folder)) 
                        and sub_dir in folder])
model_kwargs = {              'base_features': base_features, 'in_channels': 3,
               'num_classes':1 , 'depth': 4, 'conv_kwargs': None, 'dropout_op': None,
               'dropout_kwargs': None, 'nonlin_kwargs': None, 'maxpool_kwargs': None, 
               'basic_block' : ConvDropoutNormNonlin}
network = Dynamic_3DUnet(**model_kwargs)
print(network)

""" opt1 = torch.optim.Adam(network.parameters(), lr=0.01)
opt2 = torch.optim.Adam(network.parameters(), lr=0.01)
opt3 = torch.optim.Adam(network.parameters(), lr=0.01)
opt4 = torch.optim.Adam(network.parameters(), lr=0.01) """
#scheduler =   lr_s.ExponentialLR()
test_set = Set(dir_path, data_folders)
load = DataLoader3D(test_set, BATCH_SIZE=1, patch_size=(64,64,64))

""" def interchangable_lr(optim, scheduler : str = None, epochs = None, verb = False):
    if scheduler == 'Exponential':
        return (lr_s.ExponentialLR(optimizer=optim, gamma=0.9, verbose=verb))
    elif scheduler == 'Lambda': 
        lambda_1 = lambda epoch: 0.9 ** epoch
        return lr_s.LambdaLR(optim, lr_lambda=[lambda_1], verbose=verb)
    elif scheduler == 'ReducePlateau': 
        return lr_s.ReduceLROnPlateau(optim, 'min', verbose=verb)
    elif scheduler is None:
        return lr_s.LinearLR(optim, start_factor=0.4, total_iters=epochs - 1, verbose=verb) """

""" sch1 = interchangable_lr(opt1, scheduler = 'Exponential', verb = True)
sch2 = interchangable_lr(opt2, scheduler = 'Lambda', verb = True)
sch3 = interchangable_lr(opt3, scheduler = 'ReducePlateau', verb=True)
sch4 = interchangable_lr(opt4, epochs = epoch, verb = True) """

for e in range(epoch):
    print(e)
    for i, image_set in enumerate(load):
        image = image_set['data'].to(device)
        labels = image_set['seg'].to(device)
        outputs = network(image)
        loss = loss_func(outputs, labels)
        loss.backward()
        if i == 2: 
            break






    



