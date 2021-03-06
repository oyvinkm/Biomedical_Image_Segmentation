from torch.optim import lr_scheduler
from Trainer import NetworkTrainer
import torch
from Loss import (DiceLoss, WeightedTverskyLoss, 
                  TverskyLoss, DiceFocalLoss, 
                  BinaryFocalLoss, FocalTversky, BCEWLLoss)
from preprocessing.datasetModule import Set
from preprocessing.DataAugmentation import (AddGaussianNoise, 
AddRicianNoise, FlipTransform, AddNoise)
from torchvision import transforms
from sklearn.model_selection import train_test_split
import os
from torch import nn
import numpy as np
import random
from Model import CNN
from Dynamic_Unet import Dynamic_3DUnet, ConvDropoutNonlinNorm, ConvDropoutNormNonlin
import nibabel as nib

np.random.seed(2021)
random.seed(2021)
data_folder = 'Numpy_Task3' 
out_folder = '3D_Unet_Train'
sub_2 = 'crop_sub-2'
sub_1 = 'crop_sub-1'
alternate_folder = 'Segmentations'
dir_path = os.path.join(os.getcwd(), f"{data_folder}/{alternate_folder}")
data_folders_1 = sorted([folder for folder  in os.listdir(dir_path) if 
                        os.path.isdir(os.path.join(dir_path, folder)) 
                        and sub_1 in folder])
data_folders_2 = sorted([folder for folder  in os.listdir(dir_path) if 
                        os.path.isdir(os.path.join(dir_path, folder)) 
                        and sub_2 in folder])
train_1, test_1 = train_test_split(data_folders_1, test_size = 0.2, random_state=11)
train_2, test_2 = train_test_split(data_folders_2, test_size = 0.2, random_state=11)
train = train_1 + train_2
train, val = train_test_split(train, train_size=0.8)
test_path = os.path.join(os.getcwd(), data_folder)
test = [folder for folder in os.listdir(test_path) if 
         (os.path.join(test_path, folder)) if os.path.isdir(os.path.join(test_path, folder))
         and folder not in (train + val)]   
test = np.random.choice(test, 6, False).tolist()
test = np.unique(test + test_1 + test_2)
print(len(train))
print(len(val))
print(len(test_1))
print(len(test_2))
print(len(test))
exit()
#Text
'''______________________________________________________________________________________________
                                            CHANGE VARIABLES HERE
   ______________________________________________________________________________________________
'''

network = Dynamic_3DUnet #No need to change
loss_func = BinaryFocalLoss()
optimizer = torch.optim.Adam
'''When choosing learning rate scheduele, either choose: 
   'Exponential', 'Lambda' or 'ReducePlateau' or None : LinearLR'''
scheduler = 'Lambda'
batch_size = 2
num_batches_per_epoch = 1 #Number of batches before new epoch
epochs = 1
patch_size = (128, 128, 128)# Make sure that each value is divisible by 2**(num_pooling)
in_channels = 3 #No need to change really
base_features = 8 #Number of base features in 3D
gaussian = AddGaussianNoise(noise_variance=(3., 10.), p_per_sample = .2, p_per_channel = .4)
rician = AddRicianNoise(noise_variance=(3., 10.), p_per_sample=.2)
transform = transforms.Compose([AddNoise(rician, gaussian), FlipTransform(prob = .3)])
learning_rate = 0.001
dialation_prob = 0.6
dialation_epochs = 50
dialation = True
#data_folder = 'Cropped_Task3'
data_folder = 'Numpy_Task3' 
out_folder = '3D_Unet_Train'
image_path = os.path.join(os.getcwd(),'Cropped_Task3')
sub_dir = 'crop_sub'
alternate_folder = 'Segmentations'

'''______________________________________________________________________________________________
                                            DON'T CHANGE UNDERLYING CODE
   ______________________________________________________________________________________________
'''


cluster_path_sub2 = os.path.join(os.getcwd(), 'Cropped_Task3/crop_sub-233/crop_sub-233_space-T1_desc-masked_T1.nii.gz')
pc_path_sub2 = os.path.join(os.getcwd(), 'Cropped_Task3/crop_sub-233/crop_sub-233_space-T1_desc-masked_T1.nii.gz')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_path_sub2 = cluster_path_sub2 if torch.cuda.is_available() else pc_path_sub2
test_imgur_affine_sub2 = nib.load(test_path_sub2).affine
cluster_path_sub1 = os.path.join(os.getcwd(), 'Cropped_Task3/crop_sub-102/crop_sub-102_space-T1_desc-masked_T1.nii.gz')
pc_path_sub1 = os.path.join(os.getcwd(), 'Cropped_Task3/crop_sub-102/crop_sub-102_space-T1_desc-masked_T1.nii.gz')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_path_sub1 = cluster_path_sub1 if torch.cuda.is_available() else pc_path_sub1
test_imgur_affine_sub1 = nib.load(test_path_sub1).affine
model_kwargs = {'base_features': base_features, 'in_channels': 3,
               'num_classes':1 , 'depth': 4, 'conv_kwargs': None, 'dropout_op': None, 'norm_op' : nn.InstanceNorm3d,
               'dropout_kwargs': None, 'nonlin_kwargs': None, 'maxpool_kwargs': None, 
               'basic_block' : ConvDropoutNormNonlin}



dir_path = os.path.join(os.getcwd(), data_folder) if (os.name == 'nt') else os.path.join(os.getcwd(), f"{data_folder}/{alternate_folder}")
dir_test_path = os.path.join(os.getcwd(), data_folder) if (os.name == 'nt') else os.path.join(os.getcwd(), f"{data_folder}")
data_folders = sorted([folder for folder  in os.listdir(dir_path) if 
                        os.path.isdir(os.path.join(dir_path, folder)) 
                        and sub_dir in folder])


X_train = Set(dir_path, train, transform=transform)
X_test = Set(dir_test_path, test)
X_val = Set(dir_path, val)


net_trainer = NetworkTrainer(device = device, network=network, epochs = epochs, image_path=image_path, loss_func=loss_func, batch_size=batch_size,
                            patch_size=(128,128,128), num_batches=num_batches_per_epoch, 
                            lr=learning_rate, train_set=X_train, test_set=X_test, val_set=X_val, optimizer=optimizer, 
                            output_folder=out_folder, affine_sub2=test_imgur_affine_sub2, affine_sub1=test_imgur_affine_sub1, 
                            model_kwargs=model_kwargs, dialate_p=dialation_prob, dialate_epochs=dialation_epochs, 
                            dialate=dialation, lr_schedule=scheduler)

net_trainer()