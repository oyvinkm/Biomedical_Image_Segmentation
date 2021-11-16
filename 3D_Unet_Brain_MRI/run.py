from Trainer import NetworkTrainer
import torch
from Loss import (DiceLoss, WeightedTverskyLoss, BinaryFocalLoss)
from preprocessing.datasetModule import Set
from sklearn.model_selection import train_test_split
import os
from torch import nn
from Model import CNN
from Dynamic_Unet import Dynamic_3DUnet, ConvDropoutNonlinNorm, ConvDropoutNormNonlin
import nibabel as nib

'''______________________________________________________________________________________________
                                            CHANGE VARIABLES HERE
   ______________________________________________________________________________________________
'''

network = Dynamic_3DUnet #No need to change
loss_func = BinaryFocalLoss()
optimizer = torch.optim.Adam
batch_size = 1
num_batches_per_epoch = 1 #Number of batches before new epoch
epochs = 1
patch_size = (128, 128, 128)# Make sure that each value is divisible by 2**(num_pooling)
in_channels = 3 #No need to change really
base_features = 4 #Number of base features in 3D
learning_rate = 3e-4
dialation_prob = 0.6
dialation_epochs = 50
dialation = True
#data_folder = 'Cropped_Task3'
data_folder = '3segmentations' 
out_folder = '3D_Unet_Train'
sub_dir = 'crop_sub-2'
alternate_folder = 'Segmentations'
cluster_path = os.path.join(os.getcwd(), 'Biomedical_Image_Segmentation/Cropped_Task3/crop_sub-233/crop_sub-233_space-T1_desc-masked_T1.nii.gz')
pc_path = os.path.join(os.getcwd(), 'Segmentations/crop_sub-233/crop_sub-233_space-T1_desc-masked_T1.nii.gz')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_path = cluster_path if torch.cuda.is_available() else pc_path
test_imgur_affine = nib.load(test_path).affine
model_kwargs = {'base_features': base_features, 'in_channels': 3,
               'num_classes':1 , 'depth': 4, 'conv_kwargs': None, 
               'dropout_kwargs': None, 'nonlin_kwargs': None, 'maxpool_kwargs': None, 
               'basic_block' : ConvDropoutNonlinNorm}

'''______________________________________________________________________________________________
                                            DON'T CHANGE UNDERLYING CODE
   ______________________________________________________________________________________________
'''

dir_path = os.path.join(os.getcwd(), data_folder) if (os.name == 'nt') else os.path.join(os.getcwd(), f"{data_folder}/{alternate_folder}")
data_folders = sorted([folder for folder  in os.listdir(dir_path) if 
                        os.path.isdir(os.path.join(dir_path, folder)) 
                        and sub_dir in folder])
train, test = train_test_split(data_folders, test_size = 0.3)
train, val = train_test_split(train, train_size=0.5)
X_train = Set(dir_path, train)
X_test = Set(dir_path, test)
X_val = Set(dir_path, val)


net_trainer = NetworkTrainer(device = device, network=network, epochs = epochs, loss_func=loss_func, batch_size=batch_size,
                            patch_size=(128,128,128), num_batches=num_batches_per_epoch, 
                            lr=learning_rate, train_set=X_train, test_set=X_test, val_set=X_val, optimizer=optimizer, 
                            output_folder=out_folder, affine=test_imgur_affine, model_kwargs=model_kwargs)

net_trainer.initialize()    
net_trainer.train()
net_trainer.test()
