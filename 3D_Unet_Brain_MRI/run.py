from torch.optim import lr_scheduler
from Trainer import NetworkTrainer
import torch
from Loss import (DiceLoss, WeightedTverskyLoss, TverskyLoss, DiceFocalLoss, BinaryFocalLoss)
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
loss_func = BinaryFocalLoss(gamma=3.5)
optimizer = torch.optim.Adam
'''When choosing learning rate scheduele, either choose: 
   'Exponential', 'Lambda' or 'ReducePlateau' or None : LinearLR'''
scheduler = 'ReducePlateau'
batch_size = 2
num_batches_per_epoch = 40 #Number of batches before new epoch
epochs = 100
patch_size = (128, 128, 128)# Make sure that each value is divisible by 2**(num_pooling)
in_channels = 3 #No need to change really
base_features = 4 #Number of base features in 3D
learning_rate = 0.01
dialation_prob = 0.6
dialation_epochs = 50
dialation = True
#data_folder = 'Cropped_Task3'
data_folder = 'Numpy_Task3' 

out_folder = '3D_Unet_Train'
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
               'num_classes':1 , 'depth': 4, 'conv_kwargs': None, 'dropout_op': None,
               'dropout_kwargs': None, 'nonlin_kwargs': None, 'maxpool_kwargs': None, 
               'basic_block' : ConvDropoutNormNonlin}



dir_path = os.path.join(os.getcwd(), data_folder) if (os.name == 'nt') else os.path.join(os.getcwd(), f"{data_folder}/{alternate_folder}")
data_folders = sorted([folder for folder  in os.listdir(dir_path) if 
                        os.path.isdir(os.path.join(dir_path, folder)) 
                        and sub_dir in folder])
train, test = train_test_split(data_folders, test_size = 0.15)
train, val = train_test_split(train, train_size=0.8)
X_train = Set(dir_path, train)
X_test = Set(dir_path, test)
X_val = Set(dir_path, val)


net_trainer = NetworkTrainer(device = device, network=network, epochs = epochs, loss_func=loss_func, batch_size=batch_size,
                            patch_size=(128,128,128), num_batches=num_batches_per_epoch, 
                            lr=learning_rate, train_set=X_train, test_set=X_test, val_set=X_val, optimizer=optimizer, 
                            output_folder=out_folder, affine_sub2=test_imgur_affine_sub2, affine_sub1=test_imgur_affine_sub1, 
                            model_kwargs=model_kwargs, dialate_p=dialation_prob, dialate_epochs=dialation_epochs, 
                            dialate=dialation, lr_schedule=scheduler)

net_trainer()