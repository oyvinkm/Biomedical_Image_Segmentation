from Trainer import NetworkTrainer
import torch
from Loss import (DiceLoss, WeightedTverskyLoss, BinaryFocalLoss)
from preprocessing.datasetModule import Set
from sklearn.model_selection import train_test_split
import os
from Model import CNN

'''______________________________________________________________________________________________
                                            CHANGE VARIABLES HERE
   ______________________________________________________________________________________________
'''

network = CNN #No need to change
loss_func = BinaryFocalLoss()
optimizer = torch.optim.Adam
batch_size = 2 
num_batches_per_epoch = 1 #Number of batches before new epoch
epochs = 2
patch_size = (128, 128, 128)# Make sure that each value is divisible by 2**(num_pooling)
in_channels = 3 #No need to change really
base_features = 8 #Number of base features in 3D
learning_rate = 3e-4
dialation_prob = 0.6
dialation_epochs = 20
dialation = True 
out_folder = '3D_Unet_Train'
sub_dir = 'crop_sub-2'
alternate_folder = 'Segmentations'

'''______________________________________________________________________________________________
                                            DON'T CHANGE UNDERLYING CODE
   ______________________________________________________________________________________________
'''


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dir_path = os.path.join(os.getcwd(), f"Cropped_Task3/{alternate_folder}")
data_folders = sorted([folder for folder  in os.listdir(dir_path) if 
                        os.path.isdir(os.path.join(dir_path, folder)) 
                        and sub_dir in folder])
train, test = train_test_split(data_folders, test_size = 0.3)
train, val = train_test_split(train, train_size=0.8)
X_train = Set(dir_path, train)
X_test = Set(dir_path, test)
X_val = Set(dir_path, val)


net_trainer = NetworkTrainer(device = device, network=network, epochs = epochs, loss_func=loss_func, batch_size=batch_size,
                            patch_size=(128,128,128), num_batches=num_batches_per_epoch, in_channels=in_channels, 
                            base_features=base_features, 
                            lr=learning_rate, train_set=X_train, test_set=X_test, val_set=X_val, optimizer=optimizer, output_folder=out_folder)

net_trainer.initialize()    
net_trainer.train()