
from torch import nn
import torch
from torch._C import device
from torch.utils.data import dataloader
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from preprocessing.DataLoader3D import DataLoader3D
from preprocessing.datasetModule import Set
from preprocessing.DataLoaderModule import DataLoaderBase as DL
from preprocessing.DataLoader3D import DataLoader3D
import os
from torch.optim import Optimizer
import numpy as np
from Image_Functions import save_slice
from matplotlib import pyplot as plt
from datetime import datetime


class NetworkTrainer():
    def __init__(self, device, network : nn.Module, epochs : int, 
                loss_func : nn.Module, batch_size : int, patch_size : tuple, num_batches : int, in_channels : int,
                base_features : int, lr : float, train_set : Set, test_set : Set, val_set : Set, optimizer : Optimizer, 
                output_folder : str, dialate_p : float = 0.6, dialate_epochs : int = 20, dialate : bool = True):
        self.dialate = dialate
        self.dialate_p = dialate_p
        self.dialate_epochs = dialate_epochs
        self.output_folder = f'{os.path.join(os.getcwd(), output_folder)}_{datetime.now().strftime("%d_%m_-%H.%M")}'
        self.optimizer = optimizer
        self.test_set = test_set
        self.train_set = train_set
        self.val_set = val_set
        self.lr = lr
        self.base_features = base_features
        self.in_channels = in_channels
        self.num_batches = num_batches
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.loss_func = loss_func
        self.epochs = epochs
        self.train_loss = []
        self.val_loss = []
        self.network= network
        self.device = device
    

    def initialize(self):
        self.network = self.network(self.in_channels, base_features = self.base_features)
        self.network.to(self.device)
        self.optimizer = self.optimizer(self.network.parameters(), lr=self.lr)
        self.train_loader = DataLoader3D(self.train_set, self.patch_size, self.batch_size, dialate=self.dialate)
        self.test_loader = DataLoader(self.test_set, shuffle=True)
        self.val_loader = DataLoader3D(self.test_set, self.patch_size, self.batch_size, dialate=False)
        self.maybe_mkdir()
        self.create_log_file()


    def maybe_mkdir(self):
        if not os.path.exists(self.output_folder):
            os.mkdir(self.output_folder)
            os.mkdir(os.path.join(self.output_folder, 'Slices'))
            os.mkdir(os.path.join(self.output_folder, 'Loss'))


    def create_log_file(self):
        log = ( f"Test with the following parameters \nLoss funciton: {type(self.loss_func).__name__}\n"
                f"Optimizer: {type(self.optimizer).__name__}\nEpochs: {self.epochs}\nBatch_size: {self.batch_size}\n"
                f"Number of batches per epoch: {self.num_batches}\n"
                f"Base features: {self.base_features}\nDialation: {self.dialate}\n"
                f"      If true: on first {self.dialate_epochs} epochs"
                f" with p = {self.dialate_p}")
        file = open(os.path.join(self.output_folder, 'log_file.txt'), 'w')
        file.write(log)
        file.close()


    def make_predictions(self):
        return 0


    def save_slice_epoch(self,output, seg, epoch):
        save_slice(output[0][0].detach().cpu().numpy(), 
                    os.path.join(self.output_folder, f'Slices/seg_{epoch+1}.png'))
        save_slice(seg[0][0].detach().cpu().numpy(), 
                    os.path.join(self.output_folder, f'Slices/GT_{epoch+1}.png'))
        plt.close()

    def create_loss_output(self):
        np.savetxt(os.path.join(self.output_folder, 'Loss/Train_Loss.csv'), self.train_loss, 
                            delimiter=",", fmt='%s')
        np.savetxt(os.path.join(self.output_folder, 'Loss/Validation_Loss.csv'), self.val_loss, 
                            delimiter=",", fmt='%s')                 
        plt.plot(self.train_loss)
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.suptitle(f'Training loss per epoch 3DUnet with {type(self.loss_func).__name__}')
        plt.savefig(os.path.join(self.output_folder, 
                    os.path.join('Loss', f'Loss_{self.epochs}_{type(self.loss_func).__name__}')))
        plt.close()

        plt.plot(self.val_loss)
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.suptitle(f'Validation loss per epoch 3DUnet with {type(self.loss_func).__name__}')
        plt.savefig(os.path.join(self.output_folder, 
                    os.path.join('Loss', f'Loss_{self.epochs}_{type(self.loss_func).__name__}')))
        plt.close()

    def validate(self):
        with torch.no_grad():
            val_loss = []
            for i, image_set in enumerate(self.val_loader):
                image = image_set['data'].to(self.device)
                label = image_set['seg'].to(self.device)
                output = self.network(image)
                loss = self.loss_func(output, label)
                val_loss.append(loss.item())
                if i == self.val_loader.get_data_length() - 1:
                    break
        self.val_loss.append(np.mean(val_loss))

    def get_test_loss(self):
        with torch.no_grad():
            test_loss = []
            for i, image_set in enumerate(self.test_loader):
                image = image_set['data'].to(self.device)
                label = image_set['seg'].to(self.device)
                output = self.network(image)
                loss = self.loss_func(output, label)
                test_loss.append(loss.item())
                if i == self.test_loader.get_data_length() - 1:
                    break
        self.val_loss.append(np.mean(test_loss))
        raise NotImplementedError


    def train(self):
        self.loss = []
        for epoch in range(self.epochs):
            loss_here = []
            if epoch <= 20 and epoch % 2 == 0:
                self.train_loader.UpdateDialPad(-1)
            elif epoch == 20:
                self.train_loader.ToggleDialate(dialate = False)     
            for i, image_set in enumerate(self.train_loader):
                image = image_set['data'].to(self.device)
                labels = image_set['seg'].to(self.device)
                outputs = self.network(image)
                self.optimizer.zero_grad(set_to_none=True) 
                loss = self.loss_func(outputs, labels)
                loss_here.append(loss.item())
                loss.backward()
                self.optimizer.step()
                if i == self.num_batches - 1:
                    self.save_slice_epoch(outputs, labels, epoch)
                    break
            self.validate()
            self.train_loss.append(np.mean(loss_here))
        self.create_loss_output()