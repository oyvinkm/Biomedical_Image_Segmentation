from typing import OrderedDict
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
import torch.optim.lr_scheduler as lr_s
import numpy as np
import csv
from Image_Functions import save_slice, show_slices, save_nii
from matplotlib import pyplot as plt
from datetime import datetime
from distutils.dir_util import copy_tree
#Text

class NetworkTrainer():
    def __init__(self, device,  network : nn.Module, epochs : int, image_path,
                loss_func : nn.Module, batch_size : int, patch_size : tuple, num_batches : int,
                lr : float, train_set : Set, test_set : Set, val_set : Set, optimizer : Optimizer, 
                output_folder : str, affine_sub2, affine_sub1, model_kwargs : OrderedDict, 
                dialate_p : float = 0.6, dialate_epochs : int = 20, dialate : bool = True,
                lr_schedule : str = None):
        self.start = datetime.now()
        self.start_string = self.start.strftime("%d/%m/%Y %H:%M:%S")
        self.lr_schedule = lr_schedule
        self.dialate = dialate
        self.dialate_p = dialate_p
        self.dialate_epochs = dialate_epochs
        self.output_folder = f'{os.path.join(os.getcwd(), output_folder)}_{datetime.now().strftime("%d_%m_-%H.%M")}'
        self.optimizer = optimizer
        self.test_img_affine_sub2 = affine_sub2
        self.test_img_affine_sub1 = affine_sub1
        self.test_set = test_set
        self.train_set = train_set
        self.val_set = val_set
        self.lr = lr
        self.model_kwargs = model_kwargs
        self.num_batches = num_batches
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.loss_func = loss_func
        self.epochs = epochs
        self.image_path = image_path
        self.train_loss = []
        self.val_loss = []
        self.test_loss = []
        self.learning_rate = []
        self.accuracy = []
        self.network= network
        self.device = device

    def initialize(self):
        self.network = self.network(**self.model_kwargs)
        self.network.to(self.device)
        self.optimizer = self.optimizer(self.network.parameters(), lr=self.lr)
        if self.lr_schedule is not None:
            self.schduler = self.interchangable_lr(self.optimizer)
        else: 
            self.schduler = None
        self.train_loader = DataLoader3D(self.train_set, self.batch_size, patch_size=self.patch_size, dialate=self.dialate)
        self.test_loader = DataLoader3D(self.test_set, 1, is_test=True)
        self.val_loader = DataLoader3D(self.test_set, self.batch_size, patch_size=self.patch_size, dialate=False)
        self.maybe_mkdir()
        self.create_log_file()
        self.write_tofile('Loss/Loss.csv', ('Train', 'Validation'))

    def interchangable_lr(self, verb = False, gamma=0.9):
        if self.lr_schedule == 'Exponential':
            return (lr_s.ExponentialLR(optimizer=self.optimizer, gamma=gamma, verbose=False))
        elif self.lr_schedule == 'Lambda': 
            lambda_1 = lambda epoch: (1-(epoch/self.epochs))**gamma
            return lr_s.LambdaLR(self.optimizer, lr_lambda=[lambda_1], verbose=False)
        elif self.lr_schedule == 'ReducePlateau':
            return lr_s.ReduceLROnPlateau(self.optimizer, 'min', verbose=False)
        elif self.lr_schedule == 'Linear':
            return lr_s.LinearLR(self.optimizer, start_factor=0.4, total_iters=self.epochs - 1, verbose=False)

    def maybe_mkdir(self):
        if not os.path.exists(self.output_folder):
            os.mkdir(self.output_folder)
            os.mkdir(os.path.join(self.output_folder, 'Slices'))
            os.mkdir(os.path.join(self.output_folder, 'Test'))
            os.mkdir(os.path.join(self.output_folder, 'Loss'))
            os.mkdir(os.path.join(self.output_folder, 'Accuracy'))
    


    def create_log_file(self):
        log = ( f"Test {self.start_string} with the following parameters \nLoss funciton: {type(self.loss_func).__name__}, param: {self.loss_func.get_fields()}\n"
                f"Optimizer: {type(self.optimizer).__name__}\nLearning Rate Schedule: {self.lr_schedule}\n"
                f"Epochs: {self.epochs}\nBatch_size: {self.batch_size}\n"
                f"Number of batches per epoch: {self.num_batches}\n"
                f"Model parameters: {self.model_kwargs}\nDialation: {self.dialate}\n"
                f"      If true: on first {self.dialate_epochs} epochs"
                f" with p = {self.dialate_p}\n")
        file = open(os.path.join(self.output_folder, 'log_file.txt'), 'w')
        file.write(log)
        file.close()

    def finish(self):
        with open(os.path.join(self.output_folder, 'log_file.txt'), 'a') as f:
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            f.write(f'\nFinished at {dt_string}, time elapsed: {now - self.start}')
            f.close()


    def make_predictions(self):
        return 0

    def save_test_nii(self, output, seg, num, key):
        if 'sub-1' in key:
            affine = self.test_img_affine_sub1
        else: 
            affine = self.test_img_affine_sub2
        key = str.replace(key, 'numpy_', '')
        os.mkdir(os.path.join(self.output_folder, f'Test/{key}'))
        copy_tree(os.path.join(self.image_path, key), os.path.join(self.output_folder, f'Test/{key}'))
        save_nii(output[0][0].detach().cpu().numpy(),affine=affine, name=os.path.join(self.output_folder, 
                f'Test/{key}/{num+1}uncertain_SEG.nii.gz'))
        threshold = torch.tensor([0.5])
        out = output[0][0].detach().cpu()
        result = (out>threshold).float()
        save_nii(result.numpy(), affine=affine, name=os.path.join(self.output_folder, 
                f'Test/{key}/{num+1}certain_SEG.nii.gz'))
        save_nii(seg[0][0].detach().cpu().numpy(), 
                affine=affine,
                name=os.path.join(self.output_folder, f'Test/{key}/{num+1}_GT.nii.gz'))
        plt.close()

    def save_slice_epoch(self,output, seg, epoch):
        save_slice(output[0][0].detach().cpu().numpy(), 
                    os.path.join(self.output_folder, f'Slices/{epoch+1}_SEG.png'))
        save_slice(seg[0][0].detach().cpu().numpy(), 
                    os.path.join(self.output_folder, f'Slices/{epoch+1}_GT.png'))
        plt.close()

    def write_tofile(self, file, loss):
        with open(os.path.join(self.output_folder, file), 'a') as f:
            csv.register_dialect("custom", delimiter=",", skipinitialspace=True)
            writer = csv.writer(f, dialect="custom")
            writer.writerow(loss)
            f.close()

    def create_loss_output(self):
        plt.plot(self.learning_rate)
        plt.ylabel('Learning Rate')
        plt.xlabel('Epochs')
        plt.suptitle(f'Learning Rate 3DUnet with {self.lr_schedule}')
        plt.savefig(os.path.join(self.output_folder, 'Learning_Rate'))
        plt.close()
        np.savetxt(os.path.join(self.output_folder, 'Loss/Test_Loss.csv'), self.test_loss, 
                          delimiter=",", fmt='%s')
        plt.plot(self.train_loss)
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.suptitle(f'Training loss per epoch 3DUnet with {type(self.loss_func).__name__}')
        plt.savefig(os.path.join(self.output_folder, 
                    os.path.join('Loss', f'Train_oss_{self.epochs}_{type(self.loss_func).__name__}')))
        plt.close()

        plt.plot(self.val_loss)
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.suptitle(f'Validation loss per epoch 3DUnet with {type(self.loss_func).__name__}')
        plt.savefig(os.path.join(self.output_folder, 
                    os.path.join('Loss', f'Val_Loss_{self.epochs}_{type(self.loss_func).__name__}')))
        plt.close()

        plt.plot(self.test_loss)
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.suptitle(f'Test loss per epoch 3DUnet with {type(self.loss_func).__name__}')
        plt.savefig(os.path.join(self.output_folder, 
                    os.path.join('Loss', f'Test_Loss_{self.epochs}_{type(self.loss_func).__name__}')))
        plt.close()

    def create_accuracy_output(self):
        plt.bar([i+1 for i in range(len(self.accuracy))],self.accuracy, log = True)
        plt.ylabel('Accuracy')
        plt.xlabel('pic number')
        plt.suptitle(f'Accuracy for each picture during testing')
        plt.savefig(os.path.join(self.output_folder,
                    os.path.join('Accuracy', f'Accuracy_{self.epochs}_{type(self.loss_func).__name__}')))
        np.savetxt(os.path.join(self.output_folder, 'Accuracy/Average_accuracy.csv'), [np.mean(self.accuracy)], 
                          delimiter=",", fmt='%s')
        plt.close()
    def get_accuracy_by_dice(self, output, target):
        smooth = 1e-5
        threshold = torch.tensor([0.5]).to(self.device)
        inputs = (output[0][0]>threshold).float()
        inputs = inputs.view(-1)
        target = target.view(-1)
        intersection = (inputs*target).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + target.sum() + smooth)
        return dice.item()

    def validate(self, epoch):
        with torch.no_grad():
            val_loss = []
            accuracy = []
            for i, image_set in enumerate(self.val_loader):
                image = image_set['data'].to(self.device)
                label = image_set['seg'].to(self.device)
                output = self.network(image)
                loss = self.loss_func(output, label)
                val_loss.append(loss.item())
                if epoch % 10 == 0:
                    accuracy.append(self.get_accuracy_by_dice(output, label))
                if i == self.val_loader.get_data_length() - 1:
                    if epoch % 10 == 0:
                        self.write_tofile('Accuracy/Val_Acc.csv', [np.mean(accuracy)])
                        self.save_slice_epoch(output, label, epoch)
                    break
        self.val_loss.append(np.mean(val_loss))

    def test(self):
        with torch.no_grad():
            test_loss = []
            for i, image_set in enumerate(self.test_loader):
                image = image_set['data'].to(self.device)
                label = image_set['seg'].to(self.device)
                key = image_set['keys']
                output = self.network(image)
                loss = self.loss_func(output, label)
                test_loss.append(loss.item())
                self.save_test_nii(output, label, i, key)
                accuracy_item = self.get_accuracy_by_dice(output, label)
                self.accuracy.append(accuracy_item)
                if i == self.test_loader.get_data_length() - 1:
                    break
        self.create_accuracy_output()
        self.test_loss.append(np.mean(test_loss))
        self.create_loss_output()


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
                    break
            self.validate(epoch)
            self.train_loss.append(np.mean(loss_here))
            if self.schduler is not None:
                if isinstance(self.schduler, lr_s.ReduceLROnPlateau):
                    self.schduler.step(self.val_loss[-1])
                    self.learning_rate.append(self.optimizer.param_groups[0]['lr'])
                else:
                    self.schduler.step()
                    self.learning_rate.append(self.schduler.get_last_lr())
                    self.write_tofile('Learning_Rate.csv', self.learning_rate[-1])
            self.write_tofile('Loss/Loss.csv', (self.train_loss[-1], self.val_loss[-1]))

    def __call__(self):
        self.initialize()    
        self.train()
        self.test()
        self.finish()