import nibabel as nib
import torch
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
from preprocessing.preprocessing import run_preprocessing


class Set(Dataset):
    def __init__(self, data_path, to_tensor = True, sub_dir = 'sub-'):
        self.data_path = data_path
        self.sub_dir = sub_dir
        self.sub_folders = sorted([folder for folder in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, folder)) and self.sub_dir in folder])
        self.to_tensor = to_tensor 

    def get_folder_content(self, index):
        sub_folder = os.path.join(self.data_path, self.sub_folders[index])
        data = sorted([name for name in os.listdir(sub_folder) if os.path.isfile(os.path.join(sub_folder,name)) and 'masked' in name])
        seg = sorted([name for name in os.listdir(sub_folder) if os.path.isfile(os.path.join(sub_folder,name)) and 'Lacunes' in name])
        return data, seg
        
    def __getitem__(self, index):
        path = os.path.join(self.data_path, self.sub_folders[index])
        data, seg = self.get_folder_content(index)
        sample = {}
        img_data = [nib.load(os.path.join(path,elm)).get_fdata() for elm in data]
        img_seg =  [nib.load(os.path.join(path,elm)).get_fdata() for elm in seg]
        img_seg = np.maximum(img_seg[0], img_seg[1])
        sample['data'] = np.array(img_data)
        sample['seg'] = np.expand_dims(np.array(img_seg), axis=0)
        sample['key'] = self.sub_folders[index]
        img_data = None
        img_seg = None
        return sample
    
    def __len__(self):
        return len(self.sub_folders)       


