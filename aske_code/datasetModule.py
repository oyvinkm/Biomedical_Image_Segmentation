import nibabel as nib
import torch
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset


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
        print(path)
        data, seg = self.get_folder_content(index)
        sample = {}
        img_data = []
        img_seg = []
        for elm in data:
            npy_data = nib.load(os.path.join(path,elm))
            tmp = npy_data.get_fdata()
            img_data.append(tmp)
        for elm in seg:
            npy_data = nib.load(os.path.join(path,elm))
            tmp = npy_data.get_fdata()
            img_seg.append((tmp))
        sample['data'] = torch.from_numpy(np.array(img_data))#.type(torch.DoubleTensor)
        sample['seg'] = torch.from_numpy(np.array(img_seg)).type(torch.LongTensor)[0]
        #One has to unsqueeze regardless of lossfunction
        sample['seg'] = sample['seg'].unsqueeze(0)#.add(sample['seg'][1]).unsqueeze(0)
        print("!!!ONLY 1/2 SEGMENTATION IS USED!!!")
        #sample['seg'] = sample['seg'].type(torch.LongTensor)
        img_seg.clear()
        img_data.clear()
        return sample
    def __len__(self):
        return len(self.sub_folders)       