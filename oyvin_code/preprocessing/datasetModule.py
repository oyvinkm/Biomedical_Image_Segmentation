import nibabel as nib
import os
import numpy as np
from torch.utils.data import Dataset


class Set(Dataset):
    def __init__(self, data_path, folders, transform = None):
        self.data_path = data_path
        self.folders = folders
        self.transform = transform 
        self.folders_cont = []
        for i in range(len(self.folders)):
            self.folders_cont.append(self.get_folder_content(i))

    def get_folder_content(self, index):
        sub_folder = os.path.join(self.data_path, self.folders[index])
        data = sorted([name for name in os.listdir(sub_folder) if os.path.isfile(os.path.join(sub_folder,name)) and 'masked' in name])
        seg = sorted([name for name in os.listdir(sub_folder) if os.path.isfile(os.path.join(sub_folder,name)) and 'Lacunes' in name])
        return data, seg
        
    def __getitem__(self, index):
        path = os.path.join(self.data_path, self.folders[index])
        print(path)
        data, seg = self.folders_cont[index]
        sample = {}
        img_data = [nib.load(os.path.join(path,elm)).get_fdata() for elm in data]
        img_seg =  [nib.load(os.path.join(path,elm)).get_fdata() for elm in seg]
        img_seg = np.maximum(img_seg[0], img_seg[1])
        sample['data'] = np.array(img_data)
        sample['seg'] = np.expand_dims(np.array(img_seg), axis=0)
        sample['key'] = self.folders[index]
        img_data = None
        img_seg = None
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def __len__(self):
        return len(self.folders)       


