import nibabel as nib
import torch
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
from collections import OrderedDict
import SimpleITK as sitk
#from oyvin_code.preprocessing.preprocessing import resample_data_seg

def load_case_from_list_of_files(path, data_files, seg_file=None):
    assert isinstance(data_files, list) or isinstance(data_files, tuple), "case must be either a list or a tuple"
    properties = OrderedDict()
    data_itk = [sitk.ReadImage(os.path.join(path,f)) for f in data_files]

    properties["original_size_of_raw_data"] = np.array(data_itk[0].GetSize())[[2, 1, 0]]
    properties["original_spacing"] = np.array(data_itk[0].GetSpacing())[[2, 1, 0]]
    properties["list_of_data_files"] = data_files
    properties["seg_file"] = seg_file

    properties["itk_origin"] = data_itk[0].GetOrigin()
    properties["itk_spacing"] = data_itk[0].GetSpacing()
    properties["itk_direction"] = data_itk[0].GetDirection()

    data_npy = np.vstack([sitk.GetArrayFromImage(d)[None] for d in data_itk])
    if seg_file is not None:
        seg_itk = [sitk.ReadImage(os.path.join(path,f)) for f in seg_file]
        seg_npy =  np.vstack([sitk.GetArrayFromImage(d)[None] for d in seg_itk])
    else:
        seg_npy = None
    return data_npy.astype(np.float32), seg_npy.astype(np.float32), properties

class Set(Dataset):
    def __init__(self, data_path, resample = False, new_shape = None, new_spacing = None, to_tensor = True, sub_dir = 'sub-'):
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
        """ data, seg = self.get_folder_content(index)
        data_npy, seg_npy, properties = load_case_from_list_of_files(path, data, seg_file=seg)
        seg_npy = np.maximum(seg_npy[0], seg_npy[1])
        if self.resample:
            resampled_data = resample_data_seg(data_npy, ) """
        data, seg = self.get_folder_content(index)
        sample = {}
        img_data = [sitk.ReadImage(os.path.join(path, f)) for f in data]
        img_data = np.vstack(img_data)
        img_seg = [sitk.ReadImage(os.path.join(path, f)) for f in seg]
        img_seg = np.vstack(img_seg)
        img_seg = np.maximum(img_seg[0], img_seg[1])
        sample['data'] = torch.from_numpy(np.array(img_data))
        sample['seg'] = torch.from_numpy(np.array(img_seg)).unsqueeze(0)
        img_data.clear()
        return sample

    
    def __len__(self):
        return len(self.sub_folders)       


