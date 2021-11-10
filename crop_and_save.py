from oyvin_code.Image_Functions import slicing, crop_to_size, crop_images_to_brain
import nibabel as nib
import numpy as np
import os
import shutil
import numpy as np
from numpy import save
from numpy import savez_compressed
import os
import nibabel as nib
dir_path = 'Cropped_Task3/'
new_dir = 'Numpy_Comp_Task3/'
seg_dir_path = 'Cropped_Task3/Segmentations/'
seg_new_dir = 'Numpy_Task3/Segmentations/'



class Crop_And_Save():
    def __init__(self, data_path, new_dir, sub_dir = 'sub-', parent_dir = ''):
        self.data_path = data_path
        self.parent_dir = parent_dir
        self.sub_dir = sub_dir
        self.new_dir = os.path.join(self.parent_dir, new_dir)
        self.sub_folders = sorted([folder for folder in os.listdir(self.data_path) \
            if os.path.isdir(os.path.join(self.data_path, folder)) \
            and self.sub_dir in folder])
        if not os.path.exists(self.new_dir):
            os.mkdir(os.path.join(self.parent_dir, self.new_dir))   

    def get_folder_content(self, index):
            sub_folder = os.path.join(self.data_path, self.sub_folders[index])
            data = sorted([name for name in os.listdir(sub_folder) \
                if os.path.isfile(os.path.join(sub_folder,name))\
                     and 'masked' in name])
            seg = sorted([name for name in os.listdir(sub_folder) \
                if os.path.isfile(os.path.join(sub_folder,name)) \
                    and 'Lacunes' in name]) 
            return data, seg

    def make_new_folder(self, folder):
        try:
            os.mkdir(os.path.join(self.new_dir, 'numpy_' + folder))
        except:
            print('Folder allready exists')

    def crop_and_save(self):
        for indx in range(len(self.sub_folders)):
            self.make_new_folder(self.sub_folders[indx])
            path = os.path.join(self.data_path, self.sub_folders[indx])
            new_path = os.path.join(self.new_dir, 'numpy_'+self.sub_folders[indx])
            print(self.sub_folders[indx])
            data,seg = self.get_folder_content(indx)
            for elm in data:
                image = nib.load(os.path.join(path, elm))
                img_data = image.get_fdata()
                elm = elm.replace('.nii.gz', '.npy')
                save(os.path.join(new_path, elm), img_data)
            for elm in seg:
                image = nib.load(os.path.join(path, elm))
                img_data = image.get_fdata()
                elm = elm.replace('.nii.gz', '.npy')
                save(os.path.join(new_path, elm), img_data)

    def undo(self):
        if os.path.exists(self.new_dir):
            if (len(os.listdir(self.new_dir)) == 0):
                os.rmdir(self.new_dir)
            else:
                print('Folder is not empty')
                input('Press enter to continue')
                try:
                    shutil.rmtree(self.new_dir)
                except OSError as e:
                    print("Error: %s - %s." % (e.filename, e.strerror))
        else:
            print('Folder does not exists')

    def create_head_folder(self):
        try:
            os.mkdir(os.path.join(self.parent_dir, self.new_dir))
        except:
            print('Doesnt work')




test = Crop_And_Save(seg_dir_path, seg_new_dir, sub_dir='sub-')
test.crop_and_save()


