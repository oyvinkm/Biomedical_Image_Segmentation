#from oyvin_code.Image_Functions import slicing, crop_to_size, crop_images_to_brain
import nibabel as nib
import numpy as np
import os
import shutil
import torchio as tio


dir_path = 'Task3/'
new_dir = 'Cropped_Task3/'



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

    def extract_brain_region(image, brain_mask, background=0.):
        ''' find the boundary of the brain region, return the resized brain image and the index of the boundaries'''    
        brain = np.where(brain_mask != background)
        min_z = int(np.min(brain[0]))
        max_z = int(np.max(brain[0]))+1
        min_y = int(np.min(brain[1]))
        max_y = int(np.max(brain[1]))+1
        min_x = int(np.min(brain[2]))
        max_x = int(np.max(brain[2]))+1
        # resize image
        resizer = (slice(min_z, max_z), slice(min_y, max_y), slice(min_x, max_x))
        return resizer

    def make_new_folder(self, folder):
        try:
            os.mkdir(os.path.join(self.new_dir, 'crop_' + folder))
        except:
            print('Folder allready exists')

    def save_image(self, data, affine, header, path):
        cropped_img = nib.Nifti1Image(data, affine, header)
        nib.save(cropped_img, path)

    def crop_and_save(self):
        for indx in range(len(self.sub_folders)):
            self.make_new_folder(self.sub_folders[indx])
            path = os.path.join(self.data_path, self.sub_folders[indx])
            new_path = os.path.join(self.new_dir, 'crop_'+self.sub_folders[indx])
            print(self.sub_folders[indx])
            data,seg = self.get_folder_content(indx)
            img = nib.load(os.path.join(path, data[0]))
            img_data = img.get_fdata()
            new_size = self.extract_brain_region(img_data)
            img = None
            img_data = None
            for elm in data:
                image = nib.load(os.path.join(path, elm))
                img_data = image.get_fdata()
                img_data = img_data[new_size]
                self.save_image(img_data, image.affine, image.header, os.path.join(new_path, 'crop_' + elm))
            for elm in seg:
                image = nib.load(os.path.join(path, elm))
                img_data = image.get_fdata()
                max_pre = img_data.max()
                img_data = img_data[new_size]
                max_post = img_data.max()
                assert max_pre == max_post, print('Data Lost when cropping Segmentation image: %s' % elm)
                self.save_image(img_data, image.affine, image.header, os.path.join(new_path, 'crop_' + elm))  

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



test = Crop_And_Save(dir_path, new_dir, sub_dir='sub-2')
test.crop_and_save()


