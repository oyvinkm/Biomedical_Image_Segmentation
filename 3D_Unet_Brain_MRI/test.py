
from matplotlib import pyplot as plt

import nibabel as nib
from nibabel.nifti1 import save
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
from skimage import (exposure, feature, filters, io, measure,
                      morphology, restoration, segmentation, transform,
                      util)
from scipy import ndimage as ndi
from Image_Functions import slicing
import os
import torchio as tio
import re
from sklearn.metrics import f1_score


def print_props(arr):
    print("shape: {}".format(arr.shape))
    print("dtype: {}".format(arr.dtype))
    print("range: ({}, {})\n".format(np.min(arr), np.max(arr)))

def save_image(arr, affine, name, transform, path = None):
    if path is not None:
        name = os.path.join(path, name)
    if len(arr.shape) == 3:
        x = np.expand_dims(arr, axis=0)
        x = transformer(x)
        img = nib.Nifti1Image(x[0], affine)
        print(img.shape)
        nib.save(img, f'{name}_{0}.nii.gz')
    else:
        x = arr
        x = transform(x)
        for i in range(len(x)):
            img = nib.Nifti1Image(x[i], affine)
            print(img.shape)
            nib.save(img, f'{name}_{i+1}.nii.gz')


def create_labels(gt):
    labels = np.int32(measure.label(gt, connectivity=3))
    labs = []
    unique = np.unique(labels)
    print(unique)
    for x in unique:
        if x != 0:
            lab = np.int32([labels == x])
            labs.append(lab[0])
    return np.asarray(labs)

def lacunes_found(labels, seg):
    label_accuracy = []
    for i in range(len(labels)):
        label_mask = np.where(labels[i] == 1)
        seg_mask = seg[label_mask]
        label_accuracy.append(seg_mask.sum()/labels[i].sum())
    return label_accuracy


folder_path = 'Test_Results/BigTest/FirstTest/Lambda/Test_7_Tversky/Test/'
folders = [folder for folder  in os.listdir(folder_path) if 
                        os.path.isdir(os.path.join(folder_path, folder))]
test_folder = 'Test_7_Tversky'
def get_images(sub_dir):
    path = os.path.join(folder_path, sub_dir)
    images = [image for image in os.listdir(path) if 
            os.path.isfile(os.path.join(path, image))]
    gt = [i for i in images if 'GT' in i]
    uncrt = [i for i in images if 'uncertain' in i]
    seg = [i for i in images if 'certain' in i and not 'uncertain' in i]
    shape = [i for i in images if 'FLAIR' in i]
    shape = nib.load(os.path.join(path, shape[0]))
    gt = nib.load(os.path.join(path, gt[0]))
    uncrt = nib.load(os.path.join(path, uncrt[0]))
    seg = nib.load(os.path.join(path, seg[0]))
    return gt.get_fdata(), uncrt.get_fdata(), seg.get_fdata(), gt.affine, shape.shape


path = os.path.join(os.getcwd(), 'Bachelor_Plots/Results/')
if not os.path.exists(os.path.join(path, test_folder)):
    os.mkdir(os.path.join(path, test_folder))
path = os.path.join(path, test_folder)
for i in range(len(folders)):
    print(folders[i])
    if not os.path.exists(os.path.join(path, folders[i])):
        os.mkdir(os.path.join(path, folders[i]))
    sub_path = os.path.join(path, folders[i])
    gt, uncrt, seg, affine, shape = get_images(folders[i])
    labels = create_labels(gt)
    labels_2 = create_labels(seg)
    flat_gt, flat_seg = gt.flatten(), seg.flatten()
    f1 = f1_score(flat_gt,flat_seg)
    accuracy = lacunes_found(labels, seg)
    accuracy_2 = np.array(lacunes_found(labels_2, gt))
    accuracy_2[accuracy_2 > 0] = -1
    accuracy_2 = accuracy_2 + 1
    file = open(os.path.join(sub_path, 'log_file.txt'), 'w')
    file.write(f'F1 Score: {f1}\nLacune Accuracy: {accuracy}\nMisclassified: {accuracy_2}')
    file.close()
    transformer = tio.CropOrPad(shape)
    save_image(labels, affine, f'{folders[i]}_Label', transformer, sub_path)
    save_image(labels_2, affine, f'{folders[i]}_seg_Labels', transformer, sub_path)
    save_image(uncrt, affine, f'{folders[i]}_uncrt', transformer, sub_path)
    save_image(seg, affine, f'{folders[i]}_seg', transformer, sub_path)
exit()
gt_img = nib.load(gt_path)
orig_shape = nib.load(original_shape)
orig_shape = orig_shape.get_fdata().shape

seg = nib.load(seg_path)
seg = seg.get_fdata()

transformer = tio.CropOrPad(orig_shape)
gt_np = gt_img.get_fdata()
gt_int = np.int32(gt_np)
#remove_holes = morphology.remove_small_holes(gt_int, 64, connectivity=3)
labels = create_labels(gt_np)
lacunes = lacunes_found(labels, seg)
save_image(seg, gt_img.affine, 'Seg', transformer)
save_image(labels, gt_img.affine, 'Label', transformer)
print(lacunes)
exit()
labels_1_mask = np.where(labels[0] == 1)
flat = np.asarray(labels_1_mask)

seg_mask = seg[labels_1_mask]


print(labels.shape)
save_image(labels, gt_img.affine, 'Test_Label', transformer)
save_image(remove_holes, gt_img.affine, 'Test_Filled_Label', transformer)
exit()
print(unique_labels)


uncrt = nib.load(uncertain_path)
uncrt = uncrt.get_fdata()

save_image(labels_float, gt_img.affine, 'Labels', transformer)
save_image(seg, gt_img.affine, 'Seg', transformer)
save_image(uncrt, gt_img.affine, 'Uncrt', transformer)






