from matplotlib import pyplot as plt
import numpy as np
import torchio as tio

def show_slices(slices):
   """ Function to display row of image slices """
   fig, axes = plt.subplots(1, len(slices))
   for i, slice in enumerate(slices):
       axes[i].imshow(slice.T, cmap='gray', origin="lower")
def slicing(img, x : int = 100, y : int = 150, z : int = 80):
    slice_0 = img[100, :, :]
    slice_1 = img[:, 150, :]
    slice_2 = img[:, :, 80]
    show_slices([slice_0, slice_1, slice_2])
    plt.suptitle("Center slices for EPI image") 
    plt.show()

def extract_brain_region(image, brain_mask, seg, background=0.):
	''' find the boundary of the brain region, return the resized brain image and the index of the boundaries'''    
	brain = np.where(brain_mask != background)
	min_z = int(np.min(brain[1]))
	max_z = int(np.max(brain[1]))+1
	min_y = int(np.min(brain[2]))
	max_y = int(np.max(brain[2]))+1
	min_x = int(np.min(brain[3]))
	max_x = int(np.max(brain[3]))+1
	# resize image
	resizer = (slice(0,3),slice(min_z, max_z), slice(min_y, max_y), slice(min_x, max_x))
	return image[resizer], seg[resizer], [[min_z, max_z], [min_y, max_y], [min_x, max_x]]


def crop_images_to_brain(data_set):
    min_z, min_y, min_x = np.inf, np.inf, np.inf
    for i in range(len(data_set)):
        elm = data_set[i]
        cropped_img, cropped_seg, new_size = extract_brain_region(elm['data'], elm['data'], elm['seg'])
        if (new_size[0][1] - new_size[0][0] < min_z): min_z = new_size[0][1] - new_size[0][0]
        if (new_size[1][1] - new_size[1][0] < min_y): min_y = new_size[1][1] - new_size[1][0]
        if (new_size[2][1] - new_size[2][0] < min_x): min_x = new_size[2][1] - new_size[2][0]
        data_set[i]['data'], data_set[i]['seg'] = cropped_img, cropped_seg
    resizer = (slice(0,3),slice(0, min_z), slice(0, min_y), slice(0, min_x))
    for i in range(len(data_set)):
        data_set[i]['data'], data_set[i]['seg'] = data_set[i]['data'][resizer], data_set[i]['seg'][resizer]
    return data_set

def crop_to_size(set, size):
    for i in range(len(set)):
        transform = tio.CropOrPad((size))
        set[i]['data'], set[i]['seg'] = transform(set[i]['data']), transform(set[i]['seg'])
    return set
