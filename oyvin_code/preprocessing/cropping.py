from matplotlib import pyplot as plt
import numpy as np
import torchio as tio
import os



def create_image_mask(data):
    from scipy.ndimage import binary_fill_holes
    assert len(data.shape) == 4 or len(data.shape) == 3, "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    nonzero_mask = np.zeros(data.shape[1:], dtype=bool)
    for c in range(data.shape[0]):
        this_mask = data[c] != 0
        nonzero_mask = nonzero_mask | this_mask
    nonzero_mask = binary_fill_holes(nonzero_mask)
    return nonzero_mask

def make_bbox(brain_mask, background=0.):
	''' find the boundary of the brain region, return the resized brain image and the index of the boundaries'''    
	brain = np.where(brain_mask != background)
	min_z = int(np.min(brain[0]))
	max_z = int(np.max(brain[0]))+1
	min_y = int(np.min(brain[1]))
	max_y = int(np.max(brain[1]))+1
	min_x = int(np.min(brain[2]))
	max_x = int(np.max(brain[2]))+1
	# resize image
	#resizer = (slice(0,3),slice(min_z, max_z), slice(min_y, max_y), slice(min_x, max_x))
	return [[min_z, max_z], [min_y, max_y], [min_x, max_x]]

def crop_to_box(image, bbox):
    assert len(image.shape) == 3, "only supports 3d images"
    resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
    return image[resizer]

def crop_img_seg(data, seg=None, outside=-1):

    nonzero_mask = create_image_mask(data)
    print(nonzero_mask.shape)
    bbox = make_bbox(nonzero_mask)
    
    cropped_data = []
    for d in range(data.shape[0]):
        cropped = crop_to_box(data[d], bbox)
        cropped_data.append(cropped[None])
    data = np.vstack(cropped_data)

    if seg is not None:
        cropped_seg = []
        for c in range(seg.shape[0]):
            cropped = crop_to_box(seg[c], bbox)
            cropped_seg.append(cropped[None])
        seg = np.vstack(cropped_seg)
    nonzero_mask = crop_to_box(nonzero_mask, bbox)[None]
    if seg is not None:
        seg[(seg == 0) & (nonzero_mask == 0)] = outside

    return data, seg, bbox


def make_new_folder(self, folder):
        try:
            os.mkdir(os.path.join('../../', folder))
        except:
            print('Folder allready exists')


class ImageCropper(object):
    def __init__(self,num_threads, output_folder = None):

        self.output_folder = output_folder
        self.num_threads = num_threads
        if self.output_folder is not None:
            make_new_folder(output_folder)

    @staticmethod
    def crop(data, props, seg=None):
        before_shape = data.shape
        data, seg, bbox = crop_img_seg(data, seg, outside=-1)
        after_shape = data.shape
        print("before crop:", before_shape, "after crop:", after_shape, "spacing:",
              np.array(props["original_spacing"]), "\n")