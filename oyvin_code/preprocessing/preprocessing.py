from collections import OrderedDict
import numpy as np
from numpy.core.fromnumeric import reshape
from skimage.transform import resize
from scipy.ndimage.interpolation import map_coordinates
import nibabel as nib
import SimpleITK as sitk
import os
from multiprocessing.pool import Pool


def get_do_separate_z(spacing, anisotropy_threshold):
    do_seperate_z = (np.max(spacing)/np.min(spacing)) > anisotropy_threshold
    return do_seperate_z

def get_lowres_axis(new_spacing):
    axis = np.where(max(new_spacing)/np.array(new_spacing) == 1)[0]
    return axis

def load_case_from_list_of_files(data_files, path, seg_file=None):
    assert isinstance(data_files, list) or isinstance(data_files, tuple), "case must be either a list or a tuple"
    properties = OrderedDict()
    affine = nib.load(os.path.join(path, data_files[0])).affine
    data_itk = [sitk.ReadImage(os.path.join(path,f)) for f in data_files]
    properties["original_size_of_raw_data"] = np.array(data_itk[0].GetSize())
    properties["original_spacing"] = np.array(data_itk[0].GetSpacing())
    properties["list_of_data_files"] = data_files
    properties["seg_file"] = seg_file
    properties["affine"] = affine
    affine = None
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




def do_separate_axis(original_spacing, target_spacing, anisotropy_treshold):
    if get_do_separate_z(original_spacing, anisotropy_treshold):
        do_separate_z = True
        axis = get_lowres_axis(original_spacing)
    elif get_do_separate_z(original_spacing, anisotropy_treshold):
        do_separate_z = True
        axis = get_lowres_axis(target_spacing)
    else:
        do_separate_z = False
        axis = None
    return do_separate_z, axis

def resample_data_seg(data, new_shape, is_seg = False, axis=None, order=3, do_seperate_z=False, order_z=0):
    """"
    seperate_z = True will resample with order 0 along z
    """

    assert len(data.shape) == 4,  "Data must be shape (c, x, y, z)"
    
    if is_seg:
        resize_fn = resize
        kwargs = OrderedDict()
    else: 
        resize_fn = resize
        kwargs = {'mode': 'edge', 'anti_aliasing': False}
    dtype_data = data.dtype
    shape = np.array(data[0].shape)
    new_shape = np.array(new_shape)
    if np.any(shape != new_shape):
        data = data.astype(float)
        if do_seperate_z:
            print('seperate z, order in z is', order_z, 'order inplane is', order)
            assert len(axis) == 1, 'only one anisotropic axis supported'
            axis = axis[0]
            
            if axis == 0:
                new_shape_2d = new_shape[1:]
            elif axis == 1:
                new_shape_2d = new_shape[[0,2]]
            else:
                new_shape_2d = new_shape[:-1]

            reshaped_final_data = []
            for c in range(data.shape[0]):
                reshaped_data = []
                for slice_id in range(shape[axis]):
                    if axis == 0:
                        reshaped_data.append(resize_fn(data[c, slice_id], new_shape_2d, order, **kwargs))
                    elif axis == 1:
                        reshaped_data.append(resize_fn(data[c, :, slice_id], new_shape_2d, order, **kwargs))
                    else:
                        reshaped_data.append(resize_fn(data[c, :, :, slice_id], new_shape_2d, order, **kwargs))
                reshaped_data = np.stack(reshaped_data, axis)
                if shape[axis] != new_shape[axis]:
                    rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
                    orig_rows, orig_cols, orig_dim = reshaped_data.shape

                    row_scale = float(orig_rows) / rows
                    col_scale = float(orig_cols) / cols
                    dim_scale = float(orig_dim) / dim

                    map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
                    map_rows = row_scale * (map_rows + 0.5) - 0.5
                    map_cols = col_scale * (map_cols + 0.5) - 0.5
                    map_dims = dim_scale * (map_dims + 0.5) - 0.5

                    coord_map = np.array([map_rows, map_cols, map_dims])
                    if not is_seg or order_z == 0:
                        reshaped_final_data.append(map_coordinates(reshaped_data,coord_map,order=order_z, 
                                                                mode='nearest')[None])
                    else:
                        unique_labels = np.unique(reshaped_data)
                        reshaped = np.zeros(new_shape, dtype=dtype_data)

                        for i, cl in enumerate(unique_labels):
                            reshaped_multihot = np.round(
                                map_coordinates((reshaped_data == cl).astype(float), coord_map, order=order_z, 
                                                mode='nearest'))
                            reshaped[reshaped_multihot > 0.5] = cl
                        reshaped_final_data.append(reshaped[None])
                else:
                    reshaped_final_data.append(reshaped_data[None])
            reshaped_final_data = np.vstack(reshaped_final_data)
        
        else:
            print('No seperate z, order ', order)
            reshaped = []
            for c in range(data.shape[0]):
                reshaped.append(resize_fn(data[c], new_shape, order, **kwargs)[None])
            reshaped_final_data = np.vstack(reshaped)
        return reshaped_final_data.astype(dtype_data)
    else:
        print('no resampling necessary')
        return data

def run_preprocessing(data, seg, path, comp_seg = True):
    data_stor, seg_stor, properties = load_case_from_list_of_files(data, path, seg)
    shape = data_stor[0].shape
    new_shape = np.array([128, 128, 128])
    target_spacing = ((shape / new_shape).astype(float) * properties['original_spacing']) 
    do_seperate_z, axis = do_separate_axis(properties['original_spacing'], target_spacing, anisotropy_treshold=1)
    print(1)
    new_data = resample_data_seg(data_stor, new_shape, is_seg=False, axis=axis, order=2, do_seperate_z= do_seperate_z, order_z = 3)
    print(2)
    new_seg = resample_data_seg(seg_stor, new_shape, is_seg=False, axis=axis, order=2, do_seperate_z=do_seperate_z, order_z=3)
    print(3)
    if comp_seg:
        assert len(new_seg)  == 2, 'Can only compress segmentation with two channels'
        assert new_seg[0].shape == new_seg[1].shape, 'Segmentation channels needs to have the same shape'
        new_seg = np.maximum(new_seg[0], new_seg[1])
    print(4)
    return new_data, new_seg

