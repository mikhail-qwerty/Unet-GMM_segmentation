import random 
import h5py as h5
import numpy as np
import tifffile as tiff
import patchify as patch 
import numpy.lib.stride_tricks as stride_tricks
from glob import glob


def circle_mask(size, radius):
    mask = np.zeros([size, size, size], dtype = np.uint8)
    X, Y = np.ogrid[:size, :size]
    dist_from_center = np.sqrt((X - size // 2)**2 + (Y - size // 2)**2, dtype = np.float32)
    mask = dist_from_center <= radius
    return mask


def univariate_normal(x, mean, variance):
    return ((1. / np.sqrt(2 * np.pi * variance)) * 
            np.exp(-(x - mean)**2 / (2 * variance)))


def adjust_data(im,mask):
    # normalize data
    if(np.max(im) > 1):
        im = im / im.max()
    if(np.max(mask) > 1):
        mask = mask /mask.max()
    # reshape data (z, y, x, ncol)
    im = np.reshape(im,im.shape + (1,))
    mask = np.reshape(mask, mask.shape + (1,))
    return (im,mask)


def dataGenerator_3D(im_list, label_list, batch_size):
    i = 0 
    while True:
        image_batch = []
        mask_batch = []
        for b in range(batch_size):
            # shuffle images after list ends
            if i == len(im_list):
                i = 0
                data_list = list(zip(im_list, label_list))
                random.shuffle(data_list)
                im_list, label_list = zip(*data_list)
            # read images, normalize and reshape
            sample_im = im_list[i]
            sample_mask = label_list[i]
            im = tiff.imread(sample_im)
            mask = tiff.imread(sample_mask)
            im, mask = adjust_data(im, mask)
            # add to batch  
            image_batch.append(im)
            mask_batch.append(mask)
            i += 1
        yield (np.array(image_batch), np.array(mask_batch))


def prepare_patches(data, patch_size=(256,256,256), patch_step=(128,242,242)):
    patch_data = patch.patchify(data, patch_size, patch_step)
    patch_shape = patch_data.shape
    n = patch_data.shape[0] * patch_data.shape[1] * patch_data.shape[2]
    patch_data = patch_data.reshape((n, patch_data.shape[3], patch_data.shape[4], patch_data.shape[5],1))
    return patch_data, patch_shape

'''
def recon_3D(data_patches, patch_step, patch_size, recon_shape):
    # initialize arrays 
    rec = np.zeros(recon_shape)
    recon_pattern = np.zeros(recon_shape)
    patched_pattern = patch.patchify(np.ones(recon_shape), patch_size, patch_step)
    # reconstruct 3D volume from overalaped patches 
    for i in range(data_patches.shape[0]):
        for j in range(data_patches.shape[1]):
            for k in range(data_patches.shape[2]):
                recon_pattern[i*patch_step[0]:i*patch_step[0] + patch_size[0], j*patch_step[1]:j*patch_step[1] + 
                patch_size[1], k*patch_step[2]:k*patch_step[2] + patch_size[2]] += patched_pattern[i,j,k,:,:,:]

                rec[i*patch_step[0]:i*patch_step[0] + patch_size[0], j*patch_step[1]:j*patch_step[1] +
                patch_size[1], k*patch_step[2]:k*patch_step[2] + patch_size[2]] += data_patches[i,j,k,:,:,:]
    return (rec / recon_pattern)
'''

def recon_3d(data_patches, patch_step, patch_size, recon_shape):
    # initialize arrays
    rec = np.zeros(recon_shape)
    recon_pattern = np.zeros(recon_shape)

    # compute the start and end indices for each patch
    i_starts = np.arange(0, recon_shape[0] - patch_size[0] + 1, patch_step[0])
    j_starts = np.arange(0, recon_shape[1] - patch_size[1] + 1, patch_step[1])
    k_starts = np.arange(0, recon_shape[2] - patch_size[2] + 1, patch_step[2])
    i_ends = i_starts + patch_size[0]
    j_ends = j_starts + patch_size[1]
    k_ends = k_starts + patch_size[2]

    # compute the sum of the recon_pattern and rec arrays for each patch
    for i, i_start, i_end in zip(range(len(i_starts)), i_starts, i_ends):
        for j, j_start, j_end in zip(range(len(j_starts)), j_starts, j_ends):
            for k, k_start, k_end in zip(range(len(k_starts)), k_starts, k_ends):
                recon_pattern[i_start:i_end, j_start:j_end, k_start:k_end] += 1
                rec[i_start:i_end, j_start:j_end, k_start:k_end] += data_patches[i, j, k]

    # divide rec by the recon_pattern to get the final result
    rec /= recon_pattern

    return rec

'''
def recon_3d(data_patches, patch_step, patch_size, recon_shape):
    # Initialize arrays
    rec = np.zeros(recon_shape)
    recon_pattern = np.zeros(recon_shape)

    # Compute the indices of the voxels covered by each patch
    i_indices = np.arange(patch_size[0])
    j_indices = np.arange(patch_size[1])
    k_indices = np.arange(patch_size[2])
    ijk_indices = np.ix_(i_indices, j_indices, k_indices)
    patch_indices = np.array(np.meshgrid(i_indices, j_indices, k_indices, indexing='ij'))
    voxel_indices = patch_indices[:, :, np.newaxis, np.newaxis] + patch_step[:, np.newaxis, np.newaxis, np.newaxis] * np.arange(data_patches.shape)[np.newaxis, np.newaxis, :, :, :]


    # Accumulate the reconstructed 3D volume and the number of times each voxel is reconstructed and covered by a patch
    np.add.at(rec, tuple(voxel_indices), data_patches)
    np.add.at(recon_pattern, tuple(voxel_indices), np.ones(data_patches.shape))
    
    # Return the element-wise division of the accumulated reconstructed 3D volume and the accumulated number of times each voxel is reconstructed
    return rec / recon_pattern
'''

def load_h5_dataset(data_path, dset_number = 25, dtype = np.uint8):
    with h5.File(data_path, 'r') as f:
        dsets = list(f.keys())
        data = np.array(f[dsets[dset_number]], dtype)
    return data


def load_tiff_slices(slices_path, im_size, dtype=np.uint8):
    flist = glob(f'{slices_path}/*tiff')
    result = np.zeros([len(flist), im_size, im_size], dtype=dtype)
    for i, fpath in enumerate(flist):
        try:
            result[i] = tiff.imread(fpath)
        except Exception as e:
            print(f"Error loading file {fpath}: {e}")
    return result


def load_tiff_volume():
    pass