from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import tifffile as tiff
import h5py as h5
import numpy as np
import cv2
import glob
import os

import random 
import patchify as patch 

from tqdm import tqdm

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
        im = im / 255
    if(np.max(mask) > 1):
        mask = mask /255
    # reshape data (z, x, y, ncol)
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