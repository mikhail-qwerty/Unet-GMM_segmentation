import random 
import numpy as np
import tifffile as tiff 
from sklearn.mixture import GaussianMixture
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def adjustData(im,mask):
    # normalize data
    if(np.max(im) > 1):
        im = im / 255
    if(np.max(mask) > 1):
        mask = mask /255
    # reshape data (z, x, y, ncol)
    im = np.reshape(im,im.shape + (1,))
    mask = np.reshape(mask, mask.shape + (1,))
    return (im,mask)


def dataGenerator(im_list, label_list, batch_size):
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
            im, mask = adjustData(im, mask)
            # add to batch  
            image_batch.append(im)
            mask_batch.append(mask)
            i += 1
        yield (np.array(image_batch), np.array(mask_batch))


def writeGMM(model, model_name, save_path):
    # get GMM params
    means = model.means_
    weights = model.weights_
    cov = model.covariances_
    # save GMM params
    np.save(f'{save_path}{model_name}_means.npy', means, allow_pickle=False)
    np.save(f'{save_path}{model_name}_cov.npy', cov, allow_pickle=False)
    np.save(f'{save_path}{model_name}_weights.npy', weights, allow_pickle=False)