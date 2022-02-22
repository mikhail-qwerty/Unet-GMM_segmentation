import random 
import numpy as np
import tifffile as tiff 
from sklearn.mixture import GaussianMixture


def data3D_generator(im_list, label_list, batch_size):
    
    i = 0 
    while True:
        
        image_batch = []
        mask_batch = []
        
        for b in range(batch_size):
            if i == len(im_list):
                
                i = 0
                data_list = list(zip(im_list, label_list))
                random.shuffle(data_list)
                im_list, label_list = zip(*data_list)
                
            sample_im = im_list[i]
            sample_mask = label_list[i]
            
            i += 1
            
            im = tiff.imread(sample_im)/255
            mask = tiff.imread(sample_mask)/255
            
            im = np.reshape(im,im.shape + (1,))
            mask = np.reshape(mask, mask.shape + (1,))
            
            image_batch.append(im)
            mask_batch.append(mask)
            
        yield (np.array(image_batch), np.array(mask_batch))


def read_GMM(model_name, load_path, covariance_type = 'spherical'):
    
    loaded_params = np.load(load_path + model_name + '_cov.npy')

    means = np.load(load_path + model_name + '_means.npy')
    weights = np.load(load_path + model_name + '_weights.npy')
    cov = np.load(load_path + model_name + '_cov.npy')

    loaded_gmm = GaussianMixture(n_components = len(means),means_init = means, covariance_type=covariance_type)

    loaded_gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(cov))
    loaded_gmm.weights_ = weights
    loaded_gmm.means_ = means

    return(loaded_gmm)


def write_GMM(model, model_name, save_path):
    
    means = model.means_
    weights = model.weights_
    cov = model.covariances_

    np.save(save_path + model_name + '_means' + '.npy', means, allow_pickle=False)
    np.save(save_path + model_name + '_cov' + '.npy', cov, allow_pickle=False)
    np.save(save_path + model_name + '_weights' + '.npy', weights, allow_pickle=False)