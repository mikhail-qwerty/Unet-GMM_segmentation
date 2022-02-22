import random 
import numpy as np
import tifffile as tiff 
from sklearn.mixture import GaussianMixture
from tf.keras.preprocessing.image import ImageDataGenerator

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


def dataGenerator3D(im_list, label_list, batch_size):
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


def dataGenerator2D(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (512,512),seed = 1):
    

    image_datagen =  ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)


def readGMM(model_path, covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=100, n_init=1):
    # load GMM params: means, weights, covariances 
    means = np.load(f'{model_path}_means.npy')
    weights = np.load(f'{model_path}_weights.npy')
    cov = np.load(f'{model_path}_cov.npy')
    # initialize GMM 
    GMM = GaussianMixture(
            n_components = len(means),
            means_init = means, 
            covariance_type=covariance_type, 
            tol=tol, 
            reg_covar=reg_covar, 
            max_iter=max_iter, 
            n_init=n_init, 
            weights_init=weights, 
            means_init=means, 
            precisions_init=np.linalg.inv(cov))
    # set GMM params
    GMM.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(cov))
    GMM.weights_ = weights
    GMM.means_ = means
    return(GMM)


def writeGMM(model, model_name, save_path):
    # get GMM params
    means = model.means_
    weights = model.weights_
    cov = model.covariances_
    # save GMM params
    np.save(f'{save_path}{model_name}_means.npy', means, allow_pickle=False)
    np.save(f'{save_path}{model_name}_cov.npy', cov, allow_pickle=False)
    np.save(f'{save_path}{model_name}_weights.npy', weights, allow_pickle=False)