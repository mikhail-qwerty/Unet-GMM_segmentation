from segmentation import Unet_segmentation, GMM_predict_labels, fit_GMM
from data import recon_3D, circle_mask, load_h5_dataset, prepare_patches, adjust_data
from Unet_models import UNet_3D

import os
import numpy as np
import importlib
import matplotlib.pyplot as plt

tf = importlib.import_module("tensorflow")
h5 = importlib.import_module("h5py")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# set mixed precision (32/16 bit) presision
tf.keras.mixed_precision.experimental.set_policy('mixed_float16')

def main():
    # load data from h5 file
    print('load data from h5 file')
    data_path = '/home/m_fokin/NN/Data/data_exp2/exp2_time_19p6_101_102_107to110_113to185.h5'
    data = load_h5_dataset(data_path)
    patches, patches_shape = prepare_patches(data)

    print(data.max())    
    print('load Unet model and segment grains')
    # load Unet model and segment grains
    model = UNet_3D(input_size=(256, 256, 256, 1))
    model.load_weights("/home/m_fokin/NN/Unet_3D_2D/weights/UNet_3D__16_256.hdf5")
    grains = Unet_segmentation(patches, model, 1, patches_shape)
    grains = recon_3D(grains, (128,242,242), (256,256,256), (512,1224,1224))

    grains[grains >= 0.8] = 1
    grains[grains < 0.8] = 0

    print('generate and apply circle mask, extract grains')
    # generate and apply circle mask, extract grains
    mask = circle_mask(size = 1224, radius = 599)
    mask = np.uint8(mask * (1 - grains))


    print('load GMM and make clusterization')
    # load GMM and make clusterization
    gmm = fit_GMM(data, 50, mask)
    result = GMM_predict_labels(gmm, data, mask)


    plt.figure(figsize = (10,10))
    plt.imshow(result[256])
    plt.savefig('clusters.png')

if __name__ == "__main__":
    main()