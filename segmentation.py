import numpy as np 

def Unet_segmentation(patch_data, model, batch_size):
    segmented = np.zeros(patch_data.shape, dtype = np.float32)
    # check noramalization
    if (patch_data.max() > 1):
        patch_data = patch_data / patch_data.max()
    # reshape patch data [nz,nx,ny,z,x,y] to [N,z,x,y]
    N = patch_data.shape[0] * patch_data.shape[1] * patch_data.shape[2]
    patch_data_merged = patch_data.reshape((N, patch_data.shape[3], patch_data.shape[4], patch_data.shape[5]))
    # segment patch_data
    patch_data_merged  = np.reshape(patch_data_merged, patch_data_merged.shape + (1,))
    result = model.predict(patch_data_merged, verbose=1, batch_size = batch_size)
    segmented = result.reshape(patch_data.shape)
    return segmented

def GMM_predict_labels(model, data, data_1d, ids_zero, mask):
    result = np.zeros(data.shape, dtype = np.uint8)
    # predict labels
    gmm_labels = model.predict(np.expand_dims(data_1d,1))
    # reshape 1D arr with labels to 3D mask 
    result[np.where(data != 0)] = gmm_labels + 1
    result = result + mask
    result[ids_zero] = 2 
    return result