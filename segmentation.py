import numpy as np
from sklearn.mixture import GaussianMixture


def Unet_segmentation(patch_data, model, batch_size, patches_shape):
    if patch_data.max() > 1:
        data = patch_data/patch_data.max()
    grains = model.predict(patch_data, verbose=1, batch_size = batch_size)
    grains = grains.reshape(patches_shape)
    return grains


def fit_GMM(data, step, mask, means_init = [[25], [75], [125], [160]]):
    # apply mask and transform 3D to 1D
    data_1d = data[mask.astype(bool)]
    # Fit GMM
    gmm = GaussianMixture(n_components = 4, max_iter=200, tol=1e-3, means_init = means_init, 
                    random_state=1, covariance_type = 'full', verbose = 0) 
    gmm = gmm.fit(X = np.expand_dims(data_1d[::step],1))
    return gmm


def GMM_predict_labels(model, data, mask):
    # apply mask and transform 3D to 1D
    data_1d = data[mask.astype(bool)]
    result = np.zeros(data.shape, dtype = np.uint8)
    # predict labels
    gmm_labels = model.predict(np.expand_dims(data_1d,1))
    print('labels:',gmm_labels.min(), gmm_labels.max())
    # reshape 1D array with labels to 3D mask 
    result[np.where(mask)] = gmm_labels
    return result