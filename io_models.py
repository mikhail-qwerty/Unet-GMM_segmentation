import numpy as np
from sklearn.mixture import GaussianMixture


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