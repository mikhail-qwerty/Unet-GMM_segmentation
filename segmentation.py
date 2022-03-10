import numpy as np
import patchify as patch 

def circle_mask(size, radius):
    mask = np.zeros([size, size, size], dtype = np.uint8)
    X, Y = np.ogrid[:size, :size]
    dist_from_center = np.sqrt((X - size // 2)**2 + (Y - size // 2)**2, dtype = np.float32)
    mask = dist_from_center <= radius
    return mask

 def recon_3D(patches, step, patch_size, recon_size):

    rec = np.zeros(recon_size)
    pattern = np.ones(recon_size)
    rec_pattern = np.zeros(pattern.shape)

    p_pattern = patch.patchify(pattern, patch_size, step)
    
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            for k in range(patches.shape[2]):
                rec_pattern[i*step[0]:i*step[0] + patch_size[0], j*step[1]:j*step[1] + patch_size[1], k*step[2]:k*step[2] + patch_size[2]] += p_pattern[i,j,k,:,:,:]

    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            for k in range(patches.shape[2]):
                rec[i*step[0]:i*step[0] + patch_size[0], j*step[1]:j*step[1] + patch_size[1], k*step[2]:k*step[2] + patch_size[2]] += patches[i,j,k,:,:,:]
                
    return (rec / rec_pattern)


f =h5.File(data_path,'r')
dsets = list(f.keys())
data = np.array(f[dsets[dset_id]], dtype = np.uint8)
f.close()

mask = circle_mask(size = 1224, radius = 599)
data = data * mask

max_iter = 200
tolerance = 1e-3
step = 50

N_first = 13
N_last = 78

data_path = '/home/m_fokin/NN/Data/wo_grains_exp2.h5'


for i in range(N_first, N_last + 1):
    
    print('step', str(i), 'from', str(len(dsets) - 1))
###############################################################################################################
    f =h5.File(data_path,'r')
    dsets = list(f.keys())
    data = np.array(f[dsets[i]], dtype = np.uint8)
    f.close()
    
    data = data * mask
###############################################################################################################    
    hist_3D = np.zeros([256, 1], dtype = np.float32)
    for j in range (data.shape[0]):
        hist = cv2.calcHist([data[j,:,:]],[0],None,[256],[0,256])
        hist[0] = 0
        hist_3D = hist_3D + hist
###############################################################################################################    
    data_1d = data.ravel()
    data_1d = data_1d[data_1d != 0] 

    # Fit GMM
    gmm = GaussianMixture(n_components = 4, max_iter = max_iter, random_state = 1, tol = tolerance, covariance_type = 'full'
                          , means_init = means, weights_init = weights, precisions_init = np.linalg.inv(cov),verbose = 0)
    
    gmm = gmm.fit(X = np.expand_dims(data_1d[::step],1))
    
    gmm_x = np.linspace(0,255,256)
    gmm_y = np.exp(gmm.score_samples(gmm_x.reshape(-1,1)))
    
    
    g1 = univariate_normal(gmm_x, gmm.means_[0], gmm.covariances_[0][0]) * gmm.weights_[0]
    g2 = univariate_normal(gmm_x, gmm.means_[1], gmm.covariances_[1][0]) * gmm.weights_[1]
    g3 = univariate_normal(gmm_x, gmm.means_[2], gmm.covariances_[2][0]) * gmm.weights_[2]
    g4 = univariate_normal(gmm_x, gmm.means_[3], gmm.covariances_[3][0]) * gmm.weights_[3]

    norm = gmm_y.max()/hist_3D.max()
###############################################################################################################
    plt.figure(figsize = (10,7))
    plt.grid()
    plt.plot(g1, 'k--', label = "gaussians")
    plt.plot(g2, 'k--')
    plt.plot(g3, 'k--')
    plt.plot(g4, 'k--')
    plt.ylim(0, xmax)
    plt.xlim(0, 256)

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel('gray level', fontsize = 18)
    plt.plot(gmm_x, gmm_y, color="r", lw=2, label="sum of gaussians")
    plt.plot(hist_3D * norm, 'bo', label="image histogram")
    plt.legend(prop={'size': 24})
    
    plt.savefig('result_img_cov_4_1000it/' + 'gmm_result_exp2_' + dsets[i] + '.png')
    
    plt.clf()
    plt.close()
############################################################################################################### 
    cov = gmm.covariances_
    means = gmm.means_
    weights = gmm.weights_
###############################################################################################################
    write_GMM(model = gmm, model_name = dsets[i], save_path = 'GMM_models_cov_4_1000it/')



max_iter = 1000
tolerance = 1e-4
step = 50
N_first = 13
N_last = 78

fin_data = '/home/m_fokin/NN/Data/data_exp2/exp2_time_19p6_101_102_107to110_113to185.h5'
fin_grains = '/home/m_fokin/NN/Data/grains_exp2.h5'
fout = '/home/m_fokin/NN/Data/exp2_time_19p6_101_102_107to110_113to185_segmented_cov_4_200it.h5'
##############################################################################################################
for i in range(N_first, N_last + 1):
    
    # Load data from h5 file
    fdata = h5.File(fin_data ,'r')
    fdata_grains = h5.File(fin_grains,'r')
    dsets = list(fdata.keys())
    data = np.array(fdata[dsets[i]], dtype = np.uint8)
    grains = np.array(fdata_grains[dsets[i]], dtype = np.uint8)
    fdata.close()
    fdata_grains.close()
###############################################################################################################       
    # Prepare data for GMM
    ids_zero = np.where(data == 0)
    
    data = data * mask
    data = data * (1 - grains // 255)
    data_1d = data.ravel()
    data_1d = data_1d[data_1d != 0] 
###############################################################################################################
    # Read GMM
    gmm = read_GMM(model_name = dsets[i], load_path = 'GMM_models_cov_4_1000it/', covariance_type = 'full')
    gmm_labels = gmm.predict(np.expand_dims(data_1d,1))
###############################################################################################################
    # Fill array with segmented data)
    result = np.zeros(data.shape, dtype = np.uint8)
    
    result[np.where(data != 0)] = gmm_labels + 1
    result = result + mask
    result[ids_zero ] = 2 
###############################################################################################################    
    fdata_out = h5.File(fout, 'a')
    fdata_out.create_dataset(dsets[i], data = np.uint8(result))
    fdata_out.close()
    
    tiff.imwrite('/home/m_fokin/NN/Data/exp_2_segmented_tiff_4_/' + dsets[i] + '.tiff', result[256, :, :])
###############################################################################################################
#    result_proba_3 = np.zeros(data.shape + (3,), dtype = np.float32)
#    result_proba_1 = np.zeros(data.shape, dtype = np.float32)
#    
#    gmm_proba = gmm.predict_proba(np.expand_dims(data_1d,1))
#    
#    for j in range(3):
#        print(j)
#        result_proba_1[np.where(data != 0)] = gmm_proba[:,j]
#        if j == 1: 
#            result_proba_1[ids] = 1
#        else:
#            result_proba_1[ids] = 0
#        result_proba_3[:,:,:,j] = result_proba_1
#        
#    tiff.imwrite('/home/m_fokin/NN/Data/exp2_proba_tiff/' + dsets[i] + '.tiff', result_proba_3[256, :, :, :])    