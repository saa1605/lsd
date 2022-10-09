import os 
import numpy as np 
from scipy.ndimage import gaussian_filter
from tqdm import tqdm 
from joblib import Parallel, delayed

def create_gaussian_targets(targets, i):
    '''Create Gaussian Targets of size HxWx1 corresponding to SDR target location. The created targets correspond to joined perspective images'''
    target = targets[i]
    gaussian_target = np.zeros(
    (num_slices, H, W)
    )
    pano_id = list(target.keys())[0]
    target_slice_number, target_x, target_y = target[pano_id]

    gaussian_target[target_slice_number, target_y, target_x] = 1
    gaussian_target = np.transpose(gaussian_target, (1, 0, 2))
    gaussian_target = np.expand_dims(np.reshape(gaussian_target, (H, num_slices*W)), 0)
    gaussian_target[0, :, :] = gaussian_filter(
        gaussian_target[0, :, :],
        sigma=40,
    )
    gaussian_target[0, :, :]= (
        gaussian_target[0, :, :]
        / gaussian_target.sum()
    )
    return gaussian_target
    
if __name__ == "__main__":  
    modes = ['dev', 'test']
    H = 800
    W = 460 
    num_slices = 8
    for mode in modes:
        target_path = f'/data1/saaket/lsd_data/data/processed/sdr_{mode}_perspective_targets_x_y.npy'
        data_path = f'/data1/saaket/touchdown/data/{mode}.json'
        targets = np.load(target_path, allow_pickle=True)
    
        gaussian_targets = Parallel(n_jobs=16)(
            delayed(create_gaussian_targets)(targets, i) for i in tqdm(range(len(targets)))
        )
        np.save(f'/data1/saaket/lsd_data/data/processed/{mode}_gaussian_targets.npy', gaussian_targets)
        