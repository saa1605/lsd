import numpy as np 
modes = ['train', 'dev', 'test']
for mode in modes:
    target_path = f"/data1/saaket/lsd_data/data/processed/sdr_{mode}_perspective_targets_x_y.npy" 
    gaussian_target_path = f"/data1/saaket/lsd_data/data/processed/{mode}_gaussian_targets.npy"
    targets = np.load(target_path, allow_pickle=True)
    gaussian_targets = np.load(gaussian_target_path, allow_pickle=True)
    print(len(gaussian_targets))
    # for i, target in enumerate(targets):
    #     pano_id = list(target.keys())[0]
    #     target_slice_number, target_x, target_y = target[pano_id]
    #     print( np.unravel_index(gaussian_targets[i].argmax(), gaussian_targets[i].shape) )
    #     print( target_y, target_x + (460) * target_slice_number )
        # assert np.unravel_index(gaussian_targets[i].argmax(), gaussian_targets[i].shape) == (0, target_y, target_x + (460) * target_slice_number)
        
    break
            

