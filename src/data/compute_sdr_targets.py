import numpy as np
import json 
import cv2 
from tqdm import tqdm 

h_stride = 45
nn_heading_angles = list(range(-180, 180, h_stride))

parameter_matrices = {}



def find_closest_index(arr, num):
    diff_arr = np.absolute(arr - num)
    return np.unravel_index(diff_arr.argmin(), diff_arr.shape)

# Persepctive transforming the image 
def get_perspective(FOV, THETA, PHI, height, width, RADIUS=128):
    equ_h = 1500
    equ_w = 3000
    equ_cx = (equ_w - 1) / 2.0
    equ_cy = (equ_h - 1) / 2.0

    wFOV = FOV
    hFOV = float(height) / width * wFOV

    c_x = (width - 1) / 2.0
    c_y = (height - 1) / 2.0

    wangle = (180 - wFOV) / 2.0
    w_len = 2 * RADIUS * np.sin(np.radians(wFOV / 2.0)) / np.sin(np.radians(wangle))
    w_interval = w_len / (width - 1)

    hangle = (180 - hFOV) / 2.0
    h_len = 2 * RADIUS * np.sin(np.radians(hFOV / 2.0)) / np.sin(np.radians(hangle))
    h_interval = h_len / (height - 1)
    x_map = np.zeros([height, width], np.float32) + RADIUS
    y_map = np.tile((np.arange(0, width) - c_x) * w_interval, [height, 1])
    z_map = -np.tile((np.arange(0, height) - c_y) * h_interval, [width, 1]).T
    D = np.sqrt(x_map ** 2 + y_map ** 2 + z_map ** 2)
    xyz = np.zeros([height, width, 3], np.float)
    xyz[:, :, 0] = (RADIUS / D * x_map)[:, :]
    xyz[:, :, 1] = (RADIUS / D * y_map)[:, :]
    xyz[:, :, 2] = (RADIUS / D * z_map)[:, :]

    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    z_axis = np.array([0.0, 0.0, 1.0], np.float32)
    [R1, _] = cv2.Rodrigues(z_axis * np.radians(THETA))
    [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-PHI))

    xyz = xyz.reshape([height * width, 3]).T
    xyz = np.dot(R1, xyz)
    xyz = np.dot(R2, xyz).T
    lat = np.arcsin(xyz[:, 2] / RADIUS)
    lon = np.zeros([height * width], np.float)
    theta = np.arctan(xyz[:, 1] / xyz[:, 0])
    idx1 = xyz[:, 0] > 0
    idx2 = xyz[:, 1] > 0

    idx3 = ((1 - idx1) * idx2).astype(np.bool)
    idx4 = ((1 - idx1) * (1 - idx2)).astype(np.bool)

    lon[idx1] = theta[idx1]
    lon[idx3] = theta[idx3] + np.pi
    lon[idx4] = theta[idx4] - np.pi

    lon = lon.reshape([height, width]) / np.pi * 180
    lat = -lat.reshape([height, width]) / np.pi * 180
    lon = lon / 180 * equ_cx + equ_cx
    lat = lat / 90 * equ_cy + equ_cy

    # closest_x_index = find_closest_index(lon, x)
    # closest_y_index = find_closest_index(lat, y)
    return lon, lat
    # return closest_x_index, closest_y_index, lon[closest_x_index[0], closest_x_index[1]], lat[closest_y_index[0], closest_y_index[1]]


for heading_angle in nn_heading_angles:
    lon, lat = get_perspective(FOV=60,
                                THETA=heading_angle,
                                PHI=0,
                                height=800,
                                width=460)

    parameter_matrices[heading_angle] = (lon, lat)


pano_dir = '/data1/saaket/jpegs_manhattan_touchdown_2021'
data_files = ['/data1/saaket/touchdown/data/train.json', '/data1/saaket/touchdown/data/dev.json', '/data1/saaket/touchdown/data/test.json']

for data_file in data_files:
    targets = []
    data_type = data_file.split("/")[-1].split(".")[0]
    with open(data_file) as f:
        for i, line in enumerate(tqdm(f)):
            obj = json.loads(line)
            pano_id = obj['main_pano']
            center_type = 'main' + '_static_center'
            center = json.loads(obj[center_type])
            # Load the image using opencv 
            image = cv2.imread(pano_dir + '/' + pano_id + '.jpg', cv2.IMREAD_COLOR)
            x, y = int(center['x']*3000), int(center['y']*1500)
            best_x_diff,  best_y_diff = float('inf'), float('inf')
            best_x_index, best_y_index = -1, -1
            target_image_number = 0
            for idx, heading_angle in enumerate(nn_heading_angles):
                current_x_index = find_closest_index(parameter_matrices[heading_angle][0], x)
                current_y_index = find_closest_index(parameter_matrices[heading_angle][1], y)
                current_x_val = parameter_matrices[heading_angle][0][current_x_index[0], current_x_index[1]]
                current_y_val = parameter_matrices[heading_angle][1][current_y_index[0], current_y_index[1]]

                if np.absolute(current_x_val - x) <= best_x_diff and np.absolute(current_y_val - y) <= best_y_diff:
                    best_x_diff = np.absolute(current_x_val - x)
                    best_y_diff = np.absolute(current_y_val - y)

                    best_x_index = current_x_index
                    best_y_index = current_y_index
                    
                    target_image_number = idx 

            targets.append({pano_id: (target_image_number, best_x_index[-1], best_y_index[0])})
    np.save(f'/data1/saaket/lsd_data/data/processed/sdr_{data_type}_perspective_targets_x_y.npy', np.array(targets))


