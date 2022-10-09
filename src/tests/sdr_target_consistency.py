import numpy as np 
import json 
import cv2 

pano_dir = '/data1/saaket/jpegs_manhattan_touchdown_2021'
data_file = '/data1/saaket/touchdown/data/train.json'
target_file = '/data1/saaket/lsd_data/data/processed/sdr_train_perspective_targets_x_y.npy'

targets = np.load(target_file, allow_pickle=True)

# Persepctive transforming the image 
def get_perspective(img, FOV, THETA, PHI, height, width, RADIUS=128):
    img = np.array(img)
    img_height, img_width, _ = img.shape
    equ_h = img_height
    equ_w = img_width
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


    persp = cv2.remap(img, lon.astype(np.float32), lat.astype(np.float32), cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_WRAP)
    return persp


with open(data_file) as f:
    for i, line in enumerate(f):
        obj = json.loads(line)
        pano_id = obj['main_pano']
        center_type = 'main' + '_static_center'
        center = json.loads(obj[center_type])

        # Load the image using opencv 
        image = cv2.imread(pano_dir + '/' + pano_id + '.jpg', cv2.IMREAD_COLOR)
        x, y = int(center['x']*3000), int(center['y']*1500)
        cv2.circle(image, (x, y), 20, (0, 0, 255), -1) # -1 fills the whole circle 
        

        image_num, mapped_x, mapped_y = targets[i][pano_id]

        nn_heading_angles = list(range(-180, 180, 45))

        perspective_image = get_perspective(image, 
                                            FOV=60, 
                                            THETA=nn_heading_angles[image_num],
                                            PHI=0,
                                            height=800,
                                            width=460, 
                                            RADIUS=128)
        print(mapped_x, mapped_y)
        cv2.circle(perspective_image, (mapped_x, mapped_y), 10, (0, 255, 0), -1)
        cv2.imwrite(f'dump/target_check_{i}.jpg', perspective_image)
        if i == 10:
            break 
