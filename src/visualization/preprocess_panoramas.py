import numpy as np
import cv2 
import json 
import py360convert
import sys 

pano_dir = '/data1/saaket/jpegs_manhattan_touchdown_2021'
data_file = '/data1/saaket/touchdown/data/train.json'

with open(data_file) as f:
    for i, line in enumerate(f):
        obj = json.loads(line)
        if i == 1:
            break



pano_id = obj['main_pano']
center_type = 'main' + '_static_center'
center = json.loads(obj[center_type])

# Load the image using opencv 
image = cv2.imread(pano_dir + '/' + pano_id + '.jpg', cv2.IMREAD_COLOR)
x, y = int(center['x']*3000), int(center['y']*1500)
cv2.circle(image, (x, y), 20, (0, 0, 255), -1) # -1 fills the whole circle 

# In case coordinates are to be inferred from the the initial pano 
# image_num = x // ( 375 )
# x2 = x - (image_num*375)

# x2 = int(x2*(460/3000))
# y2 = int(y*(800/1500))

def find_closest_index(arr, num):
    diff_arr = np.absolute(arr - num)
    return np.unravel_index(diff_arr.argmin(), diff_arr.shape)

# Persepctive transforming the image 
def get_perspective(img, target, FOV, THETA, PHI, height, width, x, y, RADIUS=128):
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

    closest_x_index = find_closest_index(lon, x)
    closest_y_index = find_closest_index(lat, y)


    persp = cv2.remap(img, lon.astype(np.float32), lat.astype(np.float32), cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_WRAP)
    # print(closest_x_index)
    return persp, closest_x_index, closest_y_index, lon[closest_x_index[0], closest_x_index[1]], lat[closest_y_index[0], closest_y_index[1]]

h_stride = 45
nn_heading_angles = list(range(-180, 180, h_stride))

slices = list()
targets = list()
target = np.zeros(image.shape[:2], dtype='uint8')
target[y-100:y+100, x-100:x+100] = 255


for heading_angle in nn_heading_angles:
    img_slice, x_index, y_index, x_val, y_val = get_perspective(image, target, FOV=60,
                            THETA=heading_angle,
                            PHI=0,
                            height=800,
                            width=460,
                            x=x,
                            y=y)
    slices.append(img_slice)
    
    print(x_index, y_index, x_val, y_val, x, y)

big_slice = np.stack(slices, axis=0)
# 8 x 800 x 460 x 3
big_slice = np.transpose(big_slice, (1, 0, 2, 3))
big_slice = np.reshape(big_slice, (800, 3680, 3))
cv2.imwrite('big_slice_demo.jpg', big_slice)
print(big_slice.shape)
cv2.circle(slices[-1], (124, 444), 10, (0, 255, 0), -1)

# cv2.circle(image, (x, y), 20, (0, 0, 255), -1)

for i in range(len(slices)):
    # cv2.circle(slices[i], (x2, y2), 100, (0, 255, 0), -1)
    cv2.imwrite(f'perspective_{i}.jpg', slices[i])

# Visualize the sdr location in the panorama 
sdr_text = obj['td_location_text']
print(sdr_text)