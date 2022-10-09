from genericpath import exists
import os 

jpegs = os.listdir('/data1/saaket/jpegs_manhattan_touchdown_2021')

errors = []

for file in jpegs:
    jpeg_name = file.replace(".jpg", "")
    if os.path.isdir(f'/data1/saaket/lsd_data/data/processed/pano_slices/{jpeg_name}'):
        for i in range(8):
            if not os.path.exists(f'/data1/saaket/lsd_data/data/processed/pano_slices/{jpeg_name}/{jpeg_name}_{i}.jpg'):
                # errors.append(f'/data1/saaket/lsd_data/data/processed/pano_slices/{jpeg_name}/{jpeg_name}_{i}.jpg')

                if jpeg_name not in errors:
                    errors.append(jpeg_name)
    else:
        if jpeg_name not in errors:
            errors.append(jpeg_name)

with open('dump/perspective_panorama_errors.txt', 'w') as f:
    content = str(errors)
    f.write(content) 
