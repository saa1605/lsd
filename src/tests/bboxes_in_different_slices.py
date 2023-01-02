import torch 
from tqdm import tqdm

def count_boxes_in_different_slices(bounding_boxes):
    count = 0
    total_boxes = 0 
    for boxes_in_pano in tqdm(bounding_boxes):
        for box in boxes_in_pano:
            if box[0] != box[3]:
                count += 1
            total_boxes += 1

    return count, ((count / total_boxes)*100)

def main():
    bounding_boxes = torch.load('/data2/saaket/mapped_best_train_bboxes_ViT_full_pano.pth') 
    count, percent = count_boxes_in_different_slices(bounding_boxes)

    print(count, percent)

if __name__=='__main__':
    main()