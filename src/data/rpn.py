import numpy as np 
import cv2 
import torch 
from torchvision.ops import nms
from itertools import chain, combinations

class RegionProposalNetwork:
    def get_coords_and_masks(self, pil_img, B=0.15, K_max_box_w=0.9, K_max_box_h=0.9,
                                K_min_box_w=0.03, K_min_box_h=0.03, iou_threshold=0.9):
        img = pil_img.copy()
        img = np.array(img)
        img = self._c_mean_shift(img)
        img = self._split_gray_img(img, n_labels=9)
        coords, masks = self._get_mixed_boxes_and_masks(
            img, B, K_max_box_w, K_max_box_h, K_min_box_w, K_min_box_h, iou_threshold
        )
        return coords, masks


    def _get_mixed_boxes_and_masks(self, image, B, K_max_box_w, K_max_box_h, K_min_box_w, K_min_box_h, iou_threshold):
        img = image.copy()
        h, w = img.shape
        max_box_w, max_box_h, min_box_w, min_box_h = w * K_max_box_w, h * K_max_box_h, w * K_min_box_w, h * K_min_box_h

        out_boxes = []
        out_masks = []

        labels = np.unique(img)
        combs = self._get_combinations(labels)

        comb_indexes = []
        for i, comb in enumerate(combs):
            n_img = np.isin(img, np.array(comb)).astype(np.uint8) * 255
            n_img = self._clear_noise(n_img)
            m_boxes = self._get_boxes_from_mask(n_img, max_box_w, max_box_h, min_box_w, min_box_h)
            out_boxes.extend(m_boxes)
            comb_indexes.extend([i] * len(m_boxes))

        comb_indexes = np.array(comb_indexes)

        boxes = torch.tensor(out_boxes, dtype=torch.float32)
        labels = torch.ones(boxes.shape[0], dtype=torch.float32)

        indexes = nms(boxes, labels, iou_threshold)

        out_boxes = boxes[indexes].numpy().astype(np.int32)
        comb_indexes = comb_indexes[indexes.numpy()]

        for (x1, y1, x2, y2), comb_index in zip(out_boxes, comb_indexes):
            comb = combs[comb_index]
            n_img = np.isin(img, np.array(comb)).astype(np.uint8) * 255
            n_img = self._clear_noise(n_img)
            mask = n_img[y1:y2, x1:x2]

            h, w = mask.shape
            h_b = int(h * B)
            w_b = int(w * B)

            mask = mask.astype(np.bool)
            if mask[:h_b, :].sum() + mask[-h_b:, :].sum() + mask[:, :w_b].sum() + \
                    mask[:, -w_b:].sum() > 4 * h * w * B * (1 - B) * 0.5:
                mask = ~mask

            mask = mask.astype(np.uint8) * 255
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)
            mask = cv2.dilate(mask, kernel, iterations=3)
            mask = mask.astype(np.bool)
            out_masks.append(mask)

        return out_boxes, out_masks
    
    def _c_mean_shift(self, image):
        img = image.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img = cv2.pyrMeanShiftFiltering(img, 16, 48)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img.astype(np.uint8)

    def _split_gray_img(self, image, n_labels=9):
        img = image.copy()
        step = 255 // n_labels
        t = list(np.arange(0, 255, step)) + [255]
        for i, (t1, t2) in enumerate(zip(t[:-1], t[1:])):
            img[(img >= t1) & (img < t2)] = t1
        return img

    @staticmethod
    def _get_combinations(array):
        combs = list(chain(*map(lambda x: combinations(array, x), range(0, len(array)+1))))
        return combs[1:]

    @staticmethod
    def _get_boxes_from_mask(mask, max_box_w, max_box_h, min_box_w, min_box_h):
        boxes = []
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        bboxes = [cv2.boundingRect(c) for c in contours]
        for i, (x, y, w, h) in enumerate(bboxes):
            if (w < max_box_w and h < max_box_h) and (w > min_box_w and h > min_box_h):
                boxes.append([x, y, x + w, y + h])
        return boxes

    @staticmethod
    def _clear_noise(image):
        img = image.copy()
        e_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        erose = cv2.morphologyEx(img, cv2.MORPH_ERODE, e_kernel)
        d_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilate = cv2.morphologyEx(erose, cv2.MORPH_DILATE, d_kernel)
        return dilate

    @staticmethod
    def _get_nearest_box_and_mask(box, gt_boxes, gt_masks):
        return sorted(zip(gt_boxes, gt_masks), key=lambda x: sum([abs(x[0][i] - box[i]) for i in range(4)]))[0]