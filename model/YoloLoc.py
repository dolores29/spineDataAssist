import os

import cv2
import numpy as np
import torch

from model.utils_bbox import DecodeBox
from utils.util import cvtColor, resize_image, preprocess_input


class LocPred:
    def __init__(self):
        # 'yolov5_2d_spine.pt'
        self.model_path = r'./pth/weight_loc/yolov5_2d_spine.pt'
        self.anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        self.phi = 's'
        self.num_classes = 3
        self.conf = 0.4
        self.anchors = np.array([[10, 13], [16, 30], [33, 23],
                                 [30, 61], [62, 45], [59, 119],
                                 [116, 90], [156, 198], [373, 326]])
        self.bbox_util = DecodeBox(self.anchors, self.num_classes, (640, 640),
                                   self.anchors_mask)

    def load_model(self, image):
        model_load = torch.jit.load(self.model_path)
        model_load.eval()
        image_shape = np.array(np.shape(image)[0:2])
        image = cvtColor(image)
        image_data = resize_image(image, (640, 640), True)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            images = images.cuda()

            outputs = model_load(images)
            outputs = self.bbox_util.decode_box(outputs)
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, [640, 640],
                                                         image_shape, True, conf_thres=self.conf,
                                                         nms_thres=0.5)
        if results[0] is None:
            return []
        top_conf = results[0][:, 4] * results[0][:, 5] > self.conf
        top_boxes = results[0][top_conf, :4]

        # top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_boxes[:, 0], -1), \
        #                                          np.expand_dims(top_boxes[:, 1], -1), \
        #                                          np.expand_dims(top_boxes[:, 2], -1), \
        #                                          np.expand_dims(top_boxes[:, 3], -1)
        # # 输出这里为正常顺序（xmin,ymin,xmax,ymax）
        # boxes = np.concatenate([top_xmin, top_ymin, top_xmax, top_ymax], axis=-1)
        # # 因为需要按照x的大小排序，所以转置
        # boxes = boxes[np.lexsort(boxes.T)]

        # 边界限定
        new_top_boxes = []
        for top_box in top_boxes:
            top, left, bottom, right = top_box
            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))
            new_top_boxes.append([top, left, bottom, right])
        return np.array(new_top_boxes)
