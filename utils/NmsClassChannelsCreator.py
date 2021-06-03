import logging
import torch
from utils.ClassChannelsCreator import ClassChannelsCreator
from utils.ClassChannelsCreator import DEFAULT_BOX_LIMIT
from utils.general import non_max_suppression
from utils.nms_feature_utils import nms_predicted_bboxes_to_pixel_map
from typing import Tuple

class NmsClassChannelsCreator(ClassChannelsCreator):
    def  __init__(self, conf = 0.25, iou = 0.45, classes = None) -> None:
        super().__init__()
        # Non-Maximum Suppression (NMS) module
        self.conf = 0.25  # confidence threshold
        self.iou = 0.45  # IoU threshold
        self.classes = None  # (optional list) filter by class

    def predicted_bboxes_to_pixel_map(self, boxes: torch.Tensor, img_shape: Tuple[int, int],
                                      keep_top_n_boxes: int = DEFAULT_BOX_LIMIT) -> torch.Tensor:
        """
        Converts a tensor of bounding box information for a minibatch of examples into a pixel map for each class.
        Each class' pixel map has the highest confidence value of any box containing that pixel. The confidence value is
        the product of the objectness score and the class confidence. The 'boxes' tensor may be obtained from the output
        of a yolo.Model's 'forward()' method in inference mode.

        :param boxes: a Tensor of box predictions organized like [batch_size, num_boxes, features]
            'features' has the following values, in this order: x, y, w, h, objectness, class_scores
            where class_scores is num_classes long, from 0 to the max class index.
        :param img_shape: the target output shape, a tuple of 2 integers - image width and image height.
        :param keep_top_n_boxes: An integer that controls the number of boxes used for the pixel map.
            Higher values are more accurate but slower.
        :return: A Tensor of pixel values between 0 and 1. Its shape will be [batch_size, nc, img_shape[0], img_shape[1]]
        """
        logger = logging.getLogger(__name__)
        _, _, num_features = boxes.shape 
        num_channels = num_features - 5
        boxesCollection = non_max_suppression(boxes, conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)
        output = nms_predicted_bboxes_to_pixel_map(boxesCollection, img_shape, num_channels, keep_top_n_boxes)
        return output

