import torch
from typing import Tuple
from typing import List
import utils.feature_utils

DEFAULT_BOX_LIMIT = 300
DEFAULT_NUM_CHANNELS = 3

def nms_predicted_bboxes_to_pixel_map(boxes: List[torch.Tensor], img_shape: Tuple[int, int], num_channels = DEFAULT_NUM_CHANNELS,
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
    :param num_channels: Number of channels in the image
    :param keep_top_n_boxes: An integer that controls the number of boxes used for the pixel map.
        Higher values are more accurate but slower.
    :return: A Tensor of pixel values between 0 and 1. Its shape will be [batch_size, nc, img_shape[0], img_shape[1]]
    """

    numFeatures = num_channels + 5

    max_width_px = img_shape[0]
    max_height_px = img_shape[1]

    device = boxes[0].device # We will have at least one image in the batch.

    numImages = len(boxes)

    output = torch.zeros((numImages, num_channels, max_width_px, max_height_px)).to(device)

    for imageIndex in range(numImages):
        imageBoxes = boxes[imageIndex] 
        numBoxes = imageBoxes.shape[0]
        # Create a tensor of shape (1, something)
        input =  torch.zeros((1, numBoxes, numFeatures)).to(device)

        # We don't have the objectness score after NMS. We will use 1.
        input[0, :, 4] = 1 # 'features' has the following values, in this order: x [0], y [1], w [2], h [3], objectness [4], class_scores [5, 6, 7]

        # Index will need to be added by 5 as input's class scores start from column 5
        indexClass = imageBoxes[:, 5] # imageBoxes shape is x [0] y [1] x [2] y [3] conf [4] class [5]
        indexClass = indexClass.long().to(device)
        indexClass = 5 + indexClass # Because class scores for input start at 5 and our class numbers are zero indexed
        input[0, :, indexClass] =  imageBoxes[:, 4] # imageBoxes shape is x [0] y [1] x [2] y [3] conf [4] class [5]

        # Populate center X
        input[0, :, 0] = (imageBoxes[:, 0] + imageBoxes[:, 2]) / (2*max_width_px) # imageBoxes shape is x [0] y [1] x [2] y [3] conf [4] class [5]
        # Populate center Y
        input[0, :, 1] = (imageBoxes[:, 1] + imageBoxes[:, 3]) / (2*max_height_px) # imageBoxes shape is x [0] y [1] x [2] y [3] conf [4] class [5]
        # Populate width of bounding box 
        input[0, :, 2] = (1 + imageBoxes[:, 2] - imageBoxes[:, 0]) / (max_width_px)  # imageBoxes shape is x [0] y [1] x [2] y [3] conf [4] class [5]
        # Populate height of bounding box 
        input[0, :, 3] = (1 + imageBoxes[:, 3] - imageBoxes[:, 1]) / (max_height_px) # imageBoxes shape is x [0] y [1] x [2] y [3] conf [4] class [5]

        # Get the result
        imageOutput = utils.feature_utils.predicted_bboxes_to_pixel_map(input, img_shape, keep_top_n_boxes)
        output[imageIndex] = imageOutput[0]

    return output


def test_pixel_map():
    boxes = []
    imageBoxes1 = torch.zeros(size=(2, 6))
    imageBoxes1[0] = torch.Tensor([2, 3, 11, 7, 1, 0])
    imageBoxes1[0] = torch.Tensor([0, 0, 8, 6, 0.7, 0])
    boxes.append(imageBoxes1)
    outputs = nms_predicted_bboxes_to_pixel_map(boxes, (12, 8))
    return outputs


if __name__ == "__main__":
    # some test code to demonstrate the effect (you can visualize it with a debugger)
    test_pixel_map()
