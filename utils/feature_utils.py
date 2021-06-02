import torch
from typing import Tuple


DEFAULT_BOX_LIMIT = 300


def ground_truth_boxes_to_pixel_map(nc: int, batch_size: int, boxes: torch.Tensor, img_shape: Tuple[int, int]) -> torch.Tensor:
    """
        Converts a tensor of bounding box information for a minibatch of examples into a pixel map for each class.
        If the bounding box extends anywhere into the space of a pixel, the entire pixel is marked as a '1'.

        :param nc: the number of classes this model discriminates between
        :param batch_size: the number of images these boxes correspond to
        :param boxes: A Tensor with a row for each bounding box and a column with 6 values, in the following order:
            image_idx (which image in the minibatch the box is for), class, x, y, w, h.
            X and Y are 0 at the top left of the image.
        :param img_shape: the target output shape, a tuple of 2 integers - image width and image height.
        :return: A Tensor of pixel values between 0 and 1. Its shape will be [batch_size, nc, img_shape[0], img_shape[1]]
        """

    max_width_px = img_shape[0]
    max_height_px = img_shape[1]

    pixel_bounds = torch.zeros((boxes.shape[0], 6)).to(boxes.device)

    # transfer over image, class information
    pixel_bounds[:, 0:2] = boxes[:, 0:2]

    # horiz left bound
    pixel_bounds[:, 2] = torch.floor(max_width_px * (boxes[:, 2] - boxes[:, 4] / 2))

    # horiz right bound
    pixel_bounds[:, 3] = torch.floor(max_width_px * (boxes[:, 2] + boxes[:, 4] / 2))

    # vert top bound
    pixel_bounds[:, 4] = torch.floor(max_height_px * (boxes[:, 3] - boxes[:, 5] / 2))

    # vert bottom bound
    pixel_bounds[:, 5] = torch.floor(max_height_px * (boxes[:, 3] + boxes[:, 5] / 2))

    torch.clamp(pixel_bounds[:, 2:4], 0, max_width_px - 1)
    torch.clamp(pixel_bounds[:, 4:], 0, max_height_px - 1)
    pixel_bounds = pixel_bounds.type(torch.LongTensor)

    output = torch.zeros((batch_size, nc, max_width_px, max_height_px)).to(boxes.device)

    # TODO: consider a vectorized alternative
    for i in range(pixel_bounds.shape[0]):
        img_idx = pixel_bounds[i, 0]
        class_idx = pixel_bounds[i, 1]
        horiz_left_bound = pixel_bounds[i, 2]

        """
        This bound is *inclusive* but torch indexing treats it as exclusive, so we add 1.
        We do the same for 'vert_bottom_bound'.
        """
        horiz_right_bound = pixel_bounds[i, 3] + 1
        vert_top_bound = pixel_bounds[i, 4]
        vert_bottom_bound = pixel_bounds[i, 5] + 1

        output[img_idx, class_idx, horiz_left_bound:horiz_right_bound, vert_top_bound:vert_bottom_bound] = 1

    return output


def predicted_bboxes_to_pixel_map(boxes: torch.Tensor, img_shape: Tuple[int, int],
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
    device = boxes.device

    batch_size, num_boxes, nc = boxes.shape
    nc -= 5  # there are five more entries than the number of classes in the last dim of 'predictions'

    max_width_px = img_shape[0]
    max_height_px = img_shape[1]

    """
    pruning step: sort boxes by objness, then prune all but the 'keep_top_n_boxes' most 'object-y' boxes
    for each image in the batch
    """
    all_boxes = topn_by_objectness(boxes, keep_top_n_boxes, nc).to(device)

    output = torch.zeros(batch_size, nc, max_width_px, max_height_px).to(device)

    for image_idx in range(batch_size):
        boxes = all_boxes[image_idx, :, :].unsqueeze(0).to(device)

        pixel_bounds = torch.zeros((boxes.shape[0], boxes.shape[1], 4)).to(device)  # hleft, hright, vtop, vbottom

        objectness = boxes[:, :, 4]  # objectness
        objectness = objectness.to(device)

        # the following pixel boundaries are inclusive
        # horiz left bound
        pixel_bounds[:, :, 0] = torch.floor(max_width_px*(boxes[:, :, 0] - boxes[:, :, 2]/2))

        # horiz right bound
        pixel_bounds[:, :, 1] = torch.floor(max_width_px*(boxes[:, :, 0] + boxes[:, :, 2]/2))

        # vert top bound
        pixel_bounds[:, :, 2] = torch.floor(max_height_px*(boxes[:, :, 1] - boxes[:, :, 3]/2))

        # vert bottom bound
        pixel_bounds[:, :, 3] = torch.floor(max_height_px * (boxes[:, :, 1] + boxes[:, :, 3]/2))

        torch.clamp(pixel_bounds[:, :, 0:2], 0, max_width_px - 1)
        torch.clamp(pixel_bounds[:, :, 2:4], 0, max_height_px - 1)
        pixel_bounds = pixel_bounds.to(device).long()

        horiz_left_bound = pixel_bounds[:, :, 0]
        horiz_left_bound = horiz_left_bound.to(device)
        horiz_right_bound = pixel_bounds[:, :, 1] + 1
        horiz_right_bound = horiz_right_bound.to(device)
        vert_top_bound = pixel_bounds[:, :, 2]
        vert_top_bound = vert_top_bound.to(device)
        vert_bottom_bound = pixel_bounds[:, :, 3] + 1
        vert_bottom_bound = vert_bottom_bound.to(device)

        # confidence score in a certain class is the product of objectness and class score for that class
        confidence = objectness.unsqueeze(-1) * boxes[:, :, 5:]
        confidence = confidence.to(device)

        # compute a 'horizontal masks' tensor of shape (batch_size, num_kept_boxes, max_width_px)
        horiz_arange = torch.arange(max_width_px).view(1, 1, -1)  # necessary for vectorized mask computation
        horiz_arange = horiz_arange.to(device)
        horiz_left_bound = horiz_left_bound.unsqueeze(-1).expand(horiz_left_bound.shape[0], horiz_left_bound.shape[1], max_width_px)
        horiz_right_bound = horiz_right_bound.unsqueeze(-1).expand(horiz_right_bound.shape[0], horiz_right_bound.shape[1], max_width_px)

        ones = torch.ones((1, 1, 1)).to(device).expand(horiz_left_bound.shape)
        zeros = torch.zeros((1, 1, 1)).to(device).expand(horiz_left_bound.shape)

        horiz_masks = torch.where(horiz_left_bound <= horiz_arange, ones, zeros).to(device) * torch.where(horiz_arange <= horiz_right_bound, ones, zeros).to(device)
        horiz_masks = horiz_masks.float().to(device)

        # compute a 'vertical masks' tensor of shape (batch_size, num_kept_boxes, max_height_px)
        vert_arange = torch.arange(max_height_px).view(1, 1, -1)  # necessary for vectorized mask computation
        vert_arange = vert_arange.float().to(device)
        vert_top_bound = vert_top_bound.unsqueeze(-1).expand(vert_top_bound.shape[0], vert_top_bound.shape[1], max_height_px)
        vert_bottom_bound = vert_bottom_bound.unsqueeze(-1).expand(vert_bottom_bound.shape[0], vert_bottom_bound.shape[1], max_height_px)

        ones = torch.ones((1, 1, 1)).to(device).expand(vert_top_bound.shape)
        zeros = torch.zeros((1, 1, 1)).to(device).expand(vert_top_bound.shape)

        vert_masks = torch.where(vert_top_bound <= vert_arange, ones, zeros).to(device) * torch.where(vert_arange <= vert_bottom_bound, ones, zeros).to(device)
        vert_masks = vert_masks.to(device)

        """
        reshape masks to get ready for batched matrix multiply (this will result in the tensor we want
        of shape (batch_size, num_kept_boxes, max_width_px, max_height_px)
        """
        vert_masks = vert_masks.unsqueeze(-1).permute(0, 1, 3, 2)
        horiz_masks = horiz_masks.unsqueeze(-1)

        # use batched matrix multiplication to compute masks
        confidence_masks = torch.matmul(horiz_masks, vert_masks).unsqueeze(2).expand(1, boxes.shape[1], nc, max_width_px, max_height_px)
        confidence_masks = confidence_masks.to(device)

        # use 'expand' and 'unsqueeze' to avoid nasty broadcasting
        confidence = confidence.unsqueeze(-1).unsqueeze(-1).expand(confidence_masks.shape)

        # compute per-box pixel values
        per_box_output = confidence_masks * confidence
        per_box_output = per_box_output.to(device)

        # compute max among pixel values for each image in the batch, class, x position, and y position
        output[image_idx, :, :, :], _ = torch.max(per_box_output, dim=1)

    return output


def sort_by_objectness(boxes: torch.Tensor):
    _, sorted_indices = torch.sort(boxes[:, :, 4].view(boxes.shape[0], boxes.shape[1]), dim=1, descending=True)
    boxes_t = boxes.permute(0, 2, 1)
    indices_t = sorted_indices.unsqueeze(-1).repeat(1, 1, 7).permute(0, 2, 1)
    sorted_boxes_t = torch.gather(boxes_t, 2, indices_t)
    return sorted_boxes_t.permute(0, 2, 1)


def topn_by_objectness(boxes: torch.Tensor, k: int, nc: int):
    k = min(k, boxes.shape[1])  # not likely to happen in practice, but nice for testing and harmless
    _, sorted_indices = torch.topk(boxes[:, :, 4].view(boxes.shape[0], boxes.shape[1]), k, dim=1, sorted=False, largest=True)
    boxes_t = boxes.permute(0, 2, 1)
    indices_t = sorted_indices.unsqueeze(-1).repeat(1, 1, 5 + nc).permute(0, 2, 1)
    sorted_boxes_t = torch.gather(boxes_t, 2, indices_t)
    return sorted_boxes_t.permute(0, 2, 1)


def test_pixel_map():
    boxes = torch.rand(size=(2, 48, 7))*(2/3)
    boxes[0, 0, :] = torch.Tensor([0.49, 0.49, 0.5, 0.5, 1, 1, 0])  # class 0 has a square in the center of its map
    boxes[0, 1, :] = torch.Tensor([0.49, 0.49, 0.5, 0.5, 1, 0, 1])  # class 1 has a square in the center of its map
    boxes[1, 2, :] = torch.Tensor([0.24, 0.24, 0.5, 0.5, 1, 0.7, 0])  # class 0 has a square in the upper-left of its map
    boxes[1, 3, :] = torch.Tensor([0.74, 0.74, 0.5, 0.5, 1, 0, 0.8])  # class 1 has a square in the bottom-right of its map
    outputs = predicted_bboxes_to_pixel_map(boxes, (8, 8))
    return outputs


if __name__ == "__main__":
    # some test code to demonstrate the effect (you can visualize it with a debugger)
    test_pixel_map()
