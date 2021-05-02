from torch import nn
import torch
import torchvision
import math

import models.common as common
import models.yolo as yolo
from utils.autoanchor import check_anchor_order
from utils.torch_utils import scale_img


class ToyModel(nn.Module):
    """
    3 input channels are hard-coded for the end-to-end model
    """

    def __init__(self, anchors, nc):
        super(ToyModel, self).__init__()
        self.resizer = torchvision.transforms.Resize((255, 255))
        convlist = [
            common.Conv(c1=3, c2=16, k=3, s=2, p=0),  #output: 127x127x16
            common.Conv(c1=16, c2=32, k=3, s=1, p=0),  #output: 125x125x32
            common.Conv(c1=32, c2=64, k=7, s=3, p=0)  #output: 40x40x64
            #common.SPP(c1=64, c2=64),  # output: (36 + 32 + 28)x64
        ]
        self.module_list = nn.ModuleList()
        self.module_list.extend(convlist)

        self.yolo_head = yolo.Detect(anchors=anchors, ch=(64,))

        # this is for compatibility reasons
        self.model = [self.yolo_head]
        self.inplace = True

        print(self.yolo_head.anchors)

        # ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        # if nc and nc != self.yaml['nc']:
        #     self.yaml['nc'] = nc  # override yaml value
        # if anchors:
        #     self.yaml['anchors'] = round(anchors)  # override yaml value
        # self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        # self.inplace = self.yaml.get('inplace', True)

        # this is also for compatibility reasons
        if isinstance(self.yolo_head, yolo.Detect):
            ch = 3  # R, G, B
            s = 256  # 2x min stride
            self.yolo_head.inplace = self.inplace
            self.yolo_head.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            self.yolo_head.anchors /= self.yolo_head.stride.view(-1, 1, 1)
            check_anchor_order(self.yolo_head)
            self.stride = self.yolo_head.stride
            self.inplace = self.yolo_head.inplace
            self._initialize_biases()  # only run once


    def forward(self, x, augment=False, profile=False):
        # ignore profile keyword for now
        if augment:
            return self.forward_augment(x)
        else:
            intermediate = self.resizer(x)
            for conv in self.module_list:
                intermediate = conv(intermediate)
            return self.yolo_head([intermediate])

    # needed for compatibility reasons
    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self.forward(xi, augment=False)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        return torch.cat(y, 1), None  # augmented inference, train

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

def init_toy_model(opt, hyp, nc) -> nn.Module:
    """
    This is simply a test to see if I can get an arbitrary model to work with the training code
    """
    return ToyModel(hyp.get('anchors'), nc).to()
