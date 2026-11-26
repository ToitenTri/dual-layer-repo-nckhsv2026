import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Resize(nn.Module):
    """
    Resize the image. The target size is original size * resize_ratio
    """
    def __init__(self, resize_ratio, interpolation_method = 'nearest'):
        super(Resize, self).__init__()
        if resize_ratio <= 0:
            raise ValueError("resize_ratio must be greater than zero")
        self.resize_ratio = resize_ratio
        self.interpolation_method = interpolation_method


    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]

        new_size = (
            max(1, int(np.rint(noised_image.shape[2] * self.resize_ratio))),
            max(1, int(np.rint(noised_image.shape[3] * self.resize_ratio))),
        )
        align_corners = False if self.interpolation_method in ("linear", "bilinear", "bicubic", "trilinear") else None
        kwargs = {"align_corners": align_corners} if align_corners is not None else {}
        noised_and_cover[0] = F.interpolate(noised_image, size=new_size, mode=self.interpolation_method, **kwargs)
        noised_and_cover[1] = F.interpolate(noised_and_cover[1], size=new_size, mode=self.interpolation_method, **kwargs)

        return noised_and_cover
