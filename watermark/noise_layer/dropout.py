import torch
import torch.nn as nn

class Dropout(nn.Module):
    """
    Drops random pixels from the noised image and substitutes them with the pixels from the cover image
    """
    def __init__(self, keep_ratio_range):
        super(Dropout, self).__init__()
        if len(keep_ratio_range) != 2 or not 0 <= keep_ratio_range[0] <= keep_ratio_range[1] <= 1:
            raise ValueError("keep ratios must satisfy 0 <= min <= max <= 1")
        self.keep_min = keep_ratio_range[0]
        self.keep_max = keep_ratio_range[1]

    def forward(self, noised_and_cover):

        noised_image = noised_and_cover[0]
        cover_image = noised_and_cover[1]

        keep_ratio = torch.empty((), device=noised_image.device).uniform_(self.keep_min, self.keep_max)
        mask = (torch.rand_like(noised_image[:, :1]) < keep_ratio).to(noised_image.dtype)
        noised_and_cover[0] = noised_image * mask + cover_image * (1 - mask)
        return noised_and_cover