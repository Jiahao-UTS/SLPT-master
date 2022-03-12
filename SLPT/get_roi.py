import torch
import torch.nn as nn

class get_roi(nn.Module):

    def __init__(self, num_points, half_length, img_size):
        super(get_roi, self).__init__()
        self.img_size = img_size
        self.num_points = num_points
        self.half_length = torch.tensor([[[half_length, half_length]]], dtype=torch.float32)
        self.half_length.requires_grad = False

    def forward(self, anchor):
        Bs = anchor.size(0)
        half_length = (self.half_length.to(anchor.device) / (self.img_size)).repeat(Bs, 1, 1)
        bounding_min = torch.clamp(anchor - half_length, 0.0, 1.0)
        bounding_max = torch.clamp(anchor + half_length, 0.0, 1.0)
        bounding_box = torch.cat((bounding_min, bounding_max), dim=2)
        bounding_length = bounding_max - bounding_min

        bounding_xs = torch.nn.functional.interpolate(bounding_box[:,:,0::2], size=self.num_points,
                                                      mode='linear', align_corners=True)
        bounding_ys = torch.nn.functional.interpolate(bounding_box[:,:,1::2], size=self.num_points,
                                                      mode='linear', align_corners=True)
        bounding_xs, bounding_ys = bounding_xs.unsqueeze(3).repeat_interleave(self.num_points, dim=3), \
                                   bounding_ys.unsqueeze(2).repeat_interleave(self.num_points, dim=2)

        meshgrid = torch.stack([bounding_xs, bounding_ys], dim=-1)

        return meshgrid, bounding_length, bounding_min
