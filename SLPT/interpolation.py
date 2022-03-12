import torch

import torch.nn as nn


class interpolation_layer(nn.Module):
    def __init__(self):
        super(interpolation_layer, self).__init__()

    def forward(self, feature_maps, init_potential_anchor):
        """
        :param feature_map: (Bs, 256, Height, Width)
        :param potential_anchor: (BS, number_point, 2)
        :return:
        """

        feature_dim = feature_maps.size()
        
        potential_anchor = init_potential_anchor * (feature_dim[2] - 1)

        potential_anchor = torch.clamp(potential_anchor, 0, feature_dim[2] - 1)

        anchor_pixel = self._get_interploate(potential_anchor, feature_maps, feature_dim)
        return anchor_pixel


    def _flatten_tensor(self, input):
        return input.contiguous().view(input.nelement())


    def _get_index_point(self, input, anchor, feature_dim):
        index = anchor[:, :, 1] * feature_dim[2] + anchor[:, :, 0]

        output_list = []
        for i in range(feature_dim[0]):
            output_list.append(torch.index_select(input[i].contiguous().flatten(1), 1, index[i]))
        output = torch.stack(output_list)

        return output.permute(0, 2, 1).contiguous()


    def _get_interploate(self, potential_anchor, feature_maps, feature_dim):
        anchors_lt = potential_anchor.floor().long()
        anchors_rb = potential_anchor.ceil().long()

        anchors_lb = torch.stack([anchors_lt[:, :, 0], anchors_rb[:, :, 1]], 2)
        anchors_rt = torch.stack([anchors_rb[:, :, 0], anchors_lt[:, :, 1]], 2)

        vals_lt = self._get_index_point(feature_maps, anchors_lt.detach(), feature_dim)
        vals_rb = self._get_index_point(feature_maps, anchors_rb.detach(), feature_dim)
        vals_lb = self._get_index_point(feature_maps, anchors_lb.detach(), feature_dim)
        vals_rt = self._get_index_point(feature_maps, anchors_rt.detach(), feature_dim)

        coords_offset_lt = potential_anchor - anchors_lt.type(potential_anchor.data.type())

        vals_t = vals_lt + (vals_rt - vals_lt) * coords_offset_lt[:, :, 0:1]
        vals_b = vals_lb + (vals_rb - vals_lb) * coords_offset_lt[:, :, 0:1]
        mapped_vals = vals_t + (vals_b - vals_t) * coords_offset_lt[:, :, 1:2]

        return mapped_vals


if __name__ == '__main__':
    model = deformable_layer(98)
    input_tensor = torch.rand(2, 512, 16, 16)
    anchor = torch.rand(2, 98, 2)
    output_tensor = model(input_tensor, anchor)




