import torch

import torch.nn as nn
import numpy as np

from Backbone import get_face_alignment_net

from .Transformer import Transformer
from .interpolation import interpolation_layer
from .get_roi import get_roi


class Sparse_alignment_network(nn.Module):
    def __init__(self, num_point, d_model, trainable,
                 return_interm_layers, dilation, nhead,  feedforward_dim,
                 initial_path, cfg):
        super(Sparse_alignment_network, self).__init__()
        # 读取初始参数
        self.num_point = num_point
        self.d_model = d_model
        self.trainable = trainable
        self.return_interm_layers = return_interm_layers
        self.dilation = dilation
        self.nhead = nhead
        self.feedforward_dim = feedforward_dim
        self.initial_path = initial_path
        self.Sample_num = cfg.MODEL.SAMPLE_NUM

        self.initial_points = torch.from_numpy(np.load(initial_path)['init_face'] / 256.0).view(1, num_point, 2).float()
        self.initial_points.requires_grad = False

        # ROI_creator
        self.ROI_1 = get_roi(self.Sample_num, 8.0, 64)
        self.ROI_2 = get_roi(self.Sample_num, 4.0, 64)
        self.ROI_3 = get_roi(self.Sample_num, 2.0, 64)

        self.interpolation = interpolation_layer()

        # feature_extractor
        self.feature_extractor = nn.Conv2d(d_model, d_model, kernel_size=self.Sample_num, bias=False)

        self.feature_norm = nn.LayerNorm(d_model)

        # Transformer
        self.Transformer = Transformer(num_point, d_model, nhead, cfg.TRANSFORMER.NUM_DECODER,
                                       feedforward_dim, dropout=0.1)

        self.out_layer = nn.Linear(d_model, 2)

        self._reset_parameters()

        # backbone
        self.backbone = get_face_alignment_net(cfg)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, image):
        # 获取Batch_size
        bs = image.size(0)

        output_list = []

        feature_map = self.backbone(image)

        initial_landmarks = self.initial_points.repeat(bs, 1, 1).to(image.device)

        # stage_1
        ROI_anchor_1, bbox_size_1, start_anchor_1 = self.ROI_1(initial_landmarks.detach())
        ROI_anchor_1 = ROI_anchor_1.view(bs, self.num_point * self.Sample_num * self.Sample_num, 2)
        ROI_feature_1 = self.interpolation(feature_map, ROI_anchor_1.detach()).view(bs, self.num_point, self.Sample_num,
                                                                            self.Sample_num, self.d_model)
        ROI_feature_1 = ROI_feature_1.view(bs * self.num_point, self.Sample_num, self.Sample_num,
                                     self.d_model).permute(0, 3, 2, 1)

        transformer_feature_1 = self.feature_extractor(ROI_feature_1).view(bs, self.num_point, self.d_model)

        offset_1 = self.Transformer(transformer_feature_1)
        offset_1 = self.out_layer(offset_1)

        landmarks_1 = start_anchor_1.unsqueeze(1) + bbox_size_1.unsqueeze(1) * offset_1
        output_list.append(landmarks_1)

        # stage_2
        ROI_anchor_2, bbox_size_2, start_anchor_2 = self.ROI_2(landmarks_1[:, -1, :, :].detach())
        ROI_anchor_2 = ROI_anchor_2.view(bs, self.num_point * self.Sample_num * self.Sample_num, 2)
        ROI_feature_2 = self.interpolation(feature_map, ROI_anchor_2.detach()).view(bs, self.num_point, self.Sample_num,
                                                                                 self.Sample_num, self.d_model)
        ROI_feature_2 = ROI_feature_2.view(bs * self.num_point, self.Sample_num, self.Sample_num,
                                           self.d_model).permute(0, 3, 2, 1)

        transformer_feature_2 = self.feature_extractor(ROI_feature_2).view(bs, self.num_point, self.d_model)

        offset_2 = self.Transformer(transformer_feature_2)
        offset_2 = self.out_layer(offset_2)

        landmarks_2 = start_anchor_2.unsqueeze(1) + bbox_size_2.unsqueeze(1) * offset_2
        output_list.append(landmarks_2)

        # stage_3
        ROI_anchor_3, bbox_size_3, start_anchor_3 = self.ROI_3(landmarks_2[:, -1, :, :].detach())
        ROI_anchor_3 = ROI_anchor_3.view(bs, self.num_point * self.Sample_num * self.Sample_num, 2)
        ROI_feature_3= self.interpolation(feature_map, ROI_anchor_3.detach()).view(bs, self.num_point, self.Sample_num,
                                                                                   self.Sample_num, self.d_model)
        ROI_feature_3 = ROI_feature_3.view(bs * self.num_point, self.Sample_num, self.Sample_num,
                                           self.d_model).permute(0, 3, 2, 1)

        transformer_feature_3 = self.feature_extractor(ROI_feature_3).view(bs, self.num_point, self.d_model)

        offset_3 = self.Transformer(transformer_feature_3)
        offset_3 = self.out_layer(offset_3)

        landmarks_3 = start_anchor_3.unsqueeze(1) + bbox_size_3.unsqueeze(1) * offset_3
        output_list.append(landmarks_3)

        return output_list