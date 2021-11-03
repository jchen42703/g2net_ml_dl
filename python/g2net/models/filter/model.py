from __future__ import annotations
import torch.nn as nn


class FilterModel(nn.Module):

    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 1,
                 hidden_dims: tuple = (8, 16, 32, 64),
                 kernel_size: int = 3,
                 reinit: bool = False):

        super().__init__()

        # first conv
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=in_channels,
                      out_channels=hidden_dims[0],
                      kernel_size=kernel_size,
                      padding='same'), nn.BatchNorm1d(hidden_dims[0]),
            nn.SiLU(inplace=True))

        for i in range(len(hidden_dims) - 1):
            conv_layer = nn.Sequential(
                nn.Conv1d(in_channels=hidden_dims[i],
                          out_channels=hidden_dims[i + 1],
                          kernel_size=kernel_size,
                          padding='same'), nn.BatchNorm1d(hidden_dims[i + 1]),
                nn.SiLU(inplace=True))
            self.features.add_module(f'block{i}', conv_layer)

        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                  nn.Linear(hidden_dims[-1], num_classes))

        if reinit:
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        return self.features(x)

    def forward(self, x):
        features = self.forward_features(x)
        return self.head(features)
