import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .config import FEATURE_MAP_SIZE, SHARED_CONV_CHANNELS, SHARED_FC_DIM, TASKS
except ImportError:
    from config import FEATURE_MAP_SIZE, SHARED_CONV_CHANNELS, SHARED_FC_DIM, TASKS


def _init_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class _SharedBackend(nn.Module):
    def __init__(self, in_channels=8, shared_conv_channels=SHARED_CONV_CHANNELS, shared_fc_dim=SHARED_FC_DIM):
        super().__init__()
        self.shared_conv = nn.Sequential(
            nn.Dropout2d(0.2),
            nn.Conv2d(in_channels, shared_conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(shared_conv_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        feat_size = shared_conv_channels * (FEATURE_MAP_SIZE // 2) * (FEATURE_MAP_SIZE // 2)
        self.flatten = nn.Flatten()
        self.shared_fc = nn.Sequential(
            nn.Linear(feat_size, shared_fc_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.heads = nn.ModuleList(nn.Linear(shared_fc_dim, task.num_classes) for task in TASKS)

    def forward_backend(self, x, task_id):
        x = self.shared_conv(x)
        x = self.flatten(x)
        x = self.shared_fc(x)
        return self.heads[task_id](x)


class HardRoutingPMoE(_SharedBackend):
    """Hard-routing PMoE backend with task-specific normalization and zero-padding to 8 channels."""

    def __init__(self):
        super().__init__(in_channels=8)
        self.adapters = nn.ModuleList(nn.BatchNorm2d(task.hard_channels) for task in TASKS)
        self.apply(_init_weights)

    def forward(self, x, task_id):
        x = self.adapters[task_id](x)
        channels_to_pad = 8 - x.size(1)
        if channels_to_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, channels_to_pad))
        return self.forward_backend(x, task_id)


class WeightedRoutingPMoE(_SharedBackend):
    """Weighted-routing PMoE backend with a trainable 1x1 projection from 18 to 8 channels."""

    def __init__(self):
        super().__init__(in_channels=8)
        self.optical_matrix = nn.Conv2d(18, 8, kernel_size=1, bias=False)
        self.adapters = nn.ModuleList(nn.BatchNorm2d(8) for _ in TASKS)
        self.apply(_init_weights)
        nn.init.kaiming_normal_(self.optical_matrix.weight, mode="fan_in", nonlinearity="linear")

    def forward(self, x, task_id):
        x = self.optical_matrix(x)
        x = self.adapters[task_id](x)
        return self.forward_backend(x, task_id)
