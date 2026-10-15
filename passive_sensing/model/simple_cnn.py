import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor

class BasicConv2d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        **kwargs
    ) -> None:
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def freeze_params(self, freeze=True):
        freeze=not freeze

        self.conv.weight.requires_grad = freeze
        self.bn.weight.requires_grad = freeze
        self.bn.bias.requires_grad = freeze

    def reset_parameters(self):
        self.conv.reset_parameters()
        self.bn.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
    

class Net(nn.Module):
    
    def __init__(self, in_dims = 1):
        super(Net, self).__init__()
        self.conv1 = BasicConv2d(in_dims, 32, kernel_size=3, padding=1) 
        self.conv2 = BasicConv2d(32, 32, kernel_size=5, padding=2)
        self.conv3 = BasicConv2d(32, 32, kernel_size=7, padding=3)
        self.conv4 = BasicConv2d(32, 64, kernel_size=10, padding=5) 
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 14, 90))
        self.fc1 = nn.Linear(1260, 250)
        self.fc2 = nn.Linear(250, 2)

    def forward(self, x):
        if len(x.shape) == 3:
            # Expand channel dimension if no present
            x = x.unsqueeze(1)
        
        # [N, 2, 10, 60]
        x = self.conv1(x)
        # [N, 32, 10, 60]
        x = self.conv2(x)
        # [N, 32, 10, 60]
        x = self.conv3(x)
        # [N, 32, 10, 60]
        x = self.conv4(x)
        # [N, 64, 10, 60]
        x = self.avgpool(x)
        # [N, 1, 10, 60]
        x = torch.flatten(x, 1)
        # [N, 1, 600]
        x = F.relu(self.fc1(x), inplace=True)
        # [N, 1, 300]
        x = self.fc2(x)

        return x
    

class CNNOptimized(nn.Module):
    
    def __init__(self, in_dims, out_dim):
        super(CNNOptimized, self).__init__()
        self.conv1 = BasicConv2d(in_dims, 8, kernel_size=3, padding=1) # TODO: Change padding to 1 to remain exact time resolution
        self.conv2 = BasicConv2d(8, 32, kernel_size=5, padding=2)
        self.conv3 = BasicConv2d(32, 64, kernel_size=8, padding=3)
        self.conv4 = BasicConv2d(64, 128, kernel_size=10, padding=5) # TODO: Change padding to 4 to remain exact time resolution
        
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(128, 200)
        self.fc2 = nn.Linear(200, out_dim)

    def forward(self, input_values, **kwargs):
        x: torch.Tensor = input_values
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.mean([-2, -1]) # Mean over the time and antenna dimension, so we only have the channel dimension left.
        x = F.relu(self.fc1(x), inplace=True)
        x = self.dropout(x)
        # [N, 1, 300]
        x = self.fc2(x)
        return x
