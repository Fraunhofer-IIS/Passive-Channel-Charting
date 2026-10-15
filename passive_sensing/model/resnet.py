import torch.nn as nn
from torch import Tensor

class ResNetBuildingBlock(nn.Module): 
    def __init__(self, in_channels, out_channels, kernel = 3, stride=1, bias=False): 
        super().__init__()    

        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = kernel, stride = stride, padding = int(stride / 2) if stride != 1 else 1, bias = bias),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = kernel, stride = 1, padding = 1, bias = bias),
                        nn.BatchNorm2d(out_channels))
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU()
    
    def forward(self, x : Tensor): 
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out
    

class ResNet(nn.Module):

    def __init__(self, in_dims, dims, kernels, num_blocks, strides, out_dim=2, bias=False):
        super(ResNet, self).__init__()
        assert len(dims) > 0, "There must be at least one layer!"
        assert len(dims) == len(num_blocks) and len(dims) == len(kernels) and len(dims) == len(strides), "The length of the dimensions array must be the same as the length of blocks array!"

        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_dims, dims[0], kernel_size = 1, stride = 1, padding = "same"),
                        nn.BatchNorm2d(dims[0]),
                        nn.ReLU())
        layers = []
        for layer_id in range(len(kernels)):
            dim = dims[layer_id] # Dim of the current layer
            next_dim = dims[layer_id + 1] if layer_id < len(num_blocks) - 1 else dim # Dim of the next layer
            stride = strides[layer_id]
            kernel = kernels[layer_id]
            num_block = num_blocks[layer_id]
            for block in range(num_block):
                if block == len(num_blocks) -1:
                    # Last block in current layer
                    layer = ResNetBuildingBlock(in_channels=dim, out_channels=next_dim, stride=stride, bias=bias, kernel=kernel)
                else:
                    layer = ResNetBuildingBlock(in_channels=dim, out_channels=dim, stride=1, bias=bias, kernel=kernel)
                layers.append(layer)
        
        self.layers = nn.Sequential(*layers)
        self.global_avg_pool2d = lambda x: x.mean([2,3]) # Mean over the W and H dimension, leaving batch and channels

        self.fc1 = nn.Linear(dim, 200)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(200, out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layers(x)
        x = self.global_avg_pool2d(x)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x