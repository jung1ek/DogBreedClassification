import torch
import torch.nn as nn
from math import ceil

# expand_ratio, channels, repeats or layers, strides, kernal_size
base_model = [[1,16,1,1,3], 
              [6,24,2,2,3], 
              [6,40,2,2,5], 
              [6,80,3,2,3], 
              [6,112,3,1,5],
              [6,192,4,2,5],
              [6,320,1,1,3]
]

phi_values = {
    # tuple of : (phi_value, rsolution, drope_rate) for each models b0-b7
    'b0': (0,224,0.2), # alpha , beta and gamma, depth = alpha**phi
    'b1': (0.5,240,0.2),
    'b2': (1,260,0.3),
    'b3': (2,300,0.3),
    'b4': (3,380,0.4),
    'b5': (4,456,0.4),
    'b6': (5,528,0.5),
    'b7': (6,600,0.5)

}

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernal_size, stride,pading,groups=1):
        super(CNNBlock,self).__init__()
        self.cnn = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=in_channels
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()
    def forward(x):
        return self.silu(self.bn(self.cnn(x)))

class SqueezeExcitation(nn.Module): # compute attention score for each channel.
    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            

        )

class InvertedResidualBlock(nn.Module):
    pass

class EfficientNet(nn.Module):
    pass




