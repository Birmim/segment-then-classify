import torch
import torch.nn as nn
from src.unet_model import DoubleConv


class LinearLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.Linear = nn.Sequential(
        nn.Linear(in_features=in_channels, out_features=out_channels),
        nn.ReLU(),
        nn.Dropout(0.5)
        )

    def forward(self, x):
        return self.Linear(x)

class CNN(nn.Module):
    def __init__(self, input_features, output_features=1, hidden_units=32):
        super().__init__()

        conections1 = hidden_units
        conections2 = hidden_units * 2
        conections3 = hidden_units * 4
        conections4 = hidden_units * 8
        conections5 = hidden_units * 16 

        conections6 = conections5 * 8 * 8

        self.enc_1 = DoubleConv(input_features, conections1)
        self.enc_2 = DoubleConv(conections1, conections2)
        self.enc_3 = DoubleConv(conections2, conections3)
        self.enc_4 = DoubleConv(conections3, conections4)
        self.bottleneck = DoubleConv(conections4, conections5)

        self.flat = nn.Flatten()
            
        self.layer_1 = LinearLayer(conections6, 128)
        self.layer_2 = LinearLayer(128, 64)
        self.final_layer = nn.Linear(64, output_features)

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )
    def forward(self, x : torch.Tensor):
        x1 = self.enc_1(x)
            
        x2 = self.enc_2(self.pool(x1))
        
        x3 = self.enc_3(self.pool(x2))
        
        x4 = self.enc_4(self.pool(x3))
        
        bottleneck = self.bottleneck(self.pool(x4))

        return (self.final_layer(self.layer_2(self.layer_1(self.flat(bottleneck))))).squeeze(1)