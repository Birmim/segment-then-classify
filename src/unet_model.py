import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
        nn.Conv2d(in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1
                ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        )
    def forward(self, x):
        return self.conv(x)

class Unet(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=64):
        super().__init__()

        conections1 = hidden_units
        conections2 = hidden_units * 2
        conections3 = hidden_units * 4
        conections4 = hidden_units * 8
        conections5 = hidden_units * 16

        self.enc_1 = DoubleConv(input_features, conections1)
        self.enc_2 = DoubleConv(conections1, conections2)
        self.enc_3 = DoubleConv(conections2, conections3)
        self.enc_4 = DoubleConv(conections3, conections4)
        self.bottleneck = DoubleConv(conections4, conections5)

        self.pool = nn.MaxPool2d(kernel_size=2,
                                 stride=2
                                 )

        self.conv_final = nn.Conv2d(
                    hidden_units,
                    output_features,
                    kernel_size = 1
        )

        
        self.dec_4 = nn.ConvTranspose2d(conections5, conections4, 2, stride=2)
        self.dec_3 = nn.ConvTranspose2d(conections4, conections3, 2, stride=2)
        self.dec_2 = nn.ConvTranspose2d(conections3, conections2, 2, stride=2)
        self.dec_1 = nn.ConvTranspose2d(conections2, conections1, 2, stride=2)

        self.dec_conv_4 = DoubleConv(conections5, conections4)
        self.dec_conv_3 = DoubleConv(conections4, conections3)
        self.dec_conv_2 = DoubleConv(conections3, conections2)
        self.dec_conv_1 = DoubleConv(conections2, conections1)

    def forward(self, x : torch.Tensor):
        x1 = self.enc_1(x)
    
        x2 = self.enc_2(self.pool(x1))

        x3 = self.enc_3(self.pool(x2))

        x4 = self.enc_4(self.pool(x3))

        bottleneck = self.bottleneck(self.pool(x4))

        y4 = self.dec_4(bottleneck)
        y4_skip = torch.cat([x4, y4], dim=1)
        y4 = self.dec_conv_4(y4_skip)

        y3 = self.dec_3(y4)
        y3_skip = torch.cat([x3, y3], dim=1)
        y3 = self.dec_conv_3(y3_skip)

        y2 = self.dec_2(y3)
        y2_skip = torch.cat([x2, y2], dim=1)
        y2 = self.dec_conv_2(y2_skip)

        y1 = self.dec_1(y2)
        y1_skip = torch.cat([x1, y1], dim=1)
        y1 = self.dec_conv_1(y1_skip)
        
        y = self.conv_final(y1)

        return y
