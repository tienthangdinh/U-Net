import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False), #no bias so that we use batch norm, this actually keeps the dimension
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512], #out channel 1 (like output) use 1 out channel for mask, but the paper uses 2
    ):
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2): #because the ups has transpose, doubleconv, transpose, doubleconv
            x = self.ups[idx](x) #do transpose
            skip_connection = skip_connections[idx//2] #take the corresponding skip connection

            if x.shape != skip_connection.shape: #use torchvision.transforms.functional for functional transforms including resize
                x = TF.resize(x, size=skip_connection.shape[2:]) #shape of height and width, (ignore batch size and channel depth)

            concat_skip = torch.cat((skip_connection, x), dim=1) #concat skip connection with x along dimension 1 (dimension 0 is batch, dimension 1 is  the channel depth a.k.a the first actual architecture dimension, also the most well rounded) 
            x = self.ups[idx+1](concat_skip) #now add the doubleconv

        return self.final_conv(x)

def test():
    x = torch.randn((8, 1, 161, 161))
    model = UNET(in_channels=x.shape[1], out_channels=x.shape[1]) #take the second dimension aka number of channels aka 1
    preds = model(x)
    print(preds.shape)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()