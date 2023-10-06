import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):

    def __init__(self, z_size, ngf, conv_dim=32):
        super(Generator, self).__init__()
        self.conv_dim = conv_dim
        self.input_layer = nn.Linear(in_features=z_size, out_features=2048, bias=True)
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=ngf * 2),
            nn.Tanh(),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features=ngf),
            nn.Tanh(),
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = x.view(-1, self.conv_dim * 4, 4, 4)
        return self.model(x)
