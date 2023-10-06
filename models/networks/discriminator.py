import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):

    def __init__(self, ndf=3, conv_dim=32):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, 1, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1),
            nn.Flatten(),
            nn.Linear(5 * 5, 1)
        )

    def forward(self, x):
        return self.model(x)
