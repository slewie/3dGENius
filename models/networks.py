import torch.nn as nn


class Discriminator(nn.Module):
    """
    The class responsible for the discriminator.
    It classifies the input graph and returns a probability of how likely it is that the given graph is real or synthetic.
    """

    def __init__(self, num_vertex):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_vertex * num_vertex, 1000),
            nn.ReLU(True),
            nn.Linear(1000, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class Generator(nn.Module):
    """
    The class responsible for the generator.
    It generates a graph with `num_vertex` vertices from input noize
    """

    def __init__(self, z_size, num_vertex):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(z_size, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(True),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(True),
            nn.Linear(1000, num_vertex * num_vertex),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
