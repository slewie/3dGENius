import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data, Batch
import torch


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


class FeatureGenerator(nn.Module):
    """
    Generates a matrix with features, but now this matrix doesn't connect to a graph
    """

    def __init__(self, z_size, num_vertex, num_features):
        super(FeatureGenerator, self).__init__()
        self.num_features = num_features
        self.num_vertex = num_vertex
        self.model = nn.Sequential(
            nn.Linear(z_size, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(True),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(True),
        )
        self.out_graph = nn.Linear(1000, num_vertex * num_vertex)
        self.out_features = nn.Linear(1000, num_vertex * num_features)

    def forward(self, x):
        x = self.model(x)
        out_graph = self.out_graph(x)
        out_features = self.out_features(x)
        return F.sigmoid(out_features.reshape(-1, self.num_vertex, self.num_features)), F.sigmoid(
            out_graph.reshape(-1, self.num_vertex, self.num_vertex))


class FeatureDiscriminator(nn.Module):
    def __init__(self, num_vertex, num_features, output_dim=2):
        super(FeatureDiscriminator, self).__init__()
        self.output_dim = output_dim
        self.num_vertex = num_vertex
        self.conv = GCNConv(num_features, output_dim)
        self.linear = nn.Linear(output_dim * num_vertex, 1)

    def forward(self, x):
        features, graphs = x

        output = torch.zeros(features.shape[0], self.num_vertex, self.output_dim)
        for i, graph in enumerate(graphs):
            edge_index, _ = dense_to_sparse(graph)
            x = self.conv(features[i], edge_index)
            output[i] = x
        x = torch.flatten(F.relu(output), start_dim=1)
        return F.sigmoid(self.linear(x))
