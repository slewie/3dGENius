from models.networks import *
from torch import nn
import torch


class GraphGAN(nn.Module):
    """
    The class for the generating graphs using GAN architecture.
    """

    def __init__(self, z_size=100, num_vertex=100, lr=0.001, beta1=0.9, beta2=0.999, device='cpu',
                 criterion=nn.BCELoss(), generator=None, discriminator=None):
        super(GraphGAN, self).__init__()

        self.z_size = z_size
        self.device = device
        self.num_vertex = num_vertex
        self.criterion = criterion
        if generator is None:
            self.generator = Generator(z_size, num_vertex)
        else:
            self.generator = generator
        if discriminator is None:
            self.discriminator = Discriminator(num_vertex)
        else:
            self.discriminator = discriminator

        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, beta2))
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, beta2))

    def _real_loss(self, d_out, input_size, smooth=False):
        """
        Function that calculates loss for images with label 1. Also supports label smoothing
        :param d_out: input from the discriminator
        :param input_size: size of the input
        :param smooth: apply label smoothing or not
        :return: loss
        """
        # label smoothing
        if smooth:
            labels = torch.FloatTensor(input_size).uniform_(0.9, 1)
        else:
            labels = torch.ones(input_size)
        labels = labels.to(self.device)
        loss = self.criterion(d_out.squeeze(), labels)
        return loss

    def _fake_loss(self, d_out, input_size, smooth=False):
        """
        Function that calculates loss for images with label 0. Also supports label smoothing
        :param d_out: input from the discriminator
        :param input_size: size of the input
        :param smooth: apply label smoothing or not
        :return: loss
        """
        if smooth:
            labels = torch.FloatTensor(input_size).uniform_(0, 0.1)
        else:
            labels = torch.zeros(input_size)
        labels = labels.to(self.device)
        loss = self.criterion(d_out.squeeze(), labels)
        return loss

    def _discriminator_part(self, real_images, input_size):
        """
        Discriminator part of the GAN: classifies real images and fake images.
        :param real_images: batch of real images
        :return: discriminator loss
        """
        D_real = self.discriminator(real_images)

        z = torch.FloatTensor(input_size, 100).uniform_(-1, 1).to(self.device)
        fake_images = self.generator(z)

        D_fake = self.discriminator(fake_images)
        return self._fake_loss(D_fake, input_size) + self._real_loss(D_real, input_size)

    def _generator_part(self, input_size):
        """
        Discriminator part of the GAN: generates images from noise
        """
        z = torch.FloatTensor(input_size, 100).uniform_(-1, 1).to(self.device)

        fake_images = self.generator(z)

        D_fake = self.discriminator(fake_images)
        return self._real_loss(D_fake, input_size)

    def _get_symmetric_adjacency_matrix(self, tensor):
        """
        Makes input tensor symmetric
        :param tensor: input adjacency matrix
        """
        output = torch.zeros(self.num_vertex, self.num_vertex)
        for i in range(tensor.shape[1]):
            for j in range(tensor.shape[2]):
                output[i, j] = max(tensor[0, i, j], tensor[0, j, i])
        return output

    def generate_adj(self, return_symmetric=False):
        """
        Function generates graph from a random noise and return adjacency matrix of the graph
        :param return_symmetric: whether adjacency matrix symmetric or not
        """
        self.generator.eval()
        with torch.no_grad():
            z = torch.FloatTensor(1, 100).uniform_(-1, 1).to(self.device)
            adjacency_matrix = torch.round(self.generator(z).view(-1, self.num_vertex, self.num_vertex))
            if return_symmetric:
                return self._get_symmetric_adjacency_matrix(adjacency_matrix)
            return adjacency_matrix

    def generate_feat(self):
        self.generator.eval()
        with torch.no_grad():
            z = torch.FloatTensor(1, 100).uniform_(-1, 1).to(self.device)
            feature_matrix = torch.round(self.generator(z).view(-1, self.num_vertex, self.generator.num_features))
            return feature_matrix

    def forward(self, real_image, input_size):
        # TODO: create methods for multiplicating in order to solve ordering invariant problem
        self.generator.train()
        self.discriminator.train()

        self.optimizer_d.zero_grad()

        d_loss = self._discriminator_part(real_image, input_size)
        d_loss.backward()
        self.optimizer_d.step()

        self.optimizer_g.zero_grad()

        g_loss = self._generator_part(input_size)

        g_loss.backward()
        self.optimizer_g.step()

        return g_loss.item(), d_loss.item()
