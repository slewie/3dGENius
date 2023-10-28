from models.networks import *
from torch import nn
import torch


class GraphGAN(nn.Module):

    def __init__(self, z_size=100, num_vertex=100, lr=0.001, beta1=0.9, beta2=0.999, device='cpu'):
        super(GraphGAN, self).__init__()

        self.z_size = z_size
        self.device = device

        self.generator = Generator(z_size, num_vertex)
        self.discriminator = Discriminator(num_vertex)

    def real_loss(self, d_out, batch_size, smooth=False):
        # label smoothing
        if smooth:
            labels = torch.FloatTensor(batch_size).uniform_(0.9, 1).to(self.device)
        else:
            labels = torch.ones(batch_size)

        labels = labels.to(self.device)
        criterion = nn.BCELoss()
        loss = criterion(d_out.squeeze(), labels)
        return loss

    def fake_loss(self, d_out, batch_size):
        labels = torch.FloatTensor(batch_size).uniform_(0, 0.1).to(self.device)  # fake labels = 0
        labels = labels.to(self.device)
        criterion = nn.BCELoss()
        # calculate loss
        loss = criterion(d_out.squeeze(), labels)
        return loss

    def discriminator_part(self, real_images, batch_size):
        D_real = self.discriminator(real_images)
        d_real_loss = self.real_loss(D_real, batch_size)

        z = torch.FloatTensor(batch_size, 100).uniform_(-1, 1).to(self.device)
        fake_images = self.generator(z)

        D_fake = self.discriminator(fake_images)
        d_fake_loss = self.fake_loss(D_fake, batch_size)

        d_loss = d_real_loss + d_fake_loss
        return d_loss

    def generator_part(self, batch_size):
        z = torch.FloatTensor(batch_size, 100).uniform_(-1, 1).to(self.device)

        fake_images = self.generator(z)

        D_fake = self.discriminator(fake_images)
        g_loss = self.real_loss(D_fake, batch_size)
        return g_loss
