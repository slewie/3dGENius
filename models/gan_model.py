from models.networks import *
from torch import nn
import torch


class GraphGAN(nn.Module):

    def __init__(self, z_size=100, num_vertex=100, batch_size=128, lr=0.001, beta1=0.9, beta2=0.999):
        super(GraphGAN, self).__init__()

        self.generator = Generator(z_size, num_vertex)
        self.discriminator = Discriminator(num_vertex)

    def forward(self, real_data, latent_vector):
        # Generate fake data
        fake_data = self.generator(latent_vector)

        # Compute the discriminator outputs on real and fake data
        real_output_d = self.discriminator(real_data)
        fake_output_d = self.discriminator(fake_data.detach())

        # Compute the discriminator loss
        # d_loss_real = self.gan_loss(real_output, True)
        # d_loss_fake = self.gan_loss(fake_output, False)
        # d_loss = (d_loss_real + d_loss_fake) * 0.5

        # Compute the generator loss
        fake_output_g = self.discriminator(fake_data.detach())
        # g_loss = self.gan_loss(fake_output, True)

        return real_output_d, fake_output_d, fake_output_g
