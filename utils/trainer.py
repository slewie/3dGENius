import sys
from pathlib import Path
import os
current_file_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_file_dir.parent))

from torch.utils.data import DataLoader
from models.gan_model import *
from models.networks import *
from utils.dataset import STLDataset

from tqdm import tqdm
import argparse


class Trainer:
    def __init__(self, num_vertex, data_path, z_size=100, batch_size=512, device='cuda', lr=1e-4):
        self.num_vertex = num_vertex
        self.data_path = data_path
        self.z_size = z_size
        self.batch_size = batch_size
        self.device = device
        self.lr = lr

        self.adj_gan = GraphGAN(num_vertex=num_vertex, lr=lr, device=device)
        self.feature_gan = GraphGAN(num_vertex=num_vertex, generator=FeatureGenerator(z_size, num_vertex, 3),
                                    discriminator=FeatureDiscriminator(num_vertex, 3), lr=lr, device=device)
        data_path = os.path.join(current_file_dir, args.data_path)
        self.dataset = STLDataset(Path(data_path), tensor_size=num_vertex)

    def train(self, save_path_adj, save_path_feature, num_epochs=10):
        data_loader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True
        )
        print("Train first gan")
        bar = tqdm(range(num_epochs))
        for epoch in bar:
            g_l = 0
            d_l = 0
            for batch in data_loader:
                adj, _ = batch
                image_size = adj.size(0)

                gl, dl = self.adj_gan.forward(adj.to(self.device), image_size)
                g_l += gl
                d_l += dl
            bar.set_postfix_str(
                f"Epoch: {epoch + 1}/{num_epochs}\td_loss: {d_l / len(data_loader)}\tg_loss: {g_l / len(data_loader)}")
        torch.save(self.adj_gan.state_dict(), save_path_adj)

        print("Train second gan")
        bar = tqdm(range(num_epochs))
        for epoch in bar:
            g_l = 0
            d_l = 0
            for batch in data_loader:
                _, features = batch
                image_size = features.size(0)

                gl, dl = self.feature_gan.forward(features.float().to(self.device), image_size)
                g_l += gl
                d_l += dl
            bar.set_postfix_str(
                f"Epoch: {epoch + 1}/{num_epochs}\td_loss: {d_l / len(data_loader)}\tg_loss: {g_l / len(data_loader)}")
        torch.save(self.feature_gan.state_dict(), save_path_feature)


parser = argparse.ArgumentParser(description="Train a GAN model.")
parser.add_argument('--num_vertex', type=int, help='Number of vertices.')
parser.add_argument('--data_path', type=str, help='Path to the data.')
parser.add_argument('--z_size', type=int, default=100, help='Size of z.')
parser.add_argument('--batch_size', type=int, default=512, help='Batch size.')
parser.add_argument('--device', type=str, default='cuda', help='Device to use for training.')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
parser.add_argument('--save_path_adj', type=str, help='Path to save the adj GAN model.')
parser.add_argument('--save_path_feature', type=str, help='Path to save the feature GAN model.')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs for training.')

args = parser.parse_args()

trainer = Trainer(args.num_vertex, args.data_path, args.z_size, args.batch_size, args.device, args.lr)
trainer.train(args.save_path_adj, args.save_path_feature, args.num_epochs)
