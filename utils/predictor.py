import sys
from pathlib import Path

current_file_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_file_dir.parent))

from models.gan_model import *
from models.networks import *
import numpy as np
import trimesh
import argparse


class Predictor:
    def __init__(self, num_vertex, model_adj_path, model_feature_path, z_size=100, device='cuda', lr=1e-4):
        self.num_vertex = num_vertex
        self.model_adj_path = model_adj_path
        self.model_feature_path = model_feature_path
        self.z_size = z_size
        self.device = device
        self.lr = lr

        self.adj_gan = GraphGAN(num_vertex=num_vertex, lr=lr, device=device)
        self.feature_gan = GraphGAN(num_vertex=num_vertex, generator=FeatureGenerator(z_size, num_vertex, 3),
                                    discriminator=FeatureDiscriminator(num_vertex, 3), lr=lr, device=device)

        self.adj_gan.load_state_dict(torch.load(model_adj_path))
        self.feature_gan.load_state_dict(torch.load(model_feature_path))

    def inference(self, file_path):
        graph_output = self.adj_gan.generate_adj(return_symmetric=True)
        graph_output = graph_output.cpu().detach().numpy()
        feature_matrix = self.feature_gan.generate_feat().cpu().detach().numpy().squeeze(0)

        triangles = []
        for i in range(len(graph_output)):
            for j in range(i + 1, len(graph_output)):
                if graph_output[i, j] == 1:
                    for k in range(j + 1, len(graph_output)):
                        if graph_output[j, k] == 1 and graph_output[k, i] == 1:
                            triangles.append([i, j, k])

        faces = np.array(triangles)

        mesh = trimesh.Trimesh(vertices=feature_matrix, faces=faces)
        mesh.export(file_path + '.stl')


parser = argparse.ArgumentParser(description='Inference script for GAN model.')
parser.add_argument('--num_vertex', type=int, required=True, help='Number of vertices.')
parser.add_argument('--model_adj_path', type=str, required=True, help='Path to the adjacency model.')
parser.add_argument('--model_feature_path', type=str, required=True, help='Path to the feature model.')
parser.add_argument('--z_size', type=int, default=100, help='Size of z.')
parser.add_argument('--device', type=str, default='cuda', help='Device to use for computations.')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
parser.add_argument('--file_path', type=str, required=True, help='Path to the output file.')
args = parser.parse_args()
predictor = Predictor(args.num_vertex, args.model_adj_path, args.model_feature_path, args.z_size, args.device, args.lr)
predictor.inference(args.file_path)
