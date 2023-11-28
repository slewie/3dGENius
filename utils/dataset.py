import trimesh
import networkx as nx
import torch
from torch.utils.data import Dataset


class STLDataset(Dataset):
    def __init__(self, root_path, tensor_size):
        self.stl_path = root_path
        self.stl_list = sorted(self._get_filenames(self.stl_path))
        self.tensor_size = tensor_size

    def __getitem__(self, idx):
        mesh = trimesh.load_mesh(self.stl_list[idx])
        try:
            adj = torch.from_numpy(
                nx.adjacency_matrix(trimesh.graph.vertex_adjacency_graph(mesh)
                                    ).toarray()).float()

            vertices_coords = self.fix_size_coords(torch.from_numpy(mesh.vertices), tensor_size=self.tensor_size)
            adj = self.fix_size_adj(adj, tensor_size=self.tensor_size)
            return adj, vertices_coords
        except:
            print(f'Error with {self.stl_list[idx]}')

    def __len__(self):
        return len(self.stl_list)

    @staticmethod
    def _get_filenames(path):
        return [f for f in path.iterdir() if f.is_file()]

    @staticmethod
    def fix_size_adj(input_tensor, tensor_size):
        if input_tensor.shape[0] < tensor_size:
            zeros = torch.zeros(input_tensor.shape[0], tensor_size - input_tensor.shape[0])
            tensor = torch.cat([input_tensor, zeros], dim=1)

            zeros = torch.zeros(tensor_size - input_tensor.shape[0], tensor_size)
            tensor = torch.cat([tensor, zeros], dim=0)
            return tensor
        elif input_tensor.shape[0] > tensor_size:
            return input_tensor[:tensor_size, :tensor_size]
        else:
            return input_tensor

    @staticmethod
    def fix_size_coords(input_tensor, tensor_size):
        if input_tensor.shape[0] < tensor_size:
            zeros = torch.zeros(tensor_size - input_tensor.shape[0], input_tensor.shape[1])
            tensor = torch.cat([input_tensor, zeros], dim=0)

            return tensor
        elif input_tensor.shape[0] > tensor_size:
            return input_tensor[:tensor_size]
        else:
            return input_tensor
