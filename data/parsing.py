import numpy as np
import stl
import pandas as pd

from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot


# Load the STL file.
my_mesh = mesh.Mesh.from_file('data/cube.stl')

# Initialise adjacency matrix with shape of NUM_VERTICES ** 2.
vertices_num = len(my_mesh.points) * 3
adjacency_matrix = np.zeros(
    shape=(vertices_num, len(my_mesh.points) * 3), dtype=int)
print(adjacency_matrix)

# Go through each vertice, starting from the first to the last one in the file.
for triangle in my_mesh.points:
    # for vertice in range(0, len(triangle), 3):
    #     print(vertice)
    vertice1 = triangle[:3]
    vertice2 = triangle[3: 6]
    vertice3 = triangle[6:]

    if vertice1 not in vertices_dict:
        vertices_dict[vertice1] = []

    print(vertice1, vertice2, vertice3)
