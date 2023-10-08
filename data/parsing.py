import numpy as np
import stl
import pandas as pd

from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot


# Load the STL file.
my_mesh = mesh.Mesh.from_file('data/cube.stl')

# print(my_mesh.vectors)

# Initialise adjacency matrix with shape of NUM_VERTICES ** 2.
vertices_num = len(my_mesh.points) * 3
adjacency_matrix = np.zeros(
    shape=(vertices_num, len(my_mesh.points) * 3), dtype=int)

# Go through each vertex and check
# O(n ** 2) complexity to build the adjacency matrix.
for i in range(len(my_mesh.vectors)):
    triangle1 = my_mesh.vectors[i]

    for j in range(len(triangle1)):
        triangle1_vertex = triangle1[j]

        for k in range(len(my_mesh.vectors)):
            triangle2 = my_mesh.vectors[j]

            for z in range(len(triangle2)):
                triangle2_vertex = triangle2[z]

                # TODO: Sanity check for float comparison.

                # Do not add the loop connection.
                if (triangle1_vertex == triangle2_vertex).all():
                    continue

                # Check if this vertex is within the same triangle with other vertex.
                if triangle1_vertex in triangle2:
                    adjacency_matrix[j][z] = 1

for row in adjacency_matrix:
    print(row)
