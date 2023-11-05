import numpy as np
import pandas as pd

from stl import mesh


class Parser:
    def parse(file_path: str, output_path: str = None) -> pd.DataFrame:
        """
        Function that parses stl file and returns adjacency matrix with coordinates of each vertex.

        file_path: path to the stl file.
        output_path: path, where to save the output file.
        """

        # Load the STL file.
        my_mesh = mesh.Mesh.from_file(file_path)

        # Initialise adjacency matrix with shape of NUM_VERTICES ** 2.
        vertices_num = len(my_mesh.points) * 3
        adjacency_matrix = np.zeros(
            shape=(vertices_num, len(my_mesh.points) * 3), dtype=int)
        print("Adjacency matrix shape: ", adjacency_matrix.shape)

        # Store coordinates of vertices.
        vertices_coordinates = []

        # Go through each vertex and check
        # O(n ** 2) complexity to build the adjacency matrix.
        for i in range(len(my_mesh.vectors)):
            triangle1 = np.array(my_mesh.vectors[i])

            for j in range(len(triangle1)):
                triangle1_vertex = np.array(triangle1[j])

                # print(triangle1_vertex)
                vertices_coordinates.append(triangle1_vertex)

                for k in range(len(my_mesh.vectors)):
                    triangle2 = np.array(my_mesh.vectors[k])
                    is_adjaced = False

                    # Check if this vertex is within the same triangle with another vertex.
                    for triangle2_vertex in triangle2:
                        if (triangle1_vertex == triangle2_vertex).all():
                            is_adjaced = True
                            break

                    for z in range(len(triangle2)):
                        triangle2_vertex = np.array(triangle2[z])

                        # TODO: Sanity check for float comparison.

                        # Do not add the loop connection.
                        if (triangle1_vertex == triangle2_vertex).all():
                            continue
                        elif is_adjaced:
                            adjacency_matrix[i * 3 + j][k * 3 + z] = 1

        adj_matrix_df = pd.DataFrame(data=adjacency_matrix)
        coord_matrix_df = pd.DataFrame(
            data=vertices_coordinates, columns=["x", "y", "z"])

        # Concatenate dataframes.
        result = pd.concat([adj_matrix_df, coord_matrix_df], axis=1)

        if output_path:
            result.to_csv("data/cube_preprocessed.csv")
        return result
