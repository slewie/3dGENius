# 3dGENius

This repository contains code for the project about using GANs for 3D model generation. It represents 3D models as graphs and generates both the adjacency matrix and vertex feature matrix using separate GANs. The models can then be constructed into triangle meshes.

## Models

The repository contains the following main model code:

- `gan_model.py`: Defines the `GraphGAN` class which encapsulates the GAN model. It contains the generator, discriminator, loss functions, and training logic.
- `networks.py`: Defines the generator and discriminator network architectures, including `AdjacencyGenerator`, `AdjacencyDiscriminator`, `FeatureGenerator`, `FeatureDiscriminator`.

The `GraphGAN` class trains a generator and discriminator model for either adjacency matrix or vertex feature generation. There are separate `GraphGAN` instances for the adjacency and feature models.

## Training

The training pipeline is managed in `trainer.py`. Key aspects:

- Loads data from an STL dataset
- Defines separate `GraphGAN` models for adjacency and features
- Trains each model for a specified number of epochs
- Saves the trained generator model state at the end

## Inference

The `predictor.py` script handles inference using trained models. Key details:

- Loads trained adjacency and feature generator models
- Feeds random input noise vectors to generate new graphs
- Constructs triangle mesh from generated adjacency and features
- Outputs .stl model file

## Usage

To train models:

```python trainer.py --num_vertex <num> --data_path <path> ...```

To run inference with trained models:

```python predictor.py --num_vertex <num> --model_adj_path <path> --model_feature_path <path> ...```
