import os

# Shared Hyperparameters
epochs = 100
lr = 0.01
sequence_length = 30
hidden_size = 256
num_layers = 2
dataset_limit = 800000

# File Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
shakespeare_path = os.path.join(base_dir, "shakespeare.txt")
weights_path = os.path.join(base_dir, "../weights/weights_2layer.json")
pytorch_weights_path = os.path.join(base_dir, "../Pytorch/weights_pytorch.pth")
