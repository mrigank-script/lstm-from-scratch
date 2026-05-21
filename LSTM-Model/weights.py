import json
import os

# Save model weights to file
def save_weights(filepath, weights):
    folder = os.path.dirname(filepath)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)
    with open(filepath, "w") as f:
        json.dump(weights, f)

# Load model weights from file
def load_weights(filepath):
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r") as f:
        return json.load(f)
