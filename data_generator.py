import numpy as np
import torch

def generate_synthetic_channels(num_samples=1000, num_antennas=32):
    """
    Generates synthetic complex channel matrices H of shape (num_samples, 2, antennas, antennas)
    where channel 0 is real part and channel 1 is imaginary part.
    """
    # Generating complex Gaussian Random Matrices
    real_part = np.random.randn(num_samples, 1, num_antennas, num_antennas) / np.sqrt(2)
    imag_part = np.random.randn(num_samples, 1, num_antennas, num_antennas) / np.sqrt(2)

    # Concatenate along channel dimension to form (Batch, 2, H, W)
    channels = np.concatenate([real_part, imag_part], axis = 1)
    return torch.tensor(channels, dtype=torch.float32)

if __name__ == "--main__":
    dataset = generate_synthetic_channels(num_samples=10, num_antennas=32)
    print("Dataset shape generated successfully:", dataset.shape)