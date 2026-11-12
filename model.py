import torch
import torch.nn as nn
class Unflatten(nn.Module):
    def __init__(self, channels, h, w):
        super(Unflatten, self).__init__()
        self.channels = channels
        self.h = h
        self.w = w
    def forward(self, w):
        return x.view(x.size(0), self.channels, self.h, self.w)

class CsiAutoencoder(nn.Module):
    def __init__(self, num_anteennas=32, latent_dim=64):
        super(CsiAutoencoder, self).__init__()

        # Iput channels = 2 (Real and imaginary parts)
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.flatten(),
            nn.Linear(16 * num_antennas * num_antennas, latent_dim),
            nn.Tanh()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16 * num_antennas * num_antennas),
            nn.ReLU(),
            Unflatten(16, num_antennas, num_antennas),
            nn.ConvTranspose2d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 2, kernel_size=3, padding= 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded