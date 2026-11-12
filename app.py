import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="CSI Feedback Compression in Massive MIMO", layout="wide")

st.title("📡 AI-Powered CSI Feedback Compression System for 5G/6G Massive MIMO")
st.markdown("### Term Project - IIT Guwahati (BSc Data Science & AI)")

# --- 1. MODEL ARCHITECTURE DEFINED DIRECTLY HERE ---
class Unflatten(nn.Module):
    def __init__(self, channels, h, w):
        super(Unflatten, self).__init__()
        self.channels = channels
        self.h = h
        self.w = w

    def forward(self, x):
        return x.view(x.size(0), self.channels, self.h, self.w)

class CsiAutoencoder(nn.Module):
    def __init__(self, num_antennas=32, latent_dim=64):
        super(CsiAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Flatten(),
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
            nn.ConvTranspose2d(8, 2, kernel_size=3, padding=1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# --- 2. DATA GENERATOR FUNCTION ---
def generate_synthetic_channels(num_samples=100, num_antennas=32):
    real_part = np.random.randn(num_samples, 1, num_antennas, num_antennas) / np.sqrt(2)
    imag_part = np.random.randn(num_samples, 1, num_antennas, num_antennas) / np.sqrt(2)
    channels = np.concatenate([real_part, imag_part], axis=1)
    return torch.tensor(channels, dtype=torch.float32)

# Sidebar controls
st.sidebar.header("Configuration Panel")
latent_dim = st.sidebar.slider("Latent Vector Dimension (Compression)", min_value=16, max_value=256, value=64, step=16)
num_antennas = st.sidebar.selectbox("Antenna Array Size", [16, 32], index=1)
epochs = st.sidebar.slider("Quick Training Epochs", min_value=1, max_value=10, value=3)

@st.cache_resource
def load_trained_model(antennas, lat_dim, n_epochs):
    data = generate_synthetic_channels(num_samples=100, num_antennas=antennas)
    model = CsiAutoencoder(num_antennas=antennas, latent_dim=lat_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()
        
    model.eval()
    return model, data

with st.spinner("Training Autoencoder on synthetic fading channels... please wait!"):
    model, dataset = load_trained_model(num_antennas, latent_dim, epochs)

sample_idx = st.slider("Select Sample Matrix Index", 0, 99, 0)
test_sample = dataset[sample_idx].unsqueeze(0)

with torch.no_grad():
    reconstructed = model(test_sample)

original_np = test_sample.numpy()
recon_np = reconstructed.numpy()
nmse = np.linalg.norm(original_np - recon_np)**2 / np.linalg.norm(original_np)**2

st.metric(label="Reconstruction NMSE (Lower is Better)", value=f"{nmse:.4f}")

st.markdown("### 📊 Channel Matrix Heatmap Comparison (Real Part)")
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

im0 = axes[0].imshow(original_np[0, 0, :, :], cmap='viridis')
axes[0].set_title("Original Channel Matrix H (Real)")
fig.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(recon_np[0, 0, :, :], cmap='viridis')
axes[1].set_title("Reconstructed Channel Matrix H_hat (Real)")
fig.colorbar(im1, ax=axes[1])

st.pyplot(fig)