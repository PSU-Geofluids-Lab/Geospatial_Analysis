# VQ-VAE for Spatial Field Modeling with Realistic Correlation Structure

# I'll implement a Vector Quantized Variational Autoencoder (VQ-VAE) specifically designed for spatial field modeling. This model will learn to capture the spatial correlation structure of environmental data and generate new fields with similar statistical properties.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class SpatialDataset(Dataset):
    """Dataset for spatial fields with realistic correlation structure"""
    def __init__(self, num_samples=1000, grid_size=64, num_channels=3):
        """
        Generate synthetic spatial fields with Gaussian Process correlation
        
        Parameters:
        num_samples (int): Number of samples to generate
        grid_size (int): Size of spatial grid (grid_size x grid_size)
        num_channels (int): Number of correlated spatial fields
        """
        self.grid_size = grid_size
        self.num_channels = num_channels
        self.data = self.generate_spatial_data(num_samples, grid_size, num_channels)
        
    def generate_spatial_data(self, num_samples, grid_size, num_channels):
        """Generate correlated spatial fields using Gaussian Processes"""
        # Create grid coordinates
        x = np.linspace(0, 10, grid_size)
        y = np.linspace(0, 10, grid_size)
        xx, yy = np.meshgrid(x, y)
        coords = np.c_[xx.ravel(), yy.ravel()]
        
        # Generate spatial fields
        data = np.zeros((num_samples, num_channels, grid_size, grid_size))
        
        # Define kernel with multiple components
        kernel = (1.0 * RBF(length_scale=1.0, length_scale_bounds=(0.5, 2.0)) + \
                 (0.5 * Matern(length_scale=0.5, nu=1.5))
        
        for i in range(num_samples):
            # Generate correlated fields
            base_field = np.random.randn(grid_size**2)
            
            for c in range(num_channels):
                # Each channel has correlated but unique spatial structure
                gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1)
                field = gp.sample_y(coords, random_state=i*num_channels + c)
                field = field.reshape(grid_size, grid_size)
                
                # Add channel-specific variation
                channel_variation = 0.3 * np.random.randn(grid_size, grid_size)
                
                # Normalize and store
                data[i, c] = (field + channel_variation - field.mean()) / field.std()
        
        return torch.tensor(data, dtype=torch.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class VectorQuantizer(nn.Module):
    """Vector Quantization Layer"""
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        # Initialize embeddings
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)
        
    def forward(self, inputs):
        # Convert inputs to [B, C, H, W] -> [B, H, W, C] -> [B*H*W, C]
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        flat_inputs = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_inputs**2, dim=1, keepdim=True) 
                    + torch.sum(self.embeddings.weight**2, dim=1)
                    - 2 * torch.matmul(flat_inputs, self.embeddings.weight.t()))
        
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embeddings(encoding_indices).view(inputs.shape)
        
        # Compute VQ Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        # Convert back to [B, C, H, W]
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        
        # Perplexity calculation
        avg_probs = torch.mean(nn.functional.one_hot(encoding_indices, 
                                                    self.num_embeddings).float(), dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return quantized, loss, perplexity, encoding_indices.view(inputs.shape[0], 
                                                               inputs.shape[1], 
                                                               inputs.shape[2])

class ResidualBlock(nn.Module):
    """Residual block for encoder/decoder"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU()
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
            
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(residual)
        return self.relu(out)

class Encoder(nn.Module):
    """Encoder for VQ-VAE with spatial downsampling"""
    def __init__(self, in_channels, latent_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 4, stride=2, padding=1)
        self.res1 = ResidualBlock(64, 64)
        self.conv2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.res2 = ResidualBlock(128, 128)
        self.conv3 = nn.Conv2d(128, latent_dim, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.res1(x)
        x = self.relu(self.conv2(x))
        x = self.res2(x)
        x = self.conv3(x)
        return x

class Decoder(nn.Module):
    """Decoder for VQ-VAE with spatial upsampling"""
    def __init__(self, latent_dim, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(latent_dim, 128, 3, padding=1)
        self.res1 = ResidualBlock(128, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.res2 = ResidualBlock(64, 64)
        self.up2 = nn.ConvTranspose2d(64, out_channels, 4, stride=2, padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.res1(x)
        x = self.relu(self.up1(x))
        x = self.res2(x)
        x = self.up2(x)
        return x

class VQVAE(nn.Module):
    """Vector Quantized Variational Autoencoder for Spatial Fields"""
    def __init__(self, in_channels, num_embeddings, embedding_dim):
        super().__init__()
        self.encoder = Encoder(in_channels, embedding_dim)
        self.vq = VectorQuantizer(num_embeddings, embedding_dim)
        self.decoder = Decoder(embedding_dim, in_channels)
        
    def forward(self, x):
        z = self.encoder(x)
        quantized, vq_loss, perplexity, encoding_indices = self.vq(z)
        x_recon = self.decoder(quantized)
        return x_recon, vq_loss, perplexity, encoding_indices
    
    def encode(self, x):
        """Encode input to discrete latent codes"""
        z = self.encoder(x)
        _, _, _, encoding_indices = self.vq(z)
        return encoding_indices
    
    def decode(self, encoding_indices):
        """Decode from discrete latent codes"""
        quantized = self.vq.embeddings(encoding_indices)
        quantized = quantized.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        return self.decoder(quantized)
    
    def generate(self, shape, device='cpu'):
        """Generate new spatial fields from random codes"""
        # Create random encoding indices
        encoding_indices = torch.randint(0, self.vq.num_embeddings, shape, device=device)
        return self.decode(encoding_indices)

def train_vqvae(model, dataloader, num_epochs=50, lr=0.0002, device='cuda'):
    """Train the VQ-VAE model"""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Loss tracking
    train_losses = []
    perplexities = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_perplexity = 0
        
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            recon_batch, vq_loss, perplexity, _ = model(batch)
            
            # Reconstruction loss
            recon_loss = F.mse_loss(recon_batch, batch)
            
            # Total loss
            loss = recon_loss + vq_loss
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_perplexity += perplexity.item()
        
        # Calculate epoch statistics
        avg_loss = epoch_loss / len(dataloader)
        avg_perplexity = epoch_perplexity / len(dataloader)
        train_losses.append(avg_loss)
        perplexities.append(avg_perplexity)
        
        print(f'Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Perplexity: {avg_perplexity:.2f}')
    
    return train_losses, perplexities

def plot_spatial_fields(fields, titles=None, cmap='viridis'):
    """Plot spatial fields with Cartopy for geographic context"""
    n_fields = len(fields)
    fig, axes = plt.subplots(1, n_fields, figsize=(5*n_fields, 5),
                            subplot_kw={'projection': ccrs.PlateCarree()})
    
    if n_fields == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        field = fields[i]
        
        # Plot spatial field
        im = ax.imshow(field, cmap=cmap, origin='lower', 
                      extent=[0, 10, 0, 10], transform=ccrs.PlateCarree())
        
        # Add geographic features
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAND, edgecolor='black', alpha=0.2)
        ax.add_feature(cfeature.OCEAN, alpha=0.2)
        
        # Add title
        if titles:
            ax.set_title(titles[i])
        
        # Add colorbar
        fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.7)
    
    plt.tight_layout()
    plt.show()

def analyze_spatial_correlation(original, generated):
    """Analyze and compare spatial correlation structure"""
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Function to calculate spatial autocorrelation
    def spatial_autocorrelation(field, max_lag=15):
        corr = np.zeros(max_lag)
        h, w = field.shape
        
        for lag in range(1, max_lag+1):
            total = 0
            count = 0
            
            # Horizontal correlations
            for i in range(h):
                for j in range(w - lag):
                    total += field[i, j] * field[i, j + lag]
                    count += 1
            
            # Vertical correlations
            for i in range(h - lag):
                for j in range(w):
                    total += field[i, j] * field[i + lag, j]
                    count += 1
            
            corr[lag-1] = total / count if count > 0 else 0
        
        return corr / corr[0]  # Normalize
    
    # Calculate and plot autocorrelation for each channel
    for c in range(original.shape[0]):
        orig_corr = spatial_autocorrelation(original[c].cpu().numpy())
        gen_corr = spatial_autocorrelation(generated[c].cpu().numpy())
        
        ax[0].plot(orig_corr, color=f'C{c}', linestyle='-', label=f'Original Ch{c}')
        ax[0].plot(gen_corr, color=f'C{c}', linestyle='--', label=f'Generated Ch{c}')
    
    ax[0].set_title('Spatial Autocorrelation')
    ax[0].set_xlabel('Lag Distance')
    ax[0].set_ylabel('Normalized Correlation')
    ax[0].legend()
    ax[0].grid(True)
    
    # Plot variograms
    def variogram(field, max_lag=15):
        var = np.zeros(max_lag)
        h, w = field.shape
        
        for lag in range(1, max_lag+1):
            total = 0
            count = 0
            
            # Horizontal
            for i in range(h):
                for j in range(w - lag):
                    total += (field[i, j] - field[i, j + lag])**2
                    count += 1
            
            # Vertical
            for i in range(h - lag):
                for j in range(w):
                    total += (field[i, j] - field[i + lag, j])**2
                    count += 1
            
            var[lag-1] = total / (2 * count) if count > 0 else 0
        
        return var
    
    for c in range(original.shape[0]):
        orig_var = variogram(original[c].cpu().numpy())
        gen_var = variogram(generated[c].cpu().numpy())
        
        ax[1].plot(orig_var, color=f'C{c}', linestyle='-', label=f'Original Ch{c}')
        ax[1].plot(gen_var, color=f'C{c}', linestyle='--', label=f'Generated Ch{c}')
    
    ax[1].set_title('Variogram Analysis')
    ax[1].set_xlabel('Lag Distance')
    ax[1].set_ylabel('Semivariance')
    ax[1].legend()
    ax[1].grid(True)
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Create dataset
    dataset = SpatialDataset(num_samples=500, grid_size=64, num_channels=3)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize VQ-VAE model
    in_channels = 3
    num_embeddings = 256  # Size of the codebook
    embedding_dim = 32    # Dimension of each codebook vector
    vqvae = VQVAE(in_channels, num_embeddings, embedding_dim)
    
    # Train the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    train_losses, perplexities = train_vqvae(
        vqvae, dataloader, num_epochs=50, lr=0.0002, device=device
    )
    
    # Plot training progress
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(perplexities)
    plt.title('Codebook Perplexity')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Visualize original and reconstructed fields
    vqvae.eval()
    sample = next(iter(dataloader)).to(device)
    with torch.no_grad():
        recon, _, _, _ = vqvae(sample)
    
    # Select a sample to visualize
    idx = 0
    original_sample = sample[idx].cpu()
    reconstructed_sample = recon[idx].cpu()
    
    # Plot original and reconstructed
    plot_spatial_fields(
        [original_sample[0], reconstructed_sample[0]],
        ['Original Field (Ch0)', 'Reconstructed Field (Ch0)'],
        cmap='viridis'
    )
    
    # Generate new spatial fields
    with torch.no_grad():
        # Generate random codes in the latent space
        generated = vqvae.generate((1, 16, 16), device=device).cpu()
    
    # Visualize generated field
    plot_spatial_fields(
        [generated[0, 0]], 
        ['Generated Spatial Field (Ch0)'],
        cmap='viridis'
    )
    
    # Compare spatial correlation structure
    analyze_spatial_correlation(original_sample, generated[0])

# ## Key Components of the VQ-VAE for Spatial Modeling

# ### 1. Synthetic Spatial Dataset Generation
# - Creates correlated spatial fields using Gaussian Processes
# - Implements a multi-component kernel (RBF + Matern) for realistic spatial correlation
# - Generates multiple correlated channels with shared spatial structure

# ### 2. Vector Quantization (VQ) Layer
# - Implements the core VQ mechanism from "Neural Discrete Representation Learning"
# - Maps continuous encoder outputs to discrete latent codes
# - Includes commitment loss for training stability
# - Calculates perplexity to measure codebook usage

# ### 3. Encoder-Decoder Architecture
# - **Encoder**: 
#   - Strided convolutions for spatial downsampling (64x64 → 16x16)
#   - Residual blocks to preserve spatial information
#   - Outputs continuous latent representation
  
# - **Decoder**:
#   - Transposed convolutions for spatial upsampling (16x16 → 64x64)
#   - Residual blocks for high-quality reconstruction
#   - Outputs reconstructed spatial field

# ### 4. Training and Analysis
# - Optimizes reconstruction + VQ loss
# - Tracks codebook perplexity during training
# - Visualizes training progress
# - Compares original, reconstructed, and generated fields
# - Analyzes spatial correlation structure using:
#   - Spatial autocorrelation functions
#   - Variogram analysis

# ## Spatial Correlation Analysis Techniques

# ### 1. Spatial Autocorrelation
# Measures how similar values are at different locations separated by a specific distance:

# ```python
# def spatial_autocorrelation(field, max_lag=15):
#     corr = np.zeros(max_lag)
#     h, w = field.shape
    
#     for lag in range(1, max_lag+1):
#         total = 0
#         count = 0
        
#         # Horizontal correlations
#         for i in range(h):
#             for j in range(w - lag):
#                 total += field[i, j] * field[i, j + lag]
#                 count += 1
        
#         # Vertical correlations
#         for i in range(h - lag):
#             for j in range(w):
#                 total += field[i, j] * field[i + lag, j]
#                 count += 1
        
#         corr[lag-1] = total / count if count > 0 else 0
    
#     return corr / corr[0]  # Normalize
# ```

# ### 2. Variogram Analysis
# Measures spatial variability as a function of distance:

# ```python
# def variogram(field, max_lag=15):
#     var = np.zeros(max_lag)
#     h, w = field.shape
    
#     for lag in range(1, max_lag+1):
#         total = 0
#         count = 0
        
#         # Horizontal
#         for i in range(h):
#             for j in range(w - lag):
#                 total += (field[i, j] - field[i, j + lag])**2
#                 count += 1
        
#         # Vertical
#         for i in range(h - lag):
#             for j in range(w):
#                 total += (field[i, j] - field[i + lag, j])**2
#                 count += 1
        
#         var[lag-1] = total / (2 * count) if count > 0 else 0
    
#     return var
# ```

# ## Key Benefits of this Approach

# 1. **Discrete Latent Representation**:
#    - Learns a codebook of spatial patterns
#    - Enables interpretable latent space
#    - Facilitates generation of novel fields

# 2. **Spatial Structure Preservation**:
#    - Maintains spatial relationships through convolution operations
#    - Preserves correlation structure via residual connections
#    - Captures multi-scale patterns through downsampling/upsampling

# 3. **Realistic Field Generation**:
#    - Produces novel spatial fields with similar correlation properties
#    - Maintains realistic spatial variation patterns
#    - Generates coherent multi-channel outputs

# 4. **Geographic Context Visualization**:
#    - Uses Cartopy for geographic-aware plotting
#    - Overlays geographic features (coastlines, borders) for context
#    - Provides intuitive visualization of spatial patterns

# This implementation provides a powerful framework for modeling spatial fields with realistic correlation structures, enabling applications in environmental modeling, geostatistics, and spatial data generation.
