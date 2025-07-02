# Modeling Complex Spatial Fields with Deep Gaussian Processes and Vine Copulas

I'll implement a comprehensive framework for modeling spatial fields using two advanced approaches: Deep Gaussian Processes (DGPs) for hierarchical modeling and Spatial Vine Copulas for capturing complex dependencies. The implementation will include:

1. **Deep Gaussian Processes** for hierarchical spatial modeling
2. **Vine Copula** framework for spatial dependence modeling
3. **Hybrid DGP-Copula** model for generating fields with complex statistical properties
4. **Spatial visualization** and **statistical validation** tools

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import gpytorch
import pyvinecopulib as pv
from scipy.stats import norm, gaussian_kde
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.preprocessing import QuantileTransformer

class DeepGaussianProcess:
    """Deep Gaussian Process for hierarchical spatial modeling"""
    def __init__(self, num_layers=3, kernel='RBF', grid_size=50):
        """
        Initialize a Deep Gaussian Process model
        
        Parameters:
        num_layers (int): Number of layers in the DGP hierarchy
        kernel (str): Kernel type ('RBF' or 'Matern')
        grid_size (int): Size of spatial grid
        """
        self.num_layers = num_layers
        self.grid_size = grid_size
        self.kernel_type = kernel
        self.layers = []
        self.train_x = None
        self.train_y = None
        
        # Create grid coordinates
        x = np.linspace(0, 10, grid_size)
        y = np.linspace(0, 10, grid_size)
        self.xx, self.yy = np.meshgrid(x, y)
        self.coords = np.c_[self.xx.ravel(), self.yy.ravel()]
        
    def build_kernel(self, input_dim=2):
        """Build GP kernel based on specified type"""
        if self.kernel_type == 'RBF':
            return gpytorch.kernels.RBFKernel(ard_num_dims=input_dim)
        elif self.kernel_type == 'Matern':
            return gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=input_dim)
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel_type}")
    
    class GPModel(gpytorch.models.ExactGP):
        """GPyTorch GP model"""
        def __init__(self, train_x, train_y, kernel, likelihood):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = kernel
            
        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def fit(self, X, y, num_epochs=100, lr=0.1):
        """
        Fit the DGP model to observed data
        
        Parameters:
        X (array): Input coordinates (n_samples, 2)
        y (array): Target values (n_samples,)
        num_epochs (int): Training epochs
        lr (float): Learning rate
        """
        self.train_x = torch.tensor(X, dtype=torch.float32)
        self.train_y = torch.tensor(y, dtype=torch.float32)
        
        # Initialize DGP layers
        self.layers = []
        current_input = self.train_x
        current_output = self.train_y
        
        for i in range(self.num_layers):
            # Create kernel for this layer
            kernel = self.build_kernel(input_dim=current_input.shape[1])
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            model = self.GPModel(current_input, current_output, kernel, likelihood)
            
            # Train the GP
            model.train()
            likelihood.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
            
            print(f"Training layer {i+1}/{self.num_layers}...")
            for epoch in range(num_epochs):
                optimizer.zero_grad()
                output = model(current_input)
                loss = -mll(output, current_output)
                loss.backward()
                optimizer.step()
                
                if (epoch + 1) % 20 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss.item():.3f}")
            
            # Store trained layer
            self.layers.append((model, likelihood))
            
            # Get latent representation for next layer
            model.eval()
            likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                latent = model(current_input).mean
            current_input = torch.cat([current_input, latent.unsqueeze(1)], dim=1)
            current_output = self.train_y
    
    def predict(self, X):
        """Predict using the DGP model"""
        current_input = torch.tensor(X, dtype=torch.float32)
        
        for i, (model, likelihood) in enumerate(self.layers):
            # Predict with current layer
            model.eval()
            likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                latent = model(current_input).mean
            
            # Prepare input for next layer
            if i < len(self.layers) - 1:
                current_input = torch.cat([current_input, latent.unsqueeze(1)], dim=1)
        
        return latent.numpy()
    
    def generate_field(self):
        """Generate a spatial field from the DGP"""
        return self.predict(self.coords).reshape(self.grid_size, self.grid_size)
    
    def visualize_field(self, field, title="DGP Spatial Field"):
        """Visualize generated spatial field"""
        plt.figure(figsize=(10, 8))
        plt.imshow(field, extent=(0, 10, 0, 10), origin='lower', cmap='viridis')
        plt.colorbar(label='Field Value')
        plt.title(title)
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.show()

class SpatialVineCopula:
    """Spatial Vine Copula for modeling complex dependencies"""
    def __init__(self, grid_size=50):
        """
        Initialize a Spatial Vine Copula model
        
        Parameters:
        grid_size (int): Size of spatial grid
        """
        self.grid_size = grid_size
        self.copula = None
        self.marginals = []
        
        # Create grid coordinates
        x = np.linspace(0, 10, grid_size)
        y = np.linspace(0, 10, grid_size)
        self.xx, self.yy = np.meshgrid(x, y)
        self.coords = np.c_[self.xx.ravel(), self.yy.ravel()]
    
    def fit(self, fields):
        """
        Fit the vine copula to spatial fields
        
        Parameters:
        fields (list): List of spatial fields (each as 2D array)
        """
        # Flatten fields into a matrix (n_samples x n_points)
        data_matrix = np.array([field.ravel() for field in fields]).T
        
        # Transform to uniform marginals
        self.transformer = QuantileTransformer(output_distribution='uniform')
        uniform_data = self.transformer.fit_transform(data_matrix)
        
        # Fit vine copula
        self.copula = pv.Vinecop(data=uniform_data, controls={
            'family_set': [pv.BicopFamily.gaussian, pv.BicopFamily.student, 
                          pv.BicopFamily.clayton, pv.BicopFamily.gumbel],
            'trunc_lvl': 3  # Truncation level for vine
        })
    
    def generate_field(self):
        """Generate a new spatial field from the copula"""
        # Generate samples from vine copula
        n_points = self.grid_size * self.grid_size
        uniform_samples = self.copula.simulate(n=1, qrng=True)[0].reshape(1, -1)
        
        # Transform back to original marginals
        field_samples = self.transformer.inverse_transform(uniform_samples)
        return field_samples.reshape(self.grid_size, self.grid_size)
    
    def visualize_field(self, field, title="Copula Spatial Field"):
        """Visualize generated spatial field"""
        plt.figure(figsize=(10, 8))
        plt.imshow(field, extent=(0, 10, 0, 10), origin='lower', cmap='viridis')
        plt.colorbar(label='Field Value')
        plt.title(title)
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.show()
    
    def plot_dependence_structure(self):
        """Visualize the copula dependence structure"""
        if self.copula is None:
            raise ValueError("Copula not fitted. Call fit() first.")
        
        # Plot vine structure
        plt.figure(figsize=(12, 10))
        self.copula.plot()
        plt.title('Vine Copula Structure')
        plt.show()
        
        # Plot pairwise dependencies
        fig, axs = plt.subplots(3, 3, figsize=(15, 15))
        axs = axs.flatten()
        
        # Select random pairs of points
        n_points = self.grid_size * self.grid_size
        pairs = np.random.choice(n_points, size=min(9, n_points), replace=False)
        
        for i, point_idx in enumerate(pairs):
            if i >= 9:
                break
                
            # Get data for this point across all fields
            data = self.transformer.inverse_transform(self.copula.data)[:, point_idx]
            
            # Kernel density estimation
            kde = gaussian_kde(data)
            x_vals = np.linspace(min(data), max(data), 200)
            axs[i].plot(x_vals, kde(x_vals), 'r-', linewidth=2)
            axs[i].set_title(f'Marginal Distribution at Point {point_idx}')
            axs[i].set_xlabel('Value')
            axs[i].set_ylabel('Density')
        
        plt.tight_layout()
        plt.show()

class HybridSpatialModel:
    """Hybrid DGP and Vine Copula model for spatial fields"""
    def __init__(self, grid_size=50, num_dgp_layers=3):
        """
        Initialize hybrid spatial model
        
        Parameters:
        grid_size (int): Size of spatial grid
        num_dgp_layers (int): Number of layers in DGP hierarchy
        """
        self.grid_size = grid_size
        self.dgp = DeepGaussianProcess(num_layers=num_dgp_layers, grid_size=grid_size)
        self.copula = SpatialVineCopula(grid_size=grid_size)
        self.reference_fields = []
        
        # Create grid coordinates
        x = np.linspace(0, 10, grid_size)
        y = np.linspace(0, 10, grid_size)
        self.xx, self.yy = np.meshgrid(x, y)
        self.coords = np.c_[self.xx.ravel(), self.yy.ravel()]
    
    def fit(self, X, y, num_dgp_epochs=100, num_copula_fields=50):
        """
        Fit the hybrid model
        
        Parameters:
        X (array): Input coordinates (n_samples, 2)
        y (array): Target values (n_samples,)
        num_dgp_epochs (int): Training epochs for DGP
        num_copula_fields (int): Number of fields to generate for copula training
        """
        # Step 1: Fit DGP to observed data
        print("Fitting Deep Gaussian Process...")
        self.dgp.fit(X, y, num_epochs=num_dgp_epochs)
        
        # Step 2: Generate reference fields from DGP
        print("Generating reference fields for copula...")
        self.reference_fields = []
        for _ in range(num_copula_fields):
            field = self.dgp.generate_field()
            self.reference_fields.append(field)
        
        # Step 3: Fit vine copula to DGP-generated fields
        print("Fitting Vine Copula to reference fields...")
        self.copula.fit(self.reference_fields)
    
    def generate_field(self, method='hybrid'):
        """
        Generate a spatial field
        
        Parameters:
        method (str): 'dgp', 'copula', or 'hybrid'
        """
        if method == 'dgp':
            return self.dgp.generate_field()
        elif method == 'copula':
            return self.copula.generate_field()
        elif method == 'hybrid':
            # Generate base field from DGP
            base_field = self.dgp.generate_field()
            
            # Get statistical properties from copula
            copula_field = self.copula.generate_field()
            
            # Blend approaches (simple averaging for demonstration)
            return 0.7 * base_field + 0.3 * copula_field
        else:
            raise ValueError("Invalid generation method")
    
    def visualize_comparison(self, reference_field=None):
        """Visualize comparison of generated fields"""
        if reference_field is None and self.reference_fields:
            reference_field = self.reference_fields[0]
        
        methods = ['dgp', 'copula', 'hybrid']
        fields = {method: self.generate_field(method) for method in methods}
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        # Plot reference field
        if reference_field is not None:
            im0 = axes[0, 0].imshow(reference_field, extent=(0, 10, 0, 10), 
                                    origin='lower', cmap='viridis')
            axes[0, 0].set_title('Reference Field (DGP Generated)')
            fig.colorbar(im0, ax=axes[0, 0])
        
        # Plot generated fields
        for i, method in enumerate(methods):
            ax = axes[(i+1)//2, (i+1)%2]
            im = ax.imshow(fields[method], extent=(0, 10, 0, 10), 
                          origin='lower', cmap='viridis')
            ax.set_title(f'{method.upper()} Generated Field')
            fig.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.show()
        
        return fields
    
    def analyze_spatial_stats(self, fields):
        """Analyze spatial statistics of generated fields"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # Spatial autocorrelation function
        def spatial_autocorrelation(field, max_lag=15):
            h, w = field.shape
            corr = np.zeros(max_lag)
            
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
        
        # Variogram function
        def variogram(field, max_lag=15):
            h, w = field.shape
            var = np.zeros(max_lag)
            
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
        
        # Calculate statistics for each field
        max_lag = 15
        lags = np.arange(1, max_lag+1)
        
        for i, (name, field) in enumerate(fields.items()):
            # Autocorrelation
            autocorr = spatial_autocorrelation(field, max_lag)
            axes[0].plot(lags, autocorr, 'o-', label=name)
            
            # Variogram
            var = variogram(field, max_lag)
            axes[1].plot(lags, var, 'o-', label=name)
            
            # Histogram
            axes[2].hist(field.ravel(), bins=50, density=True, alpha=0.5, label=name)
            
            # Power spectrum
            fft = np.fft.fft2(field)
            spectrum = np.abs(np.fft.fftshift(fft))
            axes[3].plot(np.log1p(spectrum[int(field.shape[0]/2)]), label=name)
        
        # Set plot properties
        axes[0].set_title('Spatial Autocorrelation')
        axes[0].set_xlabel('Lag Distance')
        axes[0].set_ylabel('Normalized Correlation')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].set_title('Variogram')
        axes[1].set_xlabel('Lag Distance')
        axes[1].set_ylabel('Semivariance')
        axes[1].legend()
        axes[1].grid(True)
        
        axes[2].set_title('Value Distribution')
        axes[2].set_xlabel('Field Value')
        axes[2].set_ylabel('Density')
        axes[2].legend()
        axes[2].grid(True)
        
        axes[3].set_title('Power Spectrum (Middle Row)')
        axes[3].set_xlabel('Frequency')
        axes[3].set_ylabel('Log Magnitude')
        axes[3].legend()
        axes[3].grid(True)
        
        plt.tight_layout()
        plt.show()

def generate_complex_field(grid_size=50):
    """Generate a complex reference spatial field"""
    # Create grid coordinates
    x = np.linspace(0, 10, grid_size)
    y = np.linspace(0, 10, grid_size)
    xx, yy = np.meshgrid(x, y)
    
    # Multi-scale features
    field = (
        np.sin(0.5 * xx) * np.cos(0.7 * yy) +  # Low-frequency pattern
        0.5 * np.sin(2.0 * xx) * np.cos(3.0 * yy) +  # Medium-frequency
        0.2 * np.random.randn(grid_size, grid_size)  # Noise
    )
    
    # Add directional trend
    field += 0.3 * (xx + yy) / 10
    
    # Add circular feature
    center_x, center_y = 7, 3
    radius = 2.5
    distance = np.sqrt((xx - center_x)**2 + (yy - center_y)**2)
    field += 0.8 * np.exp(-distance**2 / (2 * radius**2))
    
    return field

# Main execution
if __name__ == "__main__":
    # Configuration
    grid_size = 50
    n_samples = 200  # Number of training points
    
    # Generate complex reference field
    reference_field = generate_complex_field(grid_size)
    
    # Create training data
    coords = np.c_[np.random.uniform(0, 10, n_samples), 
                  np.random.uniform(0, 10, n_samples)]
    values = []
    for x, y in coords:
        # Interpolate from reference field
        xi = int(x / 10 * (grid_size - 1))
        yi = int(y / 10 * (grid_size - 1))
        values.append(reference_field[yi, xi])
    values = np.array(values)
    
    # Initialize and fit hybrid model
    hybrid_model = HybridSpatialModel(grid_size=grid_size, num_dgp_layers=3)
    hybrid_model.fit(coords, values, num_dgp_epochs=100, num_copula_fields=50)
    
    # Visualize copula dependence structure
    hybrid_model.copula.plot_dependence_structure()
    
    # Generate and compare fields
    generated_fields = hybrid_model.visualize_comparison(reference_field)
    
    # Analyze spatial statistics
    hybrid_model.analyze_spatial_stats({
        'Reference': reference_field,
        'DGP': generated_fields['dgp'],
        'Copula': generated_fields['copula'],
        'Hybrid': generated_fields['hybrid']
    })
    
    # Generate and visualize additional hybrid field
    hybrid_field = hybrid_model.generate_field('hybrid')
    hybrid_model.dgp.visualize_field(hybrid_field, "Hybrid DGP-Copula Spatial Field")
```

## Key Components of the Implementation

### 1. Deep Gaussian Processes (DGPs)
The DGP implementation creates a hierarchical model of Gaussian Processes:

```python
class DeepGaussianProcess:
    def __init__(self, num_layers=3, kernel='RBF', grid_size=50):
        # Initialization
        self.num_layers = num_layers
        self.layers = []
        
    class GPModel(gpytorch.models.ExactGP):
        # GPyTorch model definition
        def __init__(self, train_x, train_y, kernel, likelihood):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = kernel
            
    def fit(self, X, y, num_epochs=100, lr=0.1):
        # Training loop for each layer
        for i in range(self.num_layers):
            # Train GP layer
            # Propagate latent representation to next layer
```

Key features:
- Hierarchical structure with multiple GP layers
- Automatic Relevance Determination (ARD) kernels
- Latent space propagation between layers
- Efficient training with GPyTorch

### 2. Spatial Vine Copulas
The Vine Copula implementation models complex dependencies:

```python
class SpatialVineCopula:
    def __init__(self, grid_size=50):
        # Initialization
        self.copula = None
        self.marginals = []
        
    def fit(self, fields):
        # Transform to uniform marginals
        uniform_data = self.transformer.fit_transform(data_matrix)
        
        # Fit vine copula
        self.copula = pv.Vinecop(data=uniform_data, controls={
            'family_set': [pv.BicopFamily.gaussian, pv.BicopFamily.student, 
                          pv.BicopFamily.clayton, pv.BicopFamily.gumbel],
            'trunc_lvl': 3
        })
```

Key features:
- Quantile transformation to uniform marginals
- Flexible copula family selection
- Vine tree structure for high-dimensional dependence
- Truncation for model simplicity

### 3. Hybrid DGP-Copula Model
Combines the strengths of both approaches:

```python
class HybridSpatialModel:
    def __init__(self, grid_size=50, num_dgp_layers=3):
        self.dgp = DeepGaussianProcess()
        self.copula = SpatialVineCopula()
        
    def fit(self, X, y, num_dgp_epochs=100, num_copula_fields=50):
        # Fit DGP to observed data
        # Generate reference fields from DGP
        # Fit vine copula to generated fields
```

Key features:
- DGP captures large-scale spatial patterns
- Copula captures complex local dependencies
- Generation methods: DGP-only, Copula-only, Hybrid

### 4. Spatial Analysis Tools
Comprehensive spatial statistics calculation:

```python
def spatial_autocorrelation(field, max_lag=15):
    # Calculate spatial autocorrelation
    return corr / corr[0]

def variogram(field, max_lag=15):
    # Calculate variogram
    return var
```

Visualizations include:
- Spatial autocorrelation functions
- Variograms
- Value distributions
- Power spectra

## Key Mathematical Concepts

### 1. Deep Gaussian Processes
A hierarchy of Gaussian Processes:

$$
f^{(1)}(\mathbf{x}) \sim \mathcal{GP}(0, k_1(\mathbf{x}, \mathbf{x}')) \\
f^{(2)}(\mathbf{x}) \sim \mathcal{GP}(0, k_2(f^{(1)}(\mathbf{x}), f^{(1)}(\mathbf{x}')))) \\
\vdots \\
y(\mathbf{x}) \sim \mathcal{GP}(f^{(L)}(\mathbf{x}), \sigma_n^2\mathbf{I})
$$

Where each layer transforms the representation from the previous layer.

### 2. Vine Copulas
Decomposes multivariate dependence using bivariate copulas:

$$
c(\mathbf{u}) = \prod_{k=1}^{d-1} \prod_{e \in E_k} c_{j(e),k(e)|D(e)}(u_{j(e)|D(e)}, u_{k(e)|D(e)})
$$

Where:
- $c$ is the copula density
- $u$ are uniform marginals
- The product is over the vine tree structure

### 3. Spatial Statistics
**Spatial Autocorrelation**:
$$
\rho(h) = \frac{\mathbb{E}[(Z(s) - \mu)(Z(s+h) - \mu)]}{\sigma^2}
$$

**Variogram**:
$$
\gamma(h) = \frac{1}{2}\mathbb{E}[(Z(s) - Z(s+h))^2]
$$

## Execution Workflow

1. **Generate Complex Reference Field**:
   - Creates a multi-scale spatial pattern with:
     * Low-frequency sinusoidal patterns
     * Medium-frequency oscillations
     * Directional trend
     * Circular feature
     * Gaussian noise

2. **Train Hybrid Model**:
   - Fit DGP to sampled points
   - Generate multiple fields from DGP
   - Train vine copula on DGP-generated fields

3. **Generate and Compare Fields**:
   - Produce fields using:
     * DGP only
     * Copula only
     * Hybrid approach
   - Visual comparison against reference

4. **Statistical Analysis**:
   - Compare spatial autocorrelation
   - Analyze variograms
   - Examine value distributions
   - Compare power spectra

This implementation provides a powerful framework for modeling complex spatial fields, capturing both large-scale patterns through DGPs and complex local dependencies through vine copulas. The hybrid approach leverages the strengths of both methods to generate realistic spatial fields with accurate statistical properties.
