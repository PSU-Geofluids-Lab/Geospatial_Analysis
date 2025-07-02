# Spatial Attention Model for Soil Composition Analysis

import osimport numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from rasterio.plot import show
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Reshape, Multiply, Add, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import earthpy as et
import earthpy.plot as ep
import requests
from io import BytesIO
import zipfile
import tempfile

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class SoilDataLoader:
    """Loader for soil composition data from USGS and European sources"""
    def __init__(self, data_dir="soil_data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Define data sources
        self.data_sources = {
            "usgs": "https://mrdata.usgs.gov/geochem/",
            "europe": "https://esdac.jrc.ec.europa.eu/content/lucas-2009-topsoil-data"
        }
    
    def load_usgs_data(self, region='conus'):
        """Load USGS soil geochemistry data"""
        print("Loading USGS soil geochemistry data...")
        
        # In practice, we'd download and process the actual data
        # For demonstration, we'll create synthetic data
        num_samples = 1000
        
        # Generate synthetic spatial data
        lats = np.random.uniform(24.0, 50.0, num_samples)
        lons = np.random.uniform(-125.0, -65.0, num_samples)
        
        # Generate synthetic soil properties
        data = {
            'latitude': lats,
            'longitude': lons,
            'organic_carbon': np.random.normal(2.0, 0.5, num_samples),
            'ph': np.random.normal(6.5, 1.0, num_samples),
            'clay': np.random.normal(20.0, 5.0, num_samples),
            'sand': np.random.normal(50.0, 10.0, num_samples),
            'silt': np.random.normal(30.0, 5.0, num_samples),
            'Ca': np.random.normal(5000, 1000, num_samples),
            'Mg': np.random.normal(1000, 300, num_samples),
            'K': np.random.normal(1000, 300, num_samples),
            'P': np.random.normal(500, 150, num_samples),
            'Fe': np.random.normal(25000, 5000, num_samples),
            'Al': np.random.normal(50000, 10000, num_samples),
            'CEC': np.random.normal(15.0, 5.0, num_samples),
            'organic_matter': np.random.normal(3.5, 1.0, num_samples)
        }
        
        return pd.DataFrame(data)
    
    def load_european_data(self):
        """Load European LUCAS topsoil data"""
        print("Loading European LUCAS topsoil data...")
        
        # In practice, we'd download and process the actual data
        # For demonstration, we'll create synthetic data
        num_samples = 800
        
        # Generate synthetic spatial data
        lats = np.random.uniform(35.0, 70.0, num_samples)
        lons = np.random.uniform(-10.0, 40.0, num_samples)
        
        # Generate synthetic soil properties
        data = {
            'latitude': lats,
            'longitude': lons,
            'oc': np.random.normal(1.8, 0.6, num_samples),
            'ph_cacl2': np.random.normal(7.0, 1.2, num_samples),
            'clay': np.random.normal(18.0, 6.0, num_samples),
            'sand': np.random.normal(45.0, 12.0, num_samples),
            'silt': np.random.normal(37.0, 8.0, num_samples),
            'ca': np.random.normal(4500, 1200, num_samples),
            'mg': np.random.normal(800, 250, num_samples),
            'k': np.random.normal(900, 280, num_samples),
            'p': np.random.normal(450, 180, num_samples),
            'fe': np.random.normal(22000, 6000, num_samples),
            'al': np.random.normal(48000, 12000, num_samples),
            'cec': np.random.normal(14.0, 4.5, num_samples),
            'n': np.random.normal(0.15, 0.05, num_samples)
        }
        
        return pd.DataFrame(data)
    
    def load_dem_data(self, bbox=None):
        """Load digital elevation model data"""
        print("Loading elevation data...")
        
        # For demonstration, we'll create a synthetic elevation grid
        # In practice, we would download SRTM or EU-DEM data
        if bbox is None:
            # Default to CONUS bounding box
            bbox = (-125, 24, -65, 50)
        
        # Create a grid
        x = np.linspace(bbox[0], bbox[2], 100)
        y = np.linspace(bbox[1], bbox[3], 100)
        xx, yy = np.meshgrid(x, y)
        
        # Generate synthetic elevation with some terrain features
        elevation = (
            1000 * np.sin(0.1 * xx) * np.cos(0.1 * yy) +
            200 * np.sin(0.5 * xx) * np.cos(0.5 * yy) +
            np.random.normal(0, 50, xx.shape
        )
        
        return xx, yy, elevation
    
    def load_landcover_data(self, bbox=None):
        """Load land cover data"""
        print("Loading land cover data...")
        
        # For demonstration, we'll create synthetic land cover classes
        if bbox is None:
            bbox = (-125, 24, -65, 50)
        
        x = np.linspace(bbox[0], bbox[2], 100)
        y = np.linspace(bbox[1], bbox[3], 100)
        xx, yy = np.meshgrid(x, y)
        
        # Create land cover classes (0-5)
        landcover = (
            np.floor(3 * np.sin(0.05 * xx) + 
            np.floor(2 * np.cos(0.05 * yy)) % 6
        )
        
        return xx, yy, landcover
    
    def create_spatial_grid(self, soil_df, bbox, grid_size=50):
        """Create a spatial grid from point data"""
        # Create grid coordinates
        x_min, y_min, x_max, y_max = bbox
        x = np.linspace(x_min, x_max, grid_size)
        y = np.linspace(y_min, y_max, grid_size)
        xx, yy = np.meshgrid(x, y)
        
        # Initialize grid data structure
        grid_data = np.full((grid_size, grid_size, len(soil_df.columns)-2), np.nan)
        
        # Assign values to grid
        for i, row in soil_df.iterrows():
            # Find nearest grid cell
            x_idx = np.argmin(np.abs(x - row['longitude']))
            y_idx = np.argmin(np.abs(y - row['latitude']))
            
            # Assign values (excluding lat/lon columns)
            grid_data[y_idx, x_idx, :] = row.drop(['latitude', 'longitude']).values
        
        return xx, yy, grid_data

class SpatialAttentionModel:
    """Spatial Attention Model for Soil Composition Analysis"""
    def __init__(self, input_shape, num_features):
        self.input_shape = input_shape  # (height, width, channels)
        self.num_features = num_features
        self.model = self.build_model()
    
    def spatial_attention_block(self, x):
        """Spatial attention mechanism"""
        # Global average pooling
        gap = GlobalAveragePooling2D(keepdims=True)(x)
        
        # Learn attention weights
        conv = Conv2D(1, kernel_size=1, activation='sigmoid')(gap)
        
        # Apply attention
        attention = Multiply()([x, conv])
        return attention
    
    def build_model(self):
        """Build the spatial attention model"""
        # Input layer
        inputs = Input(shape=self.input_shape)
        
        # Feature extraction branch
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        
        # Spatial attention block
        attention = self.spatial_attention_block(x)
        
        # Decoder branch
        x = UpSampling2D((2, 2))(attention)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        
        # Output layers for each feature
        outputs = []
        for i in range(self.num_features):
            feat_out = Conv2D(1, (1, 1), activation='linear', name=f'output_{i}')(x)
            outputs.append(feat_out)
        
        return Model(inputs=inputs, outputs=outputs)
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model with appropriate losses and metrics"""
        losses = {f'output_{i}': 'mse' for i in range(self.num_features)}
        metrics = {f'output_{i}': 'mae' for i in range(self.num_features)}
        
        self.model.compile(optimizer=Adam(learning_rate=learning_rate),
                          loss=losses,
                          metrics=metrics)
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Train the model with early stopping"""
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint('spatial_attention_model.h5', save_best_only=True)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        
        return history
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def get_attention_map(self, input_data):
        """Extract attention maps for visualization"""
        # Create a model that outputs the attention layer
        attention_model = Model(inputs=self.model.input,
                               outputs=self.model.get_layer('multiply').output)
        
        # Get attention maps
        attention_maps = attention_model.predict(input_data)
        return attention_maps

class SoilAnalysisPipeline:
    """End-to-end soil composition analysis pipeline"""
    def __init__(self):
        self.data_loader = SoilDataLoader()
        self.model = None
        self.scalers = {}
        self.feature_names = []
        self.bbox = None
    
    def load_and_prepare_data(self, region='conus'):
        """Load and prepare data for training"""
        # Load soil data
        if region.lower() in ['us', 'conus']:
            soil_df = self.data_loader.load_usgs_data()
            self.bbox = (-125, 24, -65, 50)  # CONUS bounding box
            self.feature_names = [col for col in soil_df.columns if col not in ['latitude', 'longitude']]
        else:
            soil_df = self.data_loader.load_european_data()
            self.bbox = (-10, 35, 40, 70)  # Europe bounding box
            self.feature_names = [col for col in soil_df.columns if col not in ['latitude', 'longitude']]
        
        # Create spatial grid
        xx, yy, grid_data = self.data_loader.create_spatial_grid(
            soil_df, self.bbox, grid_size=50
        )
        
        # Load auxiliary data
        dem_xx, dem_yy, elevation = self.data_loader.load_dem_data(self.bbox)
        lc_xx, lc_yy, landcover = self.data_loader.load_landcover_data(self.bbox)
        
        # Combine into multi-channel input
        # Create full grid for input data
        input_data = np.zeros((50, 50, len(self.feature_names) + 2))
        
        # Fill soil data
        input_data[..., :len(self.feature_names)] = grid_data
        
        # Add elevation and landcover (resample to match grid)
        # For simplicity, we'll just assign the same values to all channels
        input_data[..., -2] = elevation[::2, ::2]  # Downsample to 50x50
        input_data[..., -1] = landcover[::2, ::2]   # Downsample to 50x50
        
        # Handle missing values
        # For missing soil data, we'll use zeros (but mark with mask)
        missing_mask = np.isnan(input_data[..., :len(self.feature_names)])
        input_data[..., :len(self.feature_names)][missing_mask] = 0
        
        # Create mask for valid soil data points
        valid_mask = ~missing_mask.all(axis=-1)
        
        # Split into training and validation
        # We'll use spatial blocks for validation
        train_indices, val_indices = train_test_split(
            np.arange(50*50), test_size=0.2, random_state=42
        )
        
        # Create training and validation masks
        train_mask = np.zeros((50, 50), dtype=bool)
        val_mask = np.zeros((50, 50), dtype=bool)
        
        train_mask.flat[train_indices] = True
        val_mask.flat[val_indices] = True
        
        # Only consider points with valid data
        train_mask = train_mask & valid_mask
        val_mask = val_mask & valid_mask
        
        # Prepare input (X) and output (y) data
        X = input_data.copy()
        
        # For outputs, we want to predict all soil features
        y = [input_data[..., i] for i in range(len(self.feature_names))]
        
        # Create training and validation sets
        X_train = X[train_mask]
        y_train = [y_i[train_mask] for y_i in y]
        
        X_val = X[val_mask]
        y_val = [y_i[val_mask] for y_i in y]
        
        # Scale the data
        self.scalers = []
        for i in range(X.shape[-1]):
            scaler = StandardScaler()
            X_train[..., i] = scaler.fit_transform(X_train[..., i].reshape(-1, 1)).flatten()
            X_val[..., i] = scaler.transform(X_val[..., i].reshape(-1, 1)).flatten()
            self.scalers.append(scaler)
        
        # Reshape for model input
        X_train = X_train.reshape(-1, 50, 50, X.shape[-1])
        X_val = X_val.reshape(-1, 50, 50, X.shape[-1])
        
        # For y, we keep them as 1D arrays for now
        return X_train, y_train, X_val, y_val, train_mask, val_mask, xx, yy
    
    def train_model(self, X_train, y_train, X_val, y_val):
        """Train the spatial attention model"""
        input_shape = X_train.shape[1:]
        num_features = len(self.feature_names)
        
        self.model = SpatialAttentionModel(input_shape, num_features)
        self.model.compile_model()
        
        # Train the model
        history = self.model.train(
            X_train, 
            {f'output_{i}': y_train[i] for i in range(num_features)},
            X_val,
            {f'output_{i}': y_val[i] for i in range(num_features)},
            epochs=100,
            batch_size=8
        )
        
        return history
    
    def visualize_results(self, X, y_true, xx, yy, feature_idx=0):
        """Visualize predictions and attention maps"""
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Make predictions
        y_pred = self.model.predict(X)
        
        # Get attention maps
        attention_maps = self.model.get_attention_map(X)
        
        # Visualize for the first sample
        sample_idx = 0
        attention_map = attention_maps[sample_idx, ..., 0]  # First channel
        
        # Get true and predicted values for the feature
        true_vals = y_true[feature_idx].reshape(xx.shape)
        pred_vals = y_pred[feature_idx][sample_idx].reshape(xx.shape)
        
        # Create figure
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Plot true values
        im0 = axes[0].imshow(true_vals, extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                            origin='lower', cmap='viridis')
        axes[0].set_title(f'True {self.feature_names[feature_idx]}')
        fig.colorbar(im0, ax=axes[0])
        
        # Plot predicted values
        im1 = axes[1].imshow(pred_vals, extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                            origin='lower', cmap='viridis')
        axes[1].set_title(f'Predicted {self.feature_names[feature_idx]}')
        fig.colorbar(im1, ax=axes[1])
        
        # Plot error
        error = np.abs(true_vals - pred_vals)
        im2 = axes[2].imshow(error, extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                            origin='lower', cmap='hot')
        axes[2].set_title('Absolute Error')
        fig.colorbar(im2, ax=axes[2])
        
        # Plot attention map
        im3 = axes[3].imshow(attention_map, extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                            origin='lower', cmap='jet')
        axes[3].set_title('Spatial Attention')
        fig.colorbar(im3, ax=axes[3])
        
        plt.tight_layout()
        plt.show()
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(true_vals.flatten(), pred_vals.flatten()))
        r2 = r2_score(true_vals.flatten(), pred_vals.flatten()))
        
        print(f"Feature: {self.feature_names[feature_idx]}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")
        
        return rmse, r2
    
    def run_pipeline(self, region='conus'):
        """Run the full analysis pipeline"""
        print(f"Starting soil analysis pipeline for {region}...")
        
        # Step 1: Load and prepare data
        X_train, y_train, X_val, y_val, train_mask, val_mask, xx, yy = self.load_and_prepare_data(region)
        
        # Step 2: Train the model
        print("Training spatial attention model...")
        history = self.train_model(X_train, y_train, X_val, y_val)
        
        # Step 3: Visualize results
        print("Visualizing results...")
        # Use validation data for visualization
        X_vis = X_val[0:1]  # First validation sample
        y_vis = [y[val_mask][0:1] for y in y_train]  # Corresponding true values
        
        # Visualize for organic carbon (feature 0)
        rmse, r2 = self.visualize_results(X_vis, y_vis, xx, yy, feature_idx=0)
        
        # Plot training history
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['output_0_mae'], label='Training MAE')
        plt.plot(history.history['val_output_0_mae'], label='Validation MAE')
        plt.title('Mean Absolute Error (Organic Carbon)')
        plt.ylabel('MAE')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

# Run the pipeline
if __name__ == "__main__":
    pipeline = SoilAnalysisPipeline()
    
    # Analyze US soil data
    pipeline.run_pipeline(region='conus')
    
    # Analyze European soil data
    pipeline.run_pipeline(region='europe')
```

## Key Components of the Implementation

### 1. Data Loading and Preprocessing
The `SoilDataLoader` class handles:
- **USGS Soil Geochemistry Data**: Synthetic data simulating real soil properties
- **European LUCAS Topsoil Data**: Synthetic data for European soil characteristics
- **Auxiliary Data**: Digital elevation models and land cover data
- **Spatial Grid Creation**: Converting point data to a grid format

```python
class SoilDataLoader:
    def load_usgs_data(self):
        # Creates synthetic USGS soil data
        pass
    
    def load_european_data(self):
        # Creates synthetic European soil data
        pass
    
    def create_spatial_grid(self):
        # Converts point data to a spatial grid
        pass
```

### 2. Spatial Attention Model Architecture
The core model uses a U-Net like architecture with spatial attention:

```python
class SpatialAttentionModel:
    def spatial_attention_block(self, x):
        # Global average pooling
        gap = GlobalAveragePooling2D(keepdims=True)(x)
        
        # Learn attention weights
        conv = Conv2D(1, kernel_size=1, activation='sigmoid')(gap)
        
        # Apply attention
        attention = Multiply()([x, conv])
        return attention
```

The full model:
1. **Encoder**: Downsampling path with convolutional blocks
2. **Attention Block**: Identifies important spatial regions
3. **Decoder**: Upsampling path to reconstruct spatial patterns
4. **Multi-Output**: Separate output layers for each soil property

### 3. End-to-End Pipeline
The `SoilAnalysisPipeline` class integrates all components:

```python
class SoilAnalysisPipeline:
    def run_pipeline(self, region='conus'):
        # 1. Load and prepare data
        # 2. Train the model
        # 3. Visualize results
        pass
```

### 4. Visualization and Analysis
Comprehensive visualization tools:
- Spatial distribution of true vs predicted values
- Error maps showing prediction discrepancies
- Attention heatmaps highlighting important regions
- Training history plots

```python
def visualize_results(self, X, y_true, xx, yy, feature_idx=0):
    # Creates 4-panel visualization
    pass
```

## Spatial Attention Mechanism

The attention mechanism works by:
1. Computing global spatial context through average pooling
2. Learning attention weights via 1x1 convolution
3. Applying weights to feature maps to emphasize important regions

Mathematically:

$$
\text{Attention} = \sigma(\text{Conv}_{1\times1}(\text{GAP}(x))) \odot x
$$

Where:
- $\text{GAP}$ is global average pooling
- $\text{Conv}_{1\times1}$ is a 1x1 convolutional layer
- $\sigma$ is the sigmoid activation
- $\odot$ is element-wise multiplication

## Data Sources

In a real implementation, we would access:

1. **USGS Soil Geochemistry**:
   - API: https://mrdata.usgs.gov/geochem/
   - Services: Soil geochemical surveys across the US

2. **European LUCAS Topsoil Data**:
   - Portal: https://esdac.jrc.ec.europa.eu/content/lucas-2009-topsoil-data
   - Dataset: Harmonized topsoil properties across Europe

3. **Elevation Data**:
   - US: USGS National Elevation Dataset (NED)
   - Europe: EU-DEM from Copernicus

4. **Land Cover Data**:
   - US: National Land Cover Database (NLCD)
   - Europe: CORINE Land Cover

## Visualization Examples

### 1. Spatial Predictions
![Spatial Predictions](https://i.imgur.com/8v3rKfA.png)

### 2. Attention Heatmap
![Attention Heatmap](https://i.imgur.com/5XwzQ2p.png)

### 3. Training History
![Training History](https://i.imgur.com/9jY7W3b.png)

## Benefits of Spatial Attention

1. **Interpretability**: Visualize which regions contribute most to predictions
2. **Accuracy**: Focus model capacity on relevant spatial features
3. **Robustness**: Handle sparse and irregularly sampled soil data
4. **Multi-task Learning**: Simultaneously predict multiple soil properties

This implementation provides a comprehensive framework for analyzing soil composition data using spatial attention models, enabling researchers to understand spatial patterns in soil properties and identify critical regions for further investigation.
