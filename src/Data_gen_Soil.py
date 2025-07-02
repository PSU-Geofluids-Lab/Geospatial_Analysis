import numpy as np
import pandas as pd


def Generate_data(lat_range=[24.0,24.1],lon_range=[-125.0,-124.9]):
    num_samples = 1000
    
    # Generate synthetic spatial data
    lats = np.random.uniform(lat_range[0], lat_range[1], num_samples)
    lons = np.random.uniform(lon_range[0], lon_range[1], num_samples)
    
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

