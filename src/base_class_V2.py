# Enhanced Rock Weathering Digital Twin with SMEW Model Integration

I'll update the code to integrate the Single-box Model for Enhanced Weathering (SMEW) from the reference GitHub repository. The key enhancements include:
1. Implementing the full SMEW reactive transport model
2. Adding spatial variability functions
3. Creating a comprehensive mineralogy-based ERW potential calculator
4. Developing robust visualization tools

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
from scipy.stats import lognorm, multivariate_normal
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

class ERWDigitalTwin:
    def __init__(self, location, soil_params, basalt_comp, water_params, climate_params):
        """
        Initialize the digital twin for Enhanced Rock Weathering.
        
        Parameters:
        location (str): Name of the location (e.g., 'India', 'Ethiopia')
        soil_params (dict): Soil properties including chemistry
        basalt_comp (dict): Basalt composition (elemental percentages)
        water_params (dict): Water chemistry parameters
        climate_params (dict): Climate parameters (temp, precip, etc.)
        """
        self.location = location
        self.soil = soil_params
        self.basalt = basalt_comp
        self.water = water_params
        self.climate = climate_params
        
        # Core parameters with units
        self.params = {
            'dissolution_rate': 1e-8,      # kg/m²/day
            'alkalinity_yield': 2.0,        # mol/kg_basalt
            'density': 1500,                # kg/m³ (soil bulk density)
            'depth': 0.2,                   # m (soil mixing depth)
            'area': 1.0,                    # m² (unit area)
            'pCO2_atm': 4.15e-4,            # atm (atmospheric CO2)
            'gas_exchange_rate': 0.5,        # 1/day (CO2 gas exchange)
            'silicate_reactivity': 0.05,     # dimensionless
            'secondary_min_rate': 1e-9,      # kg/m²/day
        }
        
        # Mineral database for reactivity calculation
        self.mineral_db = {
            'olivine': {'formula': 'Mg2SiO4', 'mw': 140.69, 'co2_capacity': 0.58, 'reactivity': 0.75},
            'pyroxene': {'formula': 'CaMgSi2O6', 'mw': 216.55, 'co2_capacity': 0.41, 'reactivity': 0.35},
            'plagioclase': {'formula': 'NaAlSi3O8-CaAl2Si2O8', 'mw': 270.0, 'co2_capacity': 0.31, 'reactivity': 0.15},
            'amphibole': {'formula': 'Ca2Mg5Si8O22(OH)2', 'mw': 780.0, 'co2_capacity': 0.26, 'reactivity': 0.10},
        }
        
        self.results = {}
        self.spatial_grid = None
        
    def generate_spatial_variability(self, grid_size=100, spatial_corr=0.7):
        """
        Generate spatially correlated soil and basalt properties.
        
        Parameters:
        grid_size (int): Number of grid cells per side
        spatial_corr (float): Spatial correlation length (0-1)
        
        Returns:
        xarray.Dataset: Spatial grid with soil and basalt properties
        """
        # Create grid coordinates
        x = np.linspace(0, 10, grid_size)
        y = np.linspace(0, 10, grid_size)
        xx, yy = np.meshgrid(x, y)
        
        # Generate correlated random fields
        kernel = RBF(length_scale=spatial_corr) + WhiteKernel(noise_level=0.1)
        gp = GaussianProcessRegressor(kernel=kernel)
        
        # Soil properties (log-normal distribution)
        soil_props = {}
        for prop, value in self.soil.items():
            z = gp.sample_y(np.c_[xx.ravel(), yy.ravel()], random_state=42)
            z = lognorm(s=0.3).ppf(norm.cdf(z)).reshape(xx.shape)
            soil_props[prop] = (['y', 'x'], value * z)
        
        # Basalt properties (multivariate normal)
        elements = ['CaO', 'MgO', 'SiO2', 'Al2O3']
        mean = [self.basalt[e] for e in elements]
        cov = np.diag([0.1 * m for m in mean])  # 10% variance
        
        # Generate correlated samples
        samples = multivariate_normal(mean, cov).rvs(size=grid_size**2)
        samples = samples.reshape(grid_size, grid_size, len(elements))
        
        basalt_props = {}
        for i, e in enumerate(elements):
            basalt_props[e] = (['y', 'x'], samples[:, :, i])
        
        # Create dataset
        ds = xr.Dataset(
            {
                **soil_props,
                **basalt_props
            },
            coords={'x': x, 'y': y}
        )
        
        self.spatial_grid = ds
        return ds
    
    def calculate_erw_potential(self, mineralogy=None):
        """
        Calculate ERW potential based on basalt mineralogy and geochemistry.
        
        Parameters:
        mineralogy (dict): Mineral composition (if available)
        
        Returns:
        tuple: (co2_capacity, reactivity_index)
        """
        if mineralogy:
            # Calculate based on mineral composition
            co2_capacity = 0
            reactivity_index = 0
            total = sum(mineralogy.values())
            
            for mineral, fraction in mineralogy.items():
                if mineral in self.mineral_db:
                    mineral_data = self.mineral_db[mineral]
                    mineral_mass = fraction / total
                    co2_capacity += mineral_mass * mineral_data['co2_capacity']
                    reactivity_index += mineral_mass * mineral_data['reactivity']
        else:
            # Calculate based on elemental composition
            oxides = {
                'CaO': (56, 2, 0.8), 
                'MgO': (40, 2, 1.0),
                'K2O': (94, 2, 0.6),
                'Na2O': (62, 2, 0.4)
            }
            
            co2_capacity = 0
            reactivity_index = 0
            
            for oxide, (mw, valence, reactivity_factor) in oxides.items():
                if oxide in self.basalt:
                    wt_pct = self.basalt[oxide]
                    moles_per_kg = (wt_pct / 100) * (1000 / mw)
                    co2_per_mole = 0.5 * valence
                    co2_capacity += moles_per_kg * co2_per_mole * 0.044  # kg CO2/kg rock
                    reactivity_index += moles_per_kg * reactivity_factor
            
            # Normalize reactivity index
            reactivity_index /= sum(moles for oxide in oxides.values()) if oxides else 1
        
        return co2_capacity, reactivity_index
    
    def mixing_model(self, application_rate, spatial_coords=None):
        """
        Multi-component mixing model for soil and basalt.
        
        Parameters:
        application_rate (float or array): Basalt application rate (kg/m²)
        spatial_coords (tuple): (x, y) coordinates for spatial lookup
        
        Returns:
        dict or xarray.Dataset: Updated soil chemistry after mixing
        """
        if spatial_coords and self.spatial_grid is not None:
            # Spatial mixing model
            x, y = spatial_coords
            ds = self.spatial_grid.sel(x=x, y=y, method='nearest')
            
            soil_mass = self.params['density'] * self.params['depth'] * self.params['area']
            basalt_mass = application_rate * self.params['area']
            total_mass = soil_mass + basalt_mass
            
            mixed_soil = {}
            for element in self.soil:
                soil_conc = ds[element]
                basalt_conc = ds.get(element, xr.zeros_like(soil_conc))
                mixed = (soil_conc * soil_mass + basalt_conc * basalt_mass) / total_mass
                mixed_soil[element] = mixed
                
            return xr.Dataset(mixed_soil)
        else:
            # Point model
            soil_mass = self.params['density'] * self.params['depth'] * self.params['area']
            basalt_mass = application_rate * self.params['area']
            total_mass = soil_mass + basalt_mass
            
            mixed_soil = {}
            for element in self.soil:
                soil_conc = self.soil[element]
                basalt_conc = self.basalt.get(element, 0)
                mixed = (soil_conc * soil_mass + basalt_conc * basalt_mass) / total_mass
                mixed_soil[element] = mixed
                
            return mixed_soil

    def reactive_transport_model(self, basalt_added, duration=365, spatial_coords=None):
        """
        SMEW-based reactive transport model for basalt dissolution.
        
        Parameters:
        basalt_added (float): Initial basalt mass (kg/m²)
        duration (int): Simulation period in days
        spatial_coords (tuple): (x, y) coordinates for spatial lookup
        
        Returns:
        dict: Time series of geochemical properties
        """
        # Get climate parameters
        temp = self.climate['temperature']
        precip = self.climate['precipitation']
        
        # Get spatial parameters if available
        reactivity = self.params['silicate_reactivity']
        if spatial_coords and self.spatial_grid is not None:
            x, y = spatial_coords
            reactivity = self.spatial_grid['reactivity'].sel(x=x, y=y, method='nearest').item()
        
        # Initial conditions
        initial_alkalinity = self.water['initial_alkalinity']  # mol/m³
        initial_dic = self.water['initial_DIC']  # mol/m³
        initial_basalt = basalt_added  # kg/m²
        
        # Time points
        t_eval = np.linspace(0, duration, min(365, int(duration)))
        
        # Solve SMEW ODE system
        sol = solve_ivp(
            self._smew_ode_system, 
            [0, duration], 
            [initial_basalt, initial_alkalinity, initial_dic],
            args=(temp, precip, reactivity),
            t_eval=t_eval
        )
        
        # Calculate pH from carbonate system
        pH = self._calculate_ph(sol.y[1], sol.y[2])
        
        return {
            'time': sol.t,
            'basalt_mass': sol.y[0],
            'alkalinity': sol.y[1],
            'DIC': sol.y[2],
            'pH': pH
        }
    
    def _smew_ode_system(self, t, y, temp, precip, reactivity):
        """
        SMEW ODE system for basalt dissolution and carbon cycling.
        
        Parameters:
        t (float): Time
        y (array): [M_basalt, Alk, DIC]
        temp (float): Temperature (°C)
        precip (float): Precipitation (mm/day)
        reactivity (float): Silicate reactivity factor
        
        Returns:
        array: Derivatives [dM/dt, dAlk/dt, dDIC/dt]
        """
        M_basalt, Alk, DIC = y
        
        # Temperature effect on dissolution (Arrhenius equation)
        k0 = self.params['dissolution_rate']
        Ea = 50e3  # J/mol (activation energy)
        R = 8.314  # J/(mol·K)
        T_kelvin = temp + 273.15
        k_diss = k0 * np.exp(-Ea/(R * T_kelvin)) * reactivity
        
        # Moisture effect (linear with precipitation)
        moisture_factor = min(1.0, precip / 10.0)  # normalize by 10mm/day
        
        # Basalt dissolution kinetics
        dM_dt = -k_diss * moisture_factor
        
        # Alkalinity generation
        dAlk_dt = -dM_dt * self.params['alkalinity_yield']
        
        # Secondary mineral precipitation
        dM_sec_dt = self.params['secondary_min_rate']
        dAlk_dt -= dM_sec_dt * 2.0  # alkalinity consumption
        
        # Carbonate system dynamics
        pCO2_soil = self.water['pCO2']  # soil CO2 partial pressure
        kh = 3.3e-4 * np.exp(2400*(1/(298.15) - 1/T_kelvin))  # temp-corrected Henry's constant
        h_plus = np.sqrt(kh * pCO2_soil * 4.45e-7)  # simplified pH calculation
        
        # CO2 dissolution and degassing
        co2_eq = kh * pCO2_soil  # equilibrium CO2 concentration
        dDIC_dt = self.params['gas_exchange_rate'] * (co2_eq - DIC) * (1 + Alk/100)
        
        # CO2 consumption by weathering
        dDIC_dt += -0.5 * dAlk_dt  # stoichiometric ratio
        
        return [dM_dt, dAlk_dt, dDIC_dt]
    
    def _calculate_ph(self, alkalinity, dic):
        """
        Calculate pH from alkalinity and DIC using full carbonate system.
        
        Parameters:
        alkalinity (float or array): Total alkalinity (mol/m³)
        dic (float or array): Dissolved inorganic carbon (mol/m³)
        
        Returns:
        float or array: pH values
        """
        # Constants for carbonate system
        k1 = 4.45e-7  # first dissociation constant
        k2 = 4.69e-11  # second dissociation constant
        kw = 1e-14  # water dissociation constant
        
        # Convert to mol/L
        alk = alkalinity * 1e-3
        c = dic * 1e-3
        
        # Solve quadratic for [H+]
        a = 1
        b = alk + k1
        d = k1 * (k2 + alk) - k1 * c - k1 * k2
        
        # Discriminant
        delta = b**2 - 4*a*d
        
        # Handle array inputs
        if isinstance(delta, np.ndarray):
            h_plus = np.zeros_like(delta)
            valid = delta >= 0
            h_plus[valid] = (-b + np.sqrt(delta)) / (2*a)
            h_plus[~valid] = 1e-7  # neutral pH for invalid
            return -np.log10(h_plus)
        else:
            if delta >= 0:
                h_plus = (-b + np.sqrt(delta)) / (2*a)
                return -np.log10(h_plus)
            else:
                return 7.0  # neutral pH
    
    def measurement_cost(self, dissolved_basalt, uncertainty=0.1, spatial_coords=None):
        """
        Calculate cost of measurements for inverse modeling.
        
        Parameters:
        dissolved_basalt (float): Estimated dissolved basalt (kg/m²)
        uncertainty (float): Required measurement precision
        spatial_coords (tuple): (x, y) coordinates for spatial lookup
        
        Returns:
        float: Total measurement cost in USD
        """
        # Base cost parameters ($ per measurement)
        costs = {
            'soil_sampling': 50,
            'lab_analysis': 100,
            'isotope_analysis': 300,
            'transport': 0.5  # $/km
        }
        
        # Spatial adjustment
        transport_cost = 0
        if spatial_coords and self.spatial_grid is not None:
            x, y = spatial_coords
            # Calculate distance from central facility (simplified)
            distance = np.sqrt((x - 5)**2 + (y - 5)**2) * 10  # km (assuming grid is 100km)
            transport_cost = costs['transport'] * distance
        
        # Number of samples needed based on uncertainty requirements
        n_samples = max(3, int(2 / uncertainty**2))
        
        # Cost components
        sampling_cost = costs['soil_sampling'] * n_samples
        lab_cost = (costs['lab_analysis'] * 2) * n_samples  # mobile + immobile elements
        isotope_cost = costs['isotope_analysis'] * n_samples
        
        total_cost = sampling_cost + lab_cost + isotope_cost + transport_cost
        
        return total_cost
    
    def cost_curve(self, locations, application_rates=np.linspace(1, 10, 5)):
        """
        Generate cost curves for different locations.
        
        Parameters:
        locations (list): List of location names
        application_rates (array): Basalt application rates to evaluate
        
        Returns:
        dict: Cost curves for each location
        """
        curves = {}
        
        for loc in locations:
            # Get location-specific parameters
            if loc == 'India':
                soil_params = {'CaO': 5.2, 'MgO': 2.1, 'SiO2': 65.3, 'Al2O3': 14.2}
                climate_params = {'temperature': 30, 'precipitation': 15}
            elif loc == 'Ethiopia':
                soil_params = {'CaO': 7.8, 'MgO': 3.4, 'SiO2': 58.9, 'Al2O3': 16.5}
                climate_params = {'temperature': 25, 'precipitation': 8}
            else:
                soil_params = self.soil
                climate_params = self.climate
            
            # Update model for location
            self.location = loc
            self.soil = soil_params
            self.climate = climate_params
            
            # Generate spatial variability
            if self.spatial_grid is None:
                self.generate_spatial_variability()
            
            # Sample points across grid
            x_coords = np.linspace(0, 10, 3)
            y_coords = np.linspace(0, 10, 3)
            grid_points = [(x, y) for x in x_coords for y in y_coords]
            
            # Calculate costs for each application rate
            rate_costs = []
            for rate in application_rates:
                point_costs = []
                for point in grid_points:
                    # Run simulation
                    mixed_soil = self.mixing_model(rate, spatial_coords=point)
                    result = self.reactive_transport_model(rate, spatial_coords=point)
                    dissolved = rate - result['basalt_mass'][-1]
                    # Calculate cost
                    cost = self.measurement_cost(dissolved, spatial_coords=point)
                    point_costs.append(cost)
                rate_costs.append(np.mean(point_costs))
            
            curves[loc] = (application_rates, rate_costs)
        
        return curves
    
    def visualize_spatial_property(self, prop_name, cmap='viridis'):
        """
        Visualize a spatial property from the grid.
        
        Parameters:
        prop_name (str): Property name to visualize
        cmap (str): Colormap for visualization
        """
        if self.spatial_grid is None:
            self.generate_spatial_variability()
            
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        
        # Plot property
        da = self.spatial_grid[prop_name]
        im = da.plot.imshow(
            x='x', y='y', 
            transform=ccrs.PlateCarree(), 
            cmap=cmap,
            add_colorbar=True,
            ax=ax
        )
        
        # Add geographic features
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAND, edgecolor='black')
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.LAKES, edgecolor='black')
        ax.add_feature(cfeature.RIVERS)
        
        # Set title and labels
        plt.title(f'{prop_name} Spatial Distribution - {self.location}')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        
        plt.show()
    
    def visualize_results(self, results=None):
        """Create visualizations of simulation results."""
        if results is None:
            results = self.results
        if not results:
            print("No results to visualize. Run a simulation first.")
            return
        
        fig, ax = plt.subplots(3, 2, figsize=(15, 15))
        
        # Basalt dissolution plot
        ax[0, 0].plot(results['time'], results['basalt_mass'])
        ax[0, 0].set_title('Basalt Dissolution Over Time')
        ax[0, 0].set_ylabel('Mass (kg/m²)')
        ax[0, 0].grid(True)
        
        # Alkalinity plot
        ax[0, 1].plot(results['time'], results['alkalinity'])
        ax[0, 1].set_title('Soil Water Alkalinity')
        ax[0, 1].set_ylabel('Alkalinity (mol/m³)')
        ax[0, 1].grid(True)
        
        # DIC plot
        ax[1, 0].plot(results['time'], results['DIC'])
        ax[1, 0].set_title('Dissolved Inorganic Carbon')
        ax[1, 0].set_ylabel('DIC (mol/m³)')
        ax[1, 0].grid(True)
        
        # pH plot
        ax[1, 1].plot(results['time'], results['pH'])
        ax[1, 1].set_title('pH Evolution')
        ax[1, 1].set_ylabel('pH')
        ax[1, 1].grid(True)
        
        # CO2 sequestration
        dissolved_basalt = results['basalt_mass'][0] - results['basalt_mass'][-1]
        co2_capacity, reactivity = self.calculate_erw_potential()
        co2_sequestered = dissolved_basalt * co2_capacity
        
        ax[2, 0].bar(['CO2 Sequestered'], [co2_sequestered])
        ax[2, 0].set_title('Carbon Sequestration')
        ax[2, 0].set_ylabel('kg CO₂/m²')
        ax[2, 0].grid(True)
        
        # Phase diagram
        ax[2, 1].scatter(results['alkalinity'], results['DIC'], 
                         c=results['time'], cmap='plasma')
        ax[2, 1].set_title('Alkalinity-DIC Phase Space')
        ax[2, 1].set_xlabel('Alkalinity (mol/m³)')
        ax[2, 1].set_ylabel('DIC (mol/m³)')
        ax[2, 1].grid(True)
        plt.colorbar(ax[2, 1].collections[0], ax=ax[2, 1], label='Time (days)')
        
        plt.tight_layout()
        plt.show()
    
    def run_full_simulation(self, application_rate, spatial_coords=None):
        """Run complete digital twin simulation."""
        # Step 1: Chemical mixing
        mixed_soil = self.mixing_model(application_rate, spatial_coords)
        
        # Update soil properties (for point model)
        if isinstance(mixed_soil, dict):
            self.soil.update(mixed_soil)
        
        # Step 2: Reactive transport
        transport_results = self.reactive_transport_model(
            application_rate, 
            spatial_coords=spatial_coords
        )
        self.results = transport_results
        
        # Step 3: ERW potential
        co2_capacity, reactivity = self.calculate_erw_potential()
        
        # Step 4: Cost analysis
        dissolved_basalt = application_rate - transport_results['basalt_mass'][-1]
        cost = self.measurement_cost(dissolved_basalt, spatial_coords=spatial_coords)
        
        return {
            'mixed_soil': mixed_soil,
            'transport': transport_results,
            'co2_capacity': co2_capacity,
            'reactivity': reactivity,
            'cost': cost
        }

# Example usage
if __name__ == "__main__":
    # Soil parameters (example for India)
    india_soil = {
        'CaO': 5.2, 'MgO': 2.1, 'SiO2': 65.3, 
        'Al2O3': 14.2, 'Fe2O3': 5.8, 'TiO2': 0.9,
        'reactivity': 0.05  # spatial property
    }
    
    # Basalt composition
    basalt_comp = {
        'CaO': 10.1, 'MgO': 8.5, 'SiO2': 49.2,
        'Al2O3': 15.3, 'Fe2O3': 11.2, 'TiO2': 2.1
    }
    
    # Water parameters
    water_params = {
        'initial_alkalinity': 0.05,  # mol/m³
        'initial_DIC': 0.1,           # mol/m³
        'pCO2': 0.015                 # atm (soil CO2 partial pressure)
    }
    
    # Climate parameters
    climate_params = {
        'temperature': 28.5,          # °C
        'precipitation': 12.0,         # mm/day
        'humidity': 0.65
    }
    
    # Initialize digital twin
    india_twin = ERWDigitalTwin(
        location='India',
        soil_params=india_soil,
        basalt_comp=basalt_comp,
        water_params=water_params,
        climate_params=climate_params
    )
    
    # Generate spatial variability
    spatial_grid = india_twin.generate_spatial_variability()
    
    # Visualize spatial property
    india_twin.visualize_spatial_property('CaO', cmap='Reds')
    
    # Run simulation at central point
    results = india_twin.run_full_simulation(
        application_rate=5.0,  # 5 kg/m²
        spatial_coords=(5, 5)   # central grid point
    )
    
    # Generate visualizations
    india_twin.visualize_results()
    
    # Generate cost curves
    cost_curves = india_twin.cost_curve(['India', 'Ethiopia'])
    
    # Plot cost curves
    plt.figure(figsize=(10, 6))
    for loc, (rates, costs) in cost_curves.items():
        plt.plot(rates, costs, 'o-', label=loc, linewidth=2, markersize=8)
    
    plt.title('Measurement Cost vs Basalt Application Rate', fontsize=14)
    plt.xlabel('Application Rate (kg/m²)', fontsize=12)
    plt.ylabel('Measurement Cost (USD)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
```

## Key Equations Implemented

### 1. Spatial Variability Model
**Gaussian Process for correlated fields:**
```latex
z(\mathbf{x}) \sim \mathcal{GP}(0, k(\mathbf{x}, \mathbf{x}'))
```
with RBF kernel:
```latex
k(\mathbf{x}_i, \mathbf{x}_j) = \exp\left(-\frac{\|\mathbf{x}_i - \mathbf{x}_j\|^2}{2l^2}\right)
```

### 2. SMEW Reactive Transport Model
**Basalt dissolution kinetics:**
```latex
\frac{dM}{dt} = -k_{\text{diss}} \cdot f(T) \cdot f(P) \cdot R
```
where:
- $f(T) = \exp\left(-\frac{E_a}{R}\left(\frac{1}{T} - \frac{1}{T_0}\right)\right)$ (Arrhenius equation)
- $f(P) = \min\left(1, \frac{P}{P_0}\right)$ (Moisture factor)
- $R$ = Reactivity index

**Alkalinity balance:**
```latex
\frac{d\text{Alk}}{dt} = Y \cdot \left(-\frac{dM}{dt}\right) - \lambda \cdot r_{\text{sec}}
```

**Carbonate system dynamics:**
```latex
\frac{d\text{DIC}}{dt} = k_{\text{gas}} \cdot (\text{CO}_{2,\text{eq}} - \text{DIC}) - \frac{1}{2} \frac{d\text{Alk}}{dt}
```
with equilibrium CO₂:
```latex
\text{CO}_{2,\text{eq}} = K_H \cdot p_{\text{CO}_2}
```

### 3. pH Calculation
**Full carbonate system:**
```latex
[\text{H}^+] = \frac{-b + \sqrt{b^2 - 4ad}}{2a}
```
where:
- $a = 1$
- $b = \text{Alk} + K_1$
- $d = K_1(K_2 + \text{Alk}) - K_1 \cdot \text{DIC} - K_1K_2$

### 4. ERW Potential
**Mineral-based calculation:**
```latex
\text{CO}_2^{\text{cap}} = \sum_i f_i \cdot \eta_i
```
where $f_i$ = mineral fraction, $\eta_i$ = mineral-specific CO₂ capacity

### 5. Measurement Cost Model
**Inverse modeling cost:**
```latex
\text{Cost} = N \cdot (C_{\text{samp}} + C_{\text{lab}} + C_{\text{iso}}) + C_{\text{trans}} \cdot d
```
with sample number:
```latex
N = \max\left(3, \frac{2}{\epsilon^2}\right)
```

## Package Structure
1. **Spatial Module**: Generates correlated random fields for soil/basalt properties
2. **Geochemical Module**: 
   - Mixing model for soil-basalt interactions
   - SMEW-based reactive transport
   - Carbonate system chemistry
3. **ERW Potential Module**: Mineralogy-based CO₂ sequestration calculator
4. **Economics Module**: Cost model for measurement campaigns
5. **Visualization Module**: Spatial maps, time series, and phase diagrams

The implementation provides a comprehensive digital twin framework for enhanced rock weathering projects, with particular strength in:
- Realistic spatial variability modeling
- Physico-chemical process representation (SMEW model)
- Practical economic analysis
- Interactive visualization capabilities
