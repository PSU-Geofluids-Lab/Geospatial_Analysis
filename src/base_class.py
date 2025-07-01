import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

class ERWDigitalTwin:
    def __init__(self, location, soil_params, basalt_comp, water_params):
        """
        Initialize the digital twin for Enhanced Rock Weathering.
        
        Parameters:
        location (str): Name of the location (e.g., 'India', 'Ethiopia')
        soil_params (dict): Soil properties including chemistry
        basalt_comp (dict): Basalt composition (elemental percentages)
        water_params (dict): Water chemistry parameters
        """
        self.location = location
        self.soil = soil_params
        self.basalt = basalt_comp
        self.water = water_params
        self.params = {
            'dissolution_rate': 1e-8,  # kg/m²/day
            'alkalinity_yield': 2.0,    # mol/kg_basalt
            'density': 1500,            # kg/m³ (soil bulk density)
            'depth': 0.2,               # m (soil mixing depth)
            'area': 1.0,                # m² (unit area)
        }
        self.results = {}
        
    def calculate_erw_potential(self):
        """
        Calculate ERW potential based on basalt mineralogy.
        Uses cation equivalents to estimate CO2 sequestration potential.
        """
        # Molecular weights and valence factors
        oxides = {
            'CaO': (56, 2), 'MgO': (40, 2), 
            'K2O': (94, 2), 'Na2O': (62, 2)
        }
        
        co2_capacity = 0
        for oxide, (mw, valence) in oxides.items():
            if oxide in self.basalt:
                wt_pct = self.basalt[oxide]
                # Convert wt% to mol/kg and calculate CO2 capture potential
                moles_per_kg = (wt_pct / 100) * (1000 / mw)
                co2_per_mole = 0.5 * valence  # 0.5 mol CO2 per equivalent
                co2_capacity += moles_per_kg * co2_per_mole * 0.044  # kg CO2/kg rock
        
        return co2_capacity
    
    def mixing_model(self, application_rate):
        """
        Multi-component mixing model for soil and basalt.
        
        Parameters:
        application_rate (float): Basalt application rate (kg/m²)
        
        Returns:
        dict: Updated soil chemistry after mixing
        """
        # Calculate mass ratios
        soil_mass = self.params['density'] * self.params['depth'] * self.params['area']
        basalt_mass = application_rate * self.params['area']
        total_mass = soil_mass + basalt_mass
        
        # Update element concentrations
        mixed_soil = {}
        for element in self.soil:
            soil_conc = self.soil[element]
            basalt_conc = self.basalt.get(element, 0)
            mixed = (soil_conc * soil_mass + basalt_conc * basalt_mass) / total_mass
            mixed_soil[element] = mixed
        
        return mixed_soil

    def reactive_transport_model(self, basalt_added, duration=365):
        """
        Single-box reactive transport model for basalt dissolution.
        
        Parameters:
        basalt_added (float): Initial basalt mass (kg/m²)
        duration (int): Simulation period in days
        
        Returns:
        dict: Time series of pH and alkalinity
        """
        # Initial conditions
        initial_alkalinity = self.water['initial_alkalinity']  # mol/m³
        initial_basalt = basalt_added  # kg/m²
        
        def model(t, y):
            M, Alk = y
            # Dissolution kinetics
            dMdt = -self.params['dissolution_rate']
            # Alkalinity generation
            dAlkdt = -dMdt * self.params['alkalinity_yield']
            return [dMdt, dAlkdt]
        
        # Solve ODE
        sol = solve_ivp(model, [0, duration], 
                        [initial_basalt, initial_alkalinity],
                        t_eval=np.linspace(0, duration, 100))
        
        # Calculate pH from alkalinity
        pH = self.calculate_ph(sol.y[1])
        
        return {
            'time': sol.t,
            'basalt_mass': sol.y[0],
            'alkalinity': sol.y[1],
            'pH': pH
        }
    
    def calculate_ph(self, alkalinity):
        """
        Calculate pH from alkalinity using simplified carbonate chemistry.
        """
        # Simplified model assuming constant pCO2
        pCO2 = self.water['pCO2']  # atm
        kh = 3.3e-4                # mol/L/atm (Henry's constant)
        k1 = 4.45e-7               # First dissociation constant
        
        h_plus = np.sqrt(kh * pCO2 * k1 / (alkalinity * 1e-3))
        return -np.log10(h_plus)
    
    def measurement_cost(self, dissolved_basalt, uncertainty=0.1):
        """
        Calculate cost of measurements for inverse modeling.
        
        Parameters:
        dissolved_basalt (float): Estimated dissolved basalt (kg/m²)
        uncertainty (float): Required measurement precision
        
        Returns:
        float: Total measurement cost in USD
        """
        # Cost parameters ($ per measurement)
        costs = {
            'soil_sampling': 50,
            'lab_analysis': 100,
            'isotope_analysis': 300
        }
        
        # Number of samples needed based on uncertainty requirements
        n_samples = int(10 / uncertainty)
        total_cost = (costs['soil_sampling'] + 
                      costs['lab_analysis'] * 2 + 
                      costs['isotope_analysis']) * n_samples
        
        return total_cost
    
    def cost_curve(self, locations):
        """
        Generate cost curves for different locations.
        
        Parameters:
        locations (list): List of location names
        
        Returns:
        dict: Cost curves for each location
        """
        curves = {}
        for loc in locations:
            # Simulate different application rates
            application_rates = np.linspace(1, 10, 5)
            costs = []
            for rate in application_rates:
                # Run simulation
                mixed_soil = self.mixing_model(rate)
                result = self.reactive_transport_model(rate)
                dissolved = rate - result['basalt_mass'][-1]
                # Calculate cost
                cost = self.measurement_cost(dissolved)
                costs.append(cost)
            
            curves[loc] = (application_rates, costs)
        
        return curves
    
    def visualize_results(self):
        """Create visualizations of simulation results."""
        if not self.results:
            print("No results to visualize. Run a simulation first.")
            return
        
        fig, ax = plt.subplots(2, 2, figsize=(12, 10))
        
        # Basalt dissolution plot
        ax[0, 0].plot(self.results['time'], self.results['basalt_mass'])
        ax[0, 0].set_title('Basalt Dissolution Over Time')
        ax[0, 0].set_ylabel('Mass (kg/m²)')
        
        # Alkalinity plot
        ax[0, 1].plot(self.results['time'], self.results['alkalinity'])
        ax[0, 1].set_title('Soil Water Alkalinity')
        ax[0, 1].set_ylabel('Alkalinity (mol/m³)')
        
        # pH plot
        ax[1, 0].plot(self.results['time'], self.results['pH'])
        ax[1, 0].set_title('pH Evolution')
        ax[1, 0].set_xlabel('Time (days)')
        ax[1, 0].set_ylabel('pH')
        
        # ERW potential
        co2_capacity = self.calculate_erw_potential()
        ax[1, 1].bar(['CO2 Capacity'], [co2_capacity])
        ax[1, 1].set_title('Carbon Sequestration Potential')
        ax[1, 1].set_ylabel('kg CO₂/kg rock')
        
        plt.tight_layout()
        plt.show()
    
    def run_full_simulation(self, application_rate):
        """Run complete digital twin simulation."""
        # Step 1: Chemical mixing
        mixed_soil = self.mixing_model(application_rate)
        self.soil.update(mixed_soil)
        
        # Step 2: Reactive transport
        transport_results = self.reactive_transport_model(application_rate)
        self.results = transport_results
        
        # Step 3: ERW potential
        erw_potential = self.calculate_erw_potential()
        
        # Step 4: Cost analysis
        dissolved_basalt = application_rate - transport_results['basalt_mass'][-1]
        cost = self.measurement_cost(dissolved_basalt)
        
        return {
            'mixed_soil': mixed_soil,
            'transport': transport_results,
            'erw_potential': erw_potential,
            'cost': cost
        }

# Example usage
if __name__ == "__main__":
    # Soil parameters (example for India)
    india_soil = {
        'CaO': 5.2, 'MgO': 2.1, 'SiO2': 65.3, 
        'Al2O3': 14.2, 'Fe2O3': 5.8, 'TiO2': 0.9
    }
    
    # Basalt composition
    basalt_comp = {
        'CaO': 10.1, 'MgO': 8.5, 'SiO2': 49.2,
        'Al2O3': 15.3, 'Fe2O3': 11.2, 'TiO2': 2.1
    }
    
    # Water parameters
    water_params = {
        'initial_alkalinity': 0.05,  # mol/m³
        'pCO2': 0.015  # atm (soil CO2 partial pressure)
    }
    
    # Initialize digital twin
    india_twin = ERWDigitalTwin(
        location='India',
        soil_params=india_soil,
        basalt_comp=basalt_comp,
        water_params=water_params
    )
    
    # Run simulation
    results = india_twin.run_full_simulation(application_rate=5.0)  # 5 kg/m²
    
    # Generate visualizations
    india_twin.visualize_results()
    
    # Generate cost curves
    cost_curves = india_twin.cost_curve(['India', 'Ethiopia'])
    
    # Plot cost curves
    plt.figure(figsize=(10, 6))
    for loc, (rates, costs) in cost_curves.items():
        plt.plot(rates, costs, 'o-', label=loc)
    plt.title('Measurement Cost vs Basalt Application Rate')
    plt.xlabel('Application Rate (kg/m²)')
    plt.ylabel('Measurement Cost (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()
