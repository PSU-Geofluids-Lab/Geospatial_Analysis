Below are the key equations for the Enhanced Rock Weathering (ERW) digital twin model, formatted in LaTeX. The equations are grouped by component for clarity.

---

### 1. **ERW Potential Calculation**
**CO₂ sequestration capacity** based on basalt mineralogy:
```latex
\text{CO}_{2,\text{capacity}} = \sum_{\text{oxide}} \left( \frac{\text{wt\% oxide}}{100} \right) \times \frac{1000}{M_{\text{oxide}}} \times \left( \frac{v_{\text{oxide}}}{2} \right) \times 0.044
```
where:
- \(M_{\text{oxide}}\) = Molecular weight of oxide (g/mol)
- \(v_{\text{oxide}}\) = Valence factor (e.g., 2 for CaO, MgO; 4 for Na₂O, K₂O)
- \(0.044\) = Molar mass of CO₂ (kg/mol)

---

### 2. **Multi-Component Mixing Model**
**Updated soil chemistry** after basalt application:
```latex
C_{\text{mixed}, i = \frac{C_{\text{soil}, i \cdot M_{\text{soil}} + C_{\text{basalt}, i \cdot M_{\text{basalt}}}{M_{\text{soil}} + M_{\text{basalt}}}
```
where:
- \(C_{\text{soil}, i\) = Concentration of element \(i\) in soil (wt%)
- \(C_{\text{basalt}, i\) = Concentration of element \(i\) in basalt (wt%)
- \(M_{\text{soil}} = \rho_{\text{soil}} \times d \times A\)
- \(M_{\text{basalt}} = R_{\text{app}} \times A\)
- \(\rho_{\text{soil}}\) = Soil bulk density (kg/m³)
- \(d\) = Soil mixing depth (m)
- \(A\) = Area (m²)
- \(R_{\text{app}}\) = Application rate (kg/m²)

---

### 3. **Reactive Transport Model**
**Basalt dissolution kinetics**:
```latex
\frac{dM}{dt} = -k_{\text{diss}} \quad \text{(kg/m²/day)}
```
**Alkalinity generation**:
```latex
\frac{d\text{Alk}}{dt} = -\frac{dM}{dt} \times Y
```
where:
- \(M\) = Basalt mass per unit area (kg/m²)
- \(k_{\text{diss}}\) = Dissolution rate constant (kg/m²/day)
- \(\text{Alk}\) = Alkalinity (mol/m³)
- \(Y\) = Alkalinity yield (mol/kg_basalt)

**pH calculation** from alkalinity:
```latex
\text{pH} = -\log_{10}\left( \sqrt{\frac{K_h \cdot p_{\text{CO}_2} \cdot K_1}{\text{Alk} \times 10^{-3}}} \right)
```
where:
- \(K_h\) = Henry's constant (mol/L/atm)
- \(p_{\text{CO}_2}\) = CO₂ partial pressure (atm)
- \(K_1\) = First dissociation constant of carbonic acid

---

### 4. **Measurement Cost Model**
**Number of samples** for uncertainty constraint:
```latex
N_{\text{samples}} = \left\lceil \frac{10}{\epsilon} \right\rceil
```
**Total cost**:
```latex
\text{Cost}_{\text{total}} = \left( C_{\text{samp}} + 2C_{\text{lab}} + C_{\text{iso}} \right) \times N_{\text{samples}}
```
where:
- \(\epsilon\) = Target uncertainty
- \(C_{\text{samp}}\) = Soil sampling cost (\$)
- \(C_{\text{lab}}\) = Lab analysis cost (\$)
- \(C_{\text{iso}}\) = Isotope analysis cost (\$)

---

### 5. **Cost Curve Generation**
**Dissolved basalt mass**:
```latex
M_{\text{dissolved}} = R_{\text{app}} - M(t_{\text{end}})
```
**Cost curve** for location \(L\):
```latex
\text{Cost}_L(R_{\text{app}}) = f\left( M_{\text{dissolved}}(R_{\text{app}}), \epsilon \right)
```

---

### 6. **Spatial Variability Model**
**Soil/basalt chemistry** represented as:
```latex
C_i(x,y) = \mu_i + \sigma_i \cdot \mathcal{N}(0,1)
```
where:
- \(\mu_i\) = Mean concentration of element \(i\)
- \(\sigma_i\) = Standard deviation (spatial variability)
- \((x,y)\) = Spatial coordinates

---

### Summary of Key Parameters
| **Symbol** | **Description** | **Typical Value/Units** |
|------------|-----------------|-------------------------|
| \(k_{\text{diss}}\) | Dissolution rate | \(10^{-8}\) kg/m²/day |
| \(Y\) | Alkalinity yield | 2.0 mol/kg_basalt |
| \(\rho_{\text{soil}}\) | Soil bulk density | 1500 kg/m³ |
| \(d\) | Soil mixing depth | 0.2 m |
| \(K_h\) | Henry's constant | \(3.3 \times 10^{-4}\) mol/L/atm |
| \(K_1\) | Carbonic acid dissociation | \(4.45 \times 10^{-7}\) |
| \(p_{\text{CO}_2}\) | Soil CO₂ partial pressure | 0.015 atm |

These equations form the core mathematical framework of the ERW digital twin, integrating geochemistry, reaction kinetics, and economic constraints. The modular implementation allows customization for different spatial contexts and rock types.
