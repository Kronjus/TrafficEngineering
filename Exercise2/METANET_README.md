# METANET Traffic Flow Simulation

## Overview

This module implements the METANET macroscopic traffic flow model, a second-order model that tracks both density and speed evolution over time. METANET is widely used for freeway traffic modeling, control design, and simulation.

## Model Description

### State Variables

METANET maintains two state variables for each cell `i`:
- **Density** `ρᵢ(k)`: vehicles per km per lane at time step `k`
- **Speed** `vᵢ(k)`: average speed in km/h at time step `k`

### Discrete-Time Dynamics

The METANET model uses the following discrete-time update equations:

#### 1. Density Update (Conservation Equation)

```
ρᵢ(k+1) = ρᵢ(k) + (T/Lᵢ) * [qᵢ₋₁(k) - qᵢ(k) + rᵢ(k) - sᵢ(k)]
```

Where:
- `T`: simulation time step (hours)
- `Lᵢ`: length of cell `i` (km)
- `qᵢ(k) = ρᵢ(k) * vᵢ(k) * λᵢ`: flow leaving cell `i` (veh/h)
- `λᵢ`: number of lanes in cell `i`
- `rᵢ(k)`: on-ramp flow entering cell `i` (veh/h)
- `sᵢ(k)`: off-ramp flow leaving cell `i` (veh/h)

#### 2. Speed Update (Dynamic Equation)

```
vᵢ(k+1) = vᵢ(k) + T * [
    (V(ρᵢ) - vᵢ) / τ                           // Relaxation term
    - (vᵢ / Lᵢ) * (vᵢ - vᵢ₋₁)                  // Convection term
    - (ν / τ) * (ρᵢ₊₁ - ρᵢ) / (ρᵢ + κ)        // Anticipation term
]
```

Where:
- `V(ρᵢ)`: equilibrium speed at density `ρᵢ` (km/h)
- `τ`: relaxation time constant (seconds)
- `ν` (nu): anticipation/diffusion coefficient (km²/h)
- `κ`: small regularization constant (veh/km/lane)

### Equilibrium Speed Function

The default equilibrium speed function is the generalized Greenshields model:

```
V(ρ) = vf * (1 - (ρ/ρⱼₐₘ)^δ)
```

Where:
- `vf`: free-flow speed (km/h)
- `ρⱼₐₘ`: jam density (veh/km/lane)
- `δ` (delta): dimensionless exponent controlling nonlinearity (typically 1-4)

When `δ = 1`, this reduces to the classic linear Greenshields model.
Higher values of `δ` create more nonlinear speed-density relationships.

Other models (e.g., exponential) can be provided via the `equilibrium_speed_func` parameter.

### Model Terms Explained

1. **Relaxation Term**: `(V(ρᵢ) - vᵢ) / τ`
   - Drives the speed toward the equilibrium speed for the current density
   - `τ` controls how quickly drivers adapt their speed
   - Typical values: 10-30 seconds

2. **Convection Term**: `-(vᵢ / Lᵢ) * (vᵢ - vᵢ₋₁)`
   - Captures the propagation of speed changes from upstream
   - Models drivers adjusting speed based on upstream traffic
   - Only active when there is upstream flow

3. **Anticipation Term**: `-(ν / τ) * (ρᵢ₊₁ - ρᵢ) / (ρᵢ + κ)`
   - Models drivers' anticipatory behavior to downstream congestion
   - Drivers slow down when they see higher density ahead
   - `ν` (nu) controls the strength of anticipation
   - `κ` prevents division by zero at low densities

## Parameters

### Cell Configuration

Each METANET cell requires:

- **Geometric parameters:**
  - `length_km`: Cell length (km)
  - `lanes`: Number of lanes

- **Fundamental diagram parameters:**
  - `free_flow_speed_kmh`: Free-flow speed vf (km/h)
  - `jam_density_veh_per_km_per_lane`: Jam density ρⱼₐₘ (veh/km/lane)
  - `capacity_veh_per_hour_per_lane`: Per-lane capacity (veh/h/lane)

- **METANET-specific parameters:**
  - `tau_s`: Relaxation time τ (seconds), default: 18.0s
  - `nu`: Anticipation/diffusion coefficient ν (km²/h), default: 60.0
  - `kappa`: Regularization constant κ (veh/km/lane), default: 40.0
  - `delta`: Dimensionless exponent δ for equilibrium speed, default: 1.0

- **Initial conditions:**
  - `initial_density_veh_per_km_per_lane`: Initial density (veh/km/lane)
  - `initial_speed_kmh`: Initial speed (km/h), defaults to V(ρ₀)

### Simulation Parameters

- `time_step_hours`: Simulation time step T (hours)
  - Typical values: 5-10 seconds (0.00139-0.00278 hours)
  - Smaller steps improve stability but increase computation time

### Boundary Conditions

- **Upstream boundary:** `upstream_demand_profile` (veh/h)
  - Specifies the demand entering the first cell
  - Can be constant, time-varying sequence, or callable

- **Downstream boundary:** `downstream_supply_profile` (veh/h, optional)
  - Constrains the flow exiting the last cell
  - If not provided, uses the last cell's capacity

### On-Ramps

On-ramps are configured similarly to CTM:

- `target_cell`: Cell index where ramp merges
- `arrival_rate_profile`: Ramp arrival demand (veh/h)
- `meter_rate_veh_per_hour`: Optional metering rate (veh/h)
- `mainline_priority`: Merge priority fraction [0, 1]
- `initial_queue_veh`: Initial queue length (vehicles)

## Usage Examples

### Basic Freeway Simulation

```python
from metanet_simulation import (
    METANETSimulation,
    build_uniform_metanet_mainline,
)

# Create 5-cell mainline
cells = build_uniform_metanet_mainline(
    num_cells=5,
    cell_length_km=0.5,
    lanes=3,
    free_flow_speed_kmh=100.0,
    jam_density_veh_per_km_per_lane=160.0,
    tau_s=18.0,
    nu=60.0,
    kappa=40.0,
    delta=1.0,
)

# Create simulation
sim = METANETSimulation(
    cells=cells,
    time_step_hours=0.05,  # 3 minutes
    upstream_demand_profile=5000.0,  # veh/h
)

# Run for 30 time steps
result = sim.run(steps=30)

# Access results
print(result.densities)  # Dict[cell_name, List[float]]
print(result.flows)      # Dict[cell_name, List[float]]
```

### With On-Ramp

```python
from metanet_simulation import METANETOnRampConfig

# Configure on-ramp
ramp = METANETOnRampConfig(
    target_cell=2,
    arrival_rate_profile=800.0,
    meter_rate_veh_per_hour=500.0,
    mainline_priority=0.6,
)

# Create simulation with ramp
sim = METANETSimulation(
    cells=cells,
    time_step_hours=0.05,
    upstream_demand_profile=4000.0,
    on_ramps=[ramp],
)

result = sim.run(steps=30)
print(result.ramp_queues)  # Ramp queue lengths
print(result.ramp_flows)   # Ramp flows
```

### Custom Equilibrium Speed

```python
def custom_speed(density, vf, rho_jam):
    """Exponential equilibrium speed function."""
    import math
    alpha = 2.0
    return vf * math.exp(-(density / rho_jam) ** alpha)

sim = METANETSimulation(
    cells=cells,
    time_step_hours=0.05,
    upstream_demand_profile=5000.0,
    equilibrium_speed_func=custom_speed,
)
```

## Comparison with CTM

### Similarities

- Both are macroscopic models (aggregate traffic flow)
- Both track cell-based densities
- Both support on-ramps and boundary conditions
- Both use discrete-time simulation
- Compatible API design (same `SimulationResult` output)

### Differences

| Feature | CTM | METANET |
|---------|-----|---------|
| **Order** | First-order (density only) | Second-order (density + speed) |
| **State variables** | Density ρ | Density ρ and speed v |
| **Speed** | Determined from FD | Dynamic state variable |
| **Wave propagation** | Characteristic-based | Anticipatory behavior |
| **Calibration** | FD parameters | FD + τ, η, κ |
| **Complexity** | Simpler, faster | More detailed dynamics |
| **Applications** | Flow modeling, capacity analysis | Control design, dynamic behavior |

### When to Use Each Model

**Use CTM when:**
- Focus is on flow and density patterns
- Computational efficiency is important
- Detailed speed dynamics are not critical
- Simple calibration is desired

**Use METANET when:**
- Speed dynamics are important (e.g., speed control)
- Modeling driver anticipatory behavior
- Designing traffic control strategies
- Studying wave propagation and oscillations
- Need more realistic transient behavior

## Validation and Testing

The implementation includes comprehensive tests:

1. **Conservation tests**: Vehicle conservation in closed systems
2. **Steady-state tests**: Equilibrium preservation
3. **Boundary condition tests**: Proper inflow/outflow handling
4. **On-ramp tests**: Merge logic and queue dynamics
5. **Speed dynamics tests**: Relaxation, convection, anticipation
6. **Comparison tests**: Side-by-side with CTM

Run tests with:
```bash
python -m unittest test_metanet_simulation
```

## References

1. Papageorgiou, M., Blosseville, J. M., & Hadj-Salem, H. (1990). "Modelling and real-time control of traffic flow on the southern part of Boulevard Périphérique in Paris: Part I: Modelling". Transportation Research Part A: General, 24(5), 345-359.

2. Hegyi, A., De Schutter, B., & Hellendoorn, H. (2005). "Model predictive control for optimal coordination of ramp metering and variable speed limits". Transportation Research Part C: Emerging Technologies, 13(3), 185-209.

3. Papamichail, I., Papageorgiou, M., Vong, V., & Gaffney, J. (2010). "Heuristic ramp-metering coordination strategy implemented at Monash Freeway, Australia". Transportation Research Record, 2178(1), 10-20.

## See Also

- `ctm_simulation.py`: Cell Transmission Model implementation
- `examples_metanet.py`: Detailed usage examples
- `compare_ctm_metanet.py`: CTM vs METANET comparison
- `test_metanet_simulation.py`: Unit tests and validation
