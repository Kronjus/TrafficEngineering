# Exercise 2 – Traffic Flow Simulation Playground

This directory contains two complementary macroscopic traffic flow simulators:

1. **Cell Transmission Model (CTM)** - First-order model tracking density
2. **METANET** - Second-order model tracking density and speed

Both implementations offer lightweight sandboxes for experimenting with heterogeneous mainline cells, on-ramp interactions, and metering strategies, with compatible APIs for easy comparison.

## Overview

### Cell Transmission Model (CTM)

The Cell Transmission Model (CTM) is a discrete approximation of the Lighthill-Whitham-Richards (LWR) traffic flow model. It discretizes highways into cells and simulates vehicle flow using:

- **Triangular fundamental diagram** (free-flow speed, congestion wave speed, capacity, jam density)
- **Time-varying demand profiles** for upstream boundary and on-ramps
- **On-ramp merging** with configurable priority and optional metering
- **Queue dynamics** for on-ramps that cannot immediately merge
- **Heterogeneous cells** supporting lane drops/additions and varying parameters

### METANET

METANET is a second-order macroscopic traffic flow model that tracks both density and speed evolution. It includes:

- **Speed dynamics** with relaxation, convection, and anticipation terms
- **Greenshields equilibrium speed** function (customizable)
- **Compatible API** with CTM for easy model comparison
- **Advanced traffic behavior** modeling driver anticipation and wave propagation
- **Same features** as CTM (on-ramps, metering, heterogeneous cells)

## Quick Start

### CTM Quick Start

Run the basic CTM demonstration scenario:

```bash
python Exercise2/ctm_simulation.py
```

This executes `run_basic_scenario`, demonstrating a three-cell freeway with an on-ramp.

### METANET Quick Start

Run the basic METANET demonstration scenario:

```bash
python Exercise2/metanet_simulation.py
```

This executes `run_basic_metanet_scenario`, showing the same scenario with METANET's speed dynamics.

### Compare CTM and METANET

Run side-by-side comparisons:

```bash
python Exercise2/compare_ctm_metanet.py
```

This compares both models under identical conditions to understand their differences.

## Running Examples

### CTM Examples

The `examples_ctm.py` script contains four comprehensive demonstration scenarios:

```bash
python Exercise2/examples_ctm.py
```

Examples include:
1. **Basic Freeway** - Time-varying demand on a 5-cell mainline
2. **On-Ramp Merge** - Single on-ramp with lane drop and queue dynamics
3. **Ramp Metering** - Comparison of metered vs unmetered scenarios
4. **Complex Scenario** - 10-cell freeway with two on-ramps and variable lanes

### METANET Examples

The `examples_metanet.py` script contains five comprehensive demonstration scenarios:

```bash
python Exercise2/examples_metanet.py
```

Examples include:
1. **Basic Freeway** - Time-varying demand with speed dynamics
2. **On-Ramp Merge** - Single on-ramp with lane drop
3. **Ramp Metering** - Comparison of metered vs unmetered scenarios
4. **Speed Dynamics** - Wave propagation and stop-and-go behavior
5. **Complex Scenario** - 10-cell freeway with two on-ramps

Install matplotlib for visualizations:
```bash
pip install matplotlib
```

## Interactive Notebook

Explore the simulator interactively using Jupyter:

```bash
jupyter notebook Exercise2/ctm_simulation.ipynb
```

The notebook includes:
- Guided examples with explanations
- Visualization of densities, flows, and queues
- Interactive experimentation section
- Educational content on CTM theory

## Programmatic Usage

### Using CTM

Import and use the CTM simulator in your own code:

```python
from Exercise2.ctm_simulation import (
    build_uniform_mainline,
    CTMSimulation,
    OnRampConfig,
)

# Create a 4-cell mainline with lane drop
cells = build_uniform_mainline(
    num_cells=4,
    cell_length_km=0.5,
    lanes=[3, 3, 3, 2],  # Lane drop at cell 3
    free_flow_speed_kmh=100.0,
    congestion_wave_speed_kmh=20.0,
    capacity_veh_per_hour_per_lane=2200.0,
    jam_density_veh_per_km_per_lane=160.0,
)

# Configure on-ramp with time-varying demand
on_ramp = OnRampConfig(
    target_cell=2,
    arrival_rate_profile=lambda step: 400 if step < 12 else 150,
    meter_rate_veh_per_hour=600.0,  # Optional metering
    mainline_priority=0.7,  # 70% mainline priority at merge
)

# Create and run simulation
sim = CTMSimulation(
    cells=cells,
    time_step_hours=0.1,  # 6 minutes
    upstream_demand_profile=[1800.0] * 24,  # Constant or list/callable
    downstream_supply_profile=1900.0,  # Optional downstream constraint
    on_ramps=[on_ramp],
)

result = sim.run(steps=24)
```

### Using METANET

Import and use the METANET simulator:

```python
from Exercise2.metanet_simulation import (
    build_uniform_metanet_mainline,
    METANETSimulation,
    METANETOnRampConfig,
)

# Create a 4-cell mainline
cells = build_uniform_metanet_mainline(
    num_cells=4,
    cell_length_km=0.5,
    lanes=[3, 3, 3, 2],
    free_flow_speed_kmh=100.0,
    jam_density_veh_per_km_per_lane=160.0,
    tau_s=18.0,  # Relaxation time (seconds)
    eta=60.0,    # Anticipation coefficient
    kappa=40.0,  # Regularization constant
)

# Configure on-ramp (same API as CTM)
on_ramp = METANETOnRampConfig(
    target_cell=2,
    arrival_rate_profile=lambda step: 400 if step < 12 else 150,
    meter_rate_veh_per_hour=600.0,
    mainline_priority=0.7,
)

# Create and run simulation (same API as CTM)
sim = METANETSimulation(
    cells=cells,
    time_step_hours=0.1,
    upstream_demand_profile=[1800.0] * 24,
    downstream_supply_profile=1900.0,
    on_ramps=[on_ramp],
)

result = sim.run(steps=24)
```

Both simulators return the same `SimulationResult` object for compatibility.

## Accessing Results

The `SimulationResult` object provides convenient access to outputs:

```python
# Time series data
result.time_vector()       # List of sample times (hours)
result.interval_vector()   # List of (start, end) tuples for flows
result.densities["cell_0"] # Density trace for cell 0
result.flows["cell_1"]     # Flow from cell 1 to cell 2
result.ramp_queues["ramp_2"]  # Queue length at on-ramp
result.ramp_flows["ramp_2"]   # Flow from on-ramp into mainline

# Convert to pandas DataFrame (optional)
df = result.to_dataframe()
print(df.head())
```

## Key Classes

### `CellConfig`
Configuration for a single CTM cell with parameters:
- `name`: Cell identifier
- `length_km`: Cell length
- `lanes`: Number of lanes
- `free_flow_speed_kmh`: Free-flow speed
- `congestion_wave_speed_kmh`: Congestion wave speed
- `capacity_veh_per_hour_per_lane`: Capacity per lane
- `jam_density_veh_per_km_per_lane`: Jam density per lane
- `initial_density_veh_per_km_per_lane`: Initial density (default: 0)

### `OnRampConfig`
Configuration for an on-ramp:
- `target_cell`: Cell index (int) or name (str) where ramp merges
- `arrival_rate_profile`: Demand profile (constant, list, or callable)
- `meter_rate_veh_per_hour`: Optional metering rate limit
- `mainline_priority`: Merge priority 0-1 (higher favors mainline)
- `initial_queue_veh`: Initial queue length (default: 0)

### `CTMSimulation`
Main simulator class:
- `cells`: List of CellConfig objects
- `time_step_hours`: Simulation time step
- `upstream_demand_profile`: Upstream boundary demand
- `downstream_supply_profile`: Optional downstream supply constraint
- `on_ramps`: Optional list of OnRampConfig objects

### `SimulationResult`
Container for simulation outputs with time series and helper methods.

## Testing

Run the comprehensive test suites:

```bash
# Test CTM (32 tests)
python -m unittest Exercise2.test_ctm_simulation -v

# Test METANET (35 tests)
python -m unittest Exercise2.test_metanet_simulation -v

# Test both together (67 tests)
python -m unittest Exercise2.test_ctm_simulation Exercise2.test_metanet_simulation -v
```

The test suites cover:
- Configuration validation
- Flow and density calculations
- Vehicle conservation
- On-ramp merging and queuing
- Profile handling (constant, list, callable)
- Capacity constraints
- Merge priority behavior
- Speed dynamics (METANET only)
- Steady-state preservation

## Helper Functions

### `build_uniform_mainline()`
Utility to quickly create a list of cells with uniform or varying parameters:

```python
cells = build_uniform_mainline(
    num_cells=5,
    cell_length_km=0.5,
    lanes=[3, 3, 3, 2, 2],  # Variable lanes (or single int)
    free_flow_speed_kmh=100.0,
    congestion_wave_speed_kmh=20.0,
    capacity_veh_per_hour_per_lane=2200.0,
    jam_density_veh_per_km_per_lane=160.0,
    initial_density_veh_per_km_per_lane=0.0,  # Scalar or sequence
)
```

### `run_basic_scenario()`
Pre-configured demonstration scenario for quick testing:

```python
from Exercise2.ctm_simulation import run_basic_scenario

result = run_basic_scenario(steps=20)
print(result.to_dataframe())
```

## CTM Theory

The Cell Transmission Model is based on:

1. **Conservation of vehicles**: Change in density = (inflow - outflow) / length
2. **Sending function**: Maximum flow a cell can send downstream
   - S(ρ) = min(v·ρ, Q_max)
3. **Receiving function**: Maximum flow a cell can accept
   - R(ρ) = min(w·(ρ_jam - ρ), Q_max)
4. **Flow between cells**: F = min(S_upstream, R_downstream)
5. **On-ramp merge**: Split available receiving capacity by priority

Where:
- v = free-flow speed
- w = congestion wave speed  
- ρ = density
- ρ_jam = jam density
- Q_max = capacity = v · ρ_critical
- ρ_critical = (w · ρ_jam) / (v + w)

## METANET Theory

METANET is a second-order model with discrete-time dynamics:

1. **Density update** (conservation):
   - ρᵢ(k+1) = ρᵢ(k) + (T/Lᵢ) × [qᵢ₋₁(k) - qᵢ(k) + rᵢ(k)]
   
2. **Speed update** (dynamics):
   - vᵢ(k+1) = vᵢ(k) + T × [(V(ρᵢ) - vᵢ)/τ - (vᵢ/Lᵢ)(vᵢ - vᵢ₋₁) - (η/τ)(ρᵢ₊₁ - ρᵢ)/(ρᵢ + κ)]
   
3. **Flow relation**: qᵢ = ρᵢ × vᵢ × λᵢ

Terms:
- **Relaxation**: (V(ρᵢ) - vᵢ)/τ - drivers adapt to equilibrium speed
- **Convection**: -(vᵢ/Lᵢ)(vᵢ - vᵢ₋₁) - speed propagation from upstream
- **Anticipation**: -(η/τ)(ρᵢ₊₁ - ρᵢ)/(ρᵢ + κ) - drivers react to downstream density

See `METANET_README.md` for detailed equations and parameter guidance.

## Choosing Between CTM and METANET

**Use CTM when:**
- Focus is on flow and density patterns
- Computational efficiency is important  
- Detailed speed dynamics are not critical
- Simple calibration is desired
- Analyzing capacity and bottleneck effects

**Use METANET when:**
- Speed dynamics are important (e.g., variable speed limits)
- Modeling driver anticipatory behavior
- Designing traffic control strategies
- Studying wave propagation and oscillations
- Need realistic transient behavior

**API Compatibility:**
Both models share the same result structure (`SimulationResult`) and similar configuration APIs, making it easy to swap models for comparison or experimentation.

## Design Choices

- **Units**: Consistent use of hours for time, km for distance, vehicles for counts
- **CFL Stability**: Not explicitly enforced but should satisfy: dt ≤ dx / v
- **Merge Logic**: Priority-based allocation with spillover to remaining capacity
- **Queue Model**: On-ramps modeled as queues (vehicles), not explicit cells
- **Single Ramp per Cell**: Current implementation supports one on-ramp per target cell

## Extensibility

The simulator can be extended for:
- Multiple on-ramps per cell (requires refactoring merge logic)
- Off-ramps (add outflow extraction before sending calculation)
- Variable speed limits (time-dependent cell parameters)
- Advanced metering strategies (dynamic control algorithms)
- Incidents and capacity reductions (time-varying parameters)
- Different fundamental diagrams (modify sending/receiving functions)

## References

- Daganzo, C. F. (1994). "The cell transmission model: A dynamic representation of highway traffic consistent with the hydrodynamic theory." *Transportation Research Part B*, 28(4), 269-287.
- Daganzo, C. F. (1995). "The cell transmission model, part II: Network traffic." *Transportation Research Part B*, 29(2), 79-93.

## Troubleshooting

**Oscillating densities**: Ensure time step satisfies CFL condition (dt ≤ dx / v). Reduce time step if oscillations occur.

**High queue buildup**: Check that merge priority allows sufficient on-ramp flow, or adjust meter rate if metering is active.

**Unrealistic flows**: Verify that capacity, free-flow speed, and jam density parameters are physically consistent.

**Missing pandas**: The simulator works without pandas, but `to_dataframe()` requires it. Install with `pip install pandas`.

## Files

### CTM Files
- `ctm_simulation.py` - Core CTM implementation
- `test_ctm_simulation.py` - CTM unit tests (32 tests)
- `examples_ctm.py` - Four demonstration scenarios with optional visualization
- `ctm_simulation.ipynb` - Interactive Jupyter notebook with guided examples

### METANET Files
- `metanet_simulation.py` - Core METANET implementation
- `test_metanet_simulation.py` - METANET unit tests (35 tests)
- `examples_metanet.py` - Five demonstration scenarios with optional visualization
- `METANET_README.md` - Detailed METANET documentation with equations

### Comparison and Documentation
- `compare_ctm_metanet.py` - Side-by-side model comparison
- `README.md` - This file