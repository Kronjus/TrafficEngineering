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
- **Upstream queue** for spillback handling when downstream capacity is exhausted
- **Queue dynamics** for on-ramps that cannot immediately merge
- **Heterogeneous cells** supporting lane drops/additions and varying parameters
- **VKT/VHT computation** built-in methods for vehicle-kilometers and vehicle-hours traveled
- **Velocity tracking** per-cell speed time series for congestion analysis

### METANET

METANET is a second-order macroscopic traffic flow model that tracks both density and speed evolution. It includes:

- **Speed dynamics** with relaxation, convection, and anticipation terms
- **Greenshields equilibrium speed** function (customizable)
- **Compatible API** with CTM for easy model comparison
- **Advanced traffic behavior** modeling driver anticipation and wave propagation
- **Same features** as CTM (on-ramps, metering, heterogeneous cells, upstream queue)

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

## ALINEA Ramp Metering

### Overview

ALINEA is a feedback control strategy for ramp metering that regulates on-ramp flow to maintain optimal density on the mainline freeway. The implementation in `metanet_model.py` includes:

- **Integral feedback control**: r[k+1] = r[k] + K_I × (ρ_crit - ρ_measured)
- **Separate integrator state**: Maintains commanded flow independent of achieved flow
- **Anti-windup logic**: Prevents integrator wind-up during saturation
- **Saturation bounds**: Respects actuator limits, demand arrivals, and available supply
- **Ramp lane accounting**: Capacity accounts for number of ramp lanes

### Running ALINEA Examples

Run the comprehensive ALINEA demonstration and parameter scan:

```bash
python Exercise2/run_alinea.py
```

This script:
1. Runs scenarios A, B, and C without ALINEA (baseline)
2. Performs parameter scan to find optimal K_I for scenarios B and C
3. Runs scenarios with optimal K_I values
4. Generates density/speed/flow time-series plots
5. Creates 2D and 3D space-time visualizations
6. Saves all plots to `Exercise2/outputs/`

Output files follow naming convention:
- `scenario_{letter}_metanet.png` - Without ALINEA
- `scenario_{letter}_alinea_metanet.png` - With ALINEA

### Verifying ALINEA Implementation

Run the verification script to confirm ALINEA is working correctly:

```bash
python Exercise2/verify_alinea.py
```

This verification script:
1. **Prints timestep-by-timestep variables**: Shows ρ_measured, r_cmd (commanded flow), q_ramp (achieved flow), queue length, and saturation status
2. **Checks integrator behavior**: Verifies separate integrator state updates correctly
3. **Validates anti-windup**: Confirms integrator freezes when saturated
4. **Compares with/without ALINEA**: Shows impact on VKT, VHT, average speed, and queue lengths

Expected verification output:
- Integrator state (r_cmd) increases when ρ_measured < ρ_crit
- Achieved flow (q_ramp) respects all saturation bounds
- Anti-windup activates when q_ramp < r_cmd
- ALINEA modifies traffic metrics compared to no-control baseline

### ALINEA Parameters

Key parameters in `metanet_model.py`:

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `K_I` | Integral gain [veh/h per veh/km/lane] | 0-20 (tune via scan) |
| `rho_crit` | Target critical density [veh/km/lane] | ~33 for standard freeways |
| `measured_cell` | Cell index for density measurement | Downstream of merge (e.g., cell 2 or 4) |
| `ramp_lanes` | Number of ramp lanes | 1.0 (single lane ramp) |
| `Q_lane` | Ramp capacity per lane [veh/h] | 2000 |

### Tuning K_I

The optimal K_I depends on:
- **Sampling time** (T_step): Smaller steps require smaller K_I
- **Traffic demand**: Higher demand may need stronger control
- **Measurement location**: Further downstream needs different gains
- **Performance objective**: Minimize VHT, maximize throughput, etc.

Use `scan_K()` function in `run_alinea.py` to find optimal K_I:

```python
from run_alinea import scan_K
import numpy as np

lanes = np.full(6, 3.0)
K_values, vht_values, avg_speed = scan_K(
    d_main=4000.0,
    d_ramp=2500.0,
    lanes=lanes,
    measured_cell=2,
    K_min=0.5,
    K_max=20.0,
    n_K=1000
)

# Find optimal K_I
idx_opt = np.argmin(vht_values)
K_opt = K_values[idx_opt]
print(f"Optimal K_I: {K_opt:.2f} with VHT: {vht_values[idx_opt]:.1f} veh·h")
```

### Implementation Details

**Integrator Update** (in `metanet_model.py`, lines 103-127):
```python
# Separate integrator/command state
r_prev_cmd = 0.0  # Initialized at start

# At each timestep:
if K_I > 0.0 and measured_cell is not None:
    rho_meas = density[measured_cell]
    # Update commanded flow (integrator)
    r_cmd = r_prev_cmd + K_I * (rho_crit - rho_meas)
    r_cmd = max(0.0, r_cmd)
else:
    r_cmd = arrivals_ramp

# Apply all saturation bounds
q_ramp = min(r_cmd, arrivals_ramp, q_ramp_max, q_supply)

# Anti-windup: update integrator only when not saturated
if K_I > 0.0 and measured_cell is not None:
    if abs(q_ramp - r_cmd) < 1e-9:
        r_prev_cmd = r_cmd  # Command achieved, update integrator
    # else: keep r_prev_cmd unchanged (integrator frozen)
```

**Key Features**:
1. `r_prev_cmd` stores integrator state (not achieved flow)
2. Anti-windup prevents wind-up when bounds are active
3. Bounds check arrivals, capacity (per lane × lanes), and available supply
4. Measured density from specified cell provides feedback signal

### Expected Results

**Scenario B** (d_main=4000, d_ramp=2500, uniform 3 lanes):
- Without ALINEA: VHT ≈ 344 veh·h, avg speed ≈ 41 km/h
- With ALINEA (K_I≈6): VHT ≈ 355 veh·h, avg speed ≈ 40 km/h
- ALINEA trades slight VHT increase for improved mainline flow stability

**Scenario C** (d_main=1500, d_ramp=1500, lane drop at cell 3):
- Without ALINEA: VHT ≈ 577 veh·h, avg speed ≈ 11 km/h (severe congestion)
- With ALINEA (K_I≈7-8): VHT ≈ 232-538 veh·h (varies with tuning)
- ALINEA can significantly reduce congestion at bottlenecks

### Troubleshooting

**Issue**: ALINEA increases VHT instead of decreasing it
- **Cause**: K_I too high (overshooting) or too low (undercontrol)
- **Solution**: Run parameter scan, adjust measured_cell location

**Issue**: Ramp queue builds up excessively
- **Cause**: K_I restricting ramp flow too much
- **Solution**: Lower K_I or verify ρ_crit is appropriate

**Issue**: No visible control action
- **Cause**: K_I=0, measured_cell=None, or density never exceeds ρ_crit
- **Solution**: Verify parameters, check demands create congestion

**Issue**: Integrator wind-up suspected
- **Cause**: Anti-windup not working
- **Solution**: Run `verify_alinea.py` to check saturation detection

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
    nu=60.0,     # Anticipation coefficient (km²/h)
    kappa=40.0,  # Regularization constant (veh/km/lane)
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

### Upstream Queue for Spillback

Both CTM and METANET simulators now include an upstream queue that properly handles spillback when downstream capacity is exhausted. This ensures vehicle conservation and realistic queueing behavior:

- **Automatic spillback handling**: Vehicles accumulate in the upstream queue when the first cell cannot accept more flow
- **Queue draining**: As downstream capacity becomes available, queued vehicles enter the system
- **Conservation**: All vehicles are accounted for (no vehicles are lost)
- **Tracking**: Upstream queue state is recorded in `SimulationResult.upstream_queue`

The upstream queue is transparent to users - it's automatically managed by the simulation engine and doesn't require additional configuration.

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
# Test CTM
python -m unittest Exercise2.test_ctm_simulation -v

# Test METANET
python -m unittest Exercise2.test_metanet_simulation -v

# Test both together
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
- Upstream queue and spillback handling

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

### METANET with ALINEA Files
- `metanet_model.py` - Core METANET simulation engine with ALINEA ramp metering implementation
- `run_alinea.py` - Scenario runner with ALINEA parameter scanning and plotting utilities
- `verify_alinea.py` - ALINEA verification script with timestep logging and comparison
- `outputs/` - Generated plots and visualizations

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