# CTM Simulation Implementation

This document describes the Cell Transmission Model (CTM) implementation in `ctm_simulation.py`, which follows the triangular fundamental diagram specification.

## Overview

The implementation has been completely reworked to implement the CTM according to the provided mathematical specification. All ALINEA and ramp metering logic has been removed.

## Core Equations

### Triangular Fundamental Diagram
```
q = rho * v
```

### First-Order Density Update
```
rho_i(k+1) = rho_i(k) + T/(L_i * lambda_i) * (q_(i-1)(k) + r_i(k) - q_i(k) - s_i(k))
```

### Cell Flow
```
q_i(k) = min(
    Q_(i,max),
    Q_(i+1,max) * (rho_jam - rho_(i+1)(k)) / (rho_jam - rho_crit),
    v_f * rho_i(k) * lambda_i
)
```

### On-Ramp Flow
```
r_i(k) = min(
    Q_(ramp,max),
    Q_(i,max) * (rho_jam - rho_i(k)) / (rho_jam - rho_crit) - q_in(k),
    d_i(k) + N_i(k) / T
)
```

### Ramp Queue Dynamics
```
N_i(k+1) = N_i(k) + T * (d_i(k) - r_i(k))
```

### Speed
```
v_i(k) = q_i(k) / (lambda_i * rho_i(k))
```

## Variables and Units

| Variable | Description | Unit |
|----------|-------------|------|
| q_i(k) | Flow exiting cell i at time step k | veh/h |
| rho_i(k) | Density within cell i at time step k | veh/km/lane |
| v_i(k) | Speed in cell i at time step k | km/h |
| r_i(k) | On-ramp flow entering cell i at time step k | veh/h |
| d_i(k) | Demand for on-ramp of cell i at time step k | veh/h |
| N_i(k) | Queue for on-ramp of cell i at time step k | veh |
| s_i(k) | Off-ramp flow exiting cell i at time step k | veh/h |

## Parameters

| Parameter | Description | Unit |
|-----------|-------------|------|
| T | Time step length | h |
| L_i | Length of cell i | km |
| lambda_i | Number of lanes at cell i | - |
| Q_max | Maximum capacity of FD | veh/h/lane |
| v_f | Free-flow speed of FD | km/h |
| rho_crit | Critical density of FD | veh/km/lane |
| rho_jam | Jam density of FD | veh/km/lane |
| Q_(i,max) | Maximum capacity at cell i = lambda_i * Q_max | veh/h |
| Q_(ramp,max) | Maximum capacity at on-ramp = lambda_ramp * Q_max | veh/h |

## Usage

### Basic Example

```python
from ctm_simulation import simulate_ctm, create_uniform_parameters
import numpy as np

# Setup parameters (CFL-compliant)
params = create_uniform_parameters(
    n_cells=5,
    cell_length_km=0.5,
    lanes=3,
    T=0.004,  # Must satisfy T <= L/v_f for stability
    Q_max=2000.0,
    v_f=100.0,
    rho_crit=30.0,
    rho_jam=150.0,
    lambda_ramp=1.0,
)

# Initial conditions
rho0 = np.array([20.0, 25.0, 30.0, 25.0, 20.0])  # Initial densities
N0 = np.array([5.0])  # Initial ramp queue

# Simulation inputs
K = 100  # Number of time steps
d = np.full((K, 1), 600.0)  # Constant ramp demand
s = np.zeros((K, 5))  # No off-ramps
q_upstream = np.full(K, 5000.0)  # Upstream inflow
ramp_to_cell = np.array([2])  # Ramp merges at cell 2

# Run simulation
results = simulate_ctm(
    rho0=rho0,
    N0=N0,
    d=d,
    s=s,
    params=params,
    K=K,
    q_upstream=q_upstream,
    ramp_to_cell=ramp_to_cell,
)

# Access results
print(f"Final densities: {results.rho[-1]}")
print(f"Final ramp queue: {results.N[-1]}")
print(f"Average speed: {np.mean(results.v):.2f} km/h")
```

## Important: CFL Stability Condition

For numerical stability, the time step must satisfy the **Courant-Friedrichs-Lewy (CFL) condition**:

```
T ≤ min(L_i) / v_f
```

If this condition is violated, the simulation will produce numerical oscillations and incorrect results. The implementation automatically checks this condition and issues a warning.

**Example:**
- Cell length L = 0.5 km
- Free-flow speed v_f = 100 km/h
- Maximum stable time step: T ≤ 0.5/100 = 0.005 h (18 seconds)

## Data Structures

### CTMParameters
Holds all simulation parameters with validation including CFL check.

### CTMResults
Contains time series outputs:
- `rho`: Densities [veh/km/lane], shape (K+1, n_cells)
- `N`: Ramp queues [veh], shape (K+1, n_ramps)
- `q`: Cell flows [veh/h], shape (K, n_cells)
- `r`: On-ramp flows [veh/h], shape (K, n_ramps)
- `v`: Speeds [km/h], shape (K, n_cells)
- `s`: Off-ramp flows [veh/h], shape (K, n_cells)
- `time`: Time vector [h], shape (K+1,)

## Testing

Run the test suite:
```bash
cd Exercise2
python test_ctm_new.py
```

All 5 tests should pass:
- ✓ Conservation: Vehicle conservation verified
- ✓ Steady State: Equilibrium maintained with constant inputs
- ✓ Ramp Queue: Queue builds up when demand exceeds service
- ✓ Capacity Constraint: Flows respect maximum capacity
- ✓ Free Flow: Free-flow speeds achieved at low densities

## Validation Scripts

- `test_ctm_new.py`: Unit tests for core functionality
- `debug_ctm.py`: Step-by-step tracing for debugging
- `validate_ctm.py`: Scenario validation with detailed output
- `comprehensive_validation.py`: Full feature validation

## Key Implementation Details

1. **Ramp/Mainline Competition**: On-ramp flows correctly account for mainline flows when computing available receiving capacity in target cells.

2. **Downstream Boundary**: Last cell uses its own capacity as free outflow boundary condition.

3. **Numerical Robustness**:
   - Division by zero avoided in speed calculation (uses v_f when rho ≈ 0)
   - All flows, densities, and queues kept non-negative
   - Densities clamped to [0, rho_jam]
   - Speeds clamped to [0, v_f]

4. **Physical Constraints**: All CTM equations respect physical limits automatically through min/max operations.

## Differences from Previous Implementation

- **Removed**: All ALINEA feedback control logic
- **Removed**: Ramp metering control algorithms
- **Removed**: Priority-based merging with complex redistribution
- **Added**: Pure CTM implementation per specification
- **Added**: CFL stability checking
- **Added**: Simplified, equation-driven approach

## References

The implementation follows the standard Cell Transmission Model formulation with triangular fundamental diagram as specified in the problem statement.
