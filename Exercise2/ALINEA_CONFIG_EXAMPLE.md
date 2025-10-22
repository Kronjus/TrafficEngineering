# ALINEA Configuration Example

This document provides an example configuration for the simplified ALINEA implementation based on a 6-cell motorway with a lane drop from 3→1 lanes at cell 5.

## Network Parameters

Based on the problem statement:
- Free-flow speed: v_f = 100 km/h
- Maximum flow: q_max = 2000 veh/h/lane
- Jam density: ρ_jam = 180 veh/km/lane
- METANET parameters: τ = 22 s, ν = 15 km²/h, κ = 10 veh/km/lane, δ = 1.4
- Lane configuration: 3 lanes → 3 lanes → 3 lanes → 3 lanes → 1 lane → 1 lane (cells 0-5)
- Lane drop location: Between cell 4 and cell 5 (3→1 lanes)

## Critical Density Calculation

For ALINEA, we need the critical density:

**ρ_cr = q_max / v_f = 2000 / 100 = 20.0 veh/km/lane**

This is the density at which capacity is reached on the motorway.

## Recommended ALINEA Configuration

### Measurement Cell Selection

The user wants to measure **upstream of the lane drop**:
- Lane drop occurs at cell 5 (1-based numbering from problem statement)
- Measure at cell 4 (upstream of drop) in 1-based numbering
- Convert to 0-based indexing: **measurement_cell = 3**

### Control Gain Selection

For a severe lane drop (3→1), recommended K_I range: **100-300 veh/h per (veh/km/lane)**

Starting recommendation: **K_I = 200.0**

This provides moderately aggressive control suitable for preventing congestion at the bottleneck.

### Example Configuration (METANET)

```python
from metanet_simulation import (
    build_uniform_metanet_mainline,
    METANETSimulation,
    METANETOnRampConfig,
)

# Build 6-cell mainline with lane drop at cell 5 (0-based: cell 4)
cells = build_uniform_metanet_mainline(
    num_cells=6,
    cell_length_km=0.5,  # Adjust to your network
    lanes=[3, 3, 3, 3, 1, 1],  # Lane drop from 3→1 at cell 4 (0-based)
    free_flow_speed_kmh=100.0,
    jam_density_veh_per_km_per_lane=180.0,
    tau_s=22.0,
    nu=15.0,
    kappa=10.0,
    delta=1.4,
    capacity_veh_per_hour_per_lane=2000.0,
)

# Ramp configuration with simplified ALINEA
# Assumes ramp merges at cell 2 (0-based) - adjust target_cell as needed
ramp = METANETOnRampConfig(
    target_cell=2,  # Adjust to your ramp location (0-based)
    arrival_rate_profile=ramp_demand,  # Your demand profile
    alinea_enabled=True,
    alinea_gain=200.0,  # Control gain K_I (veh/h per veh/km/lane)
    alinea_measurement_cell=3,  # Measure upstream of drop (0-based: cell 4 in 1-based)
)

# Create and run simulation
sim = METANETSimulation(
    cells=cells,
    time_step_hours=0.05,  # 3 minutes
    upstream_demand_profile=upstream_demand,  # Your demand profile
    on_ramps=[ramp],
)

result = sim.run(steps=100)  # Adjust number of steps as needed
```

### Example Configuration (CTM)

```python
from ctm_simulation import (
    build_uniform_mainline,
    CTMSimulation,
    OnRampConfig,
)

# Build 6-cell mainline with lane drop
cells = build_uniform_mainline(
    num_cells=6,
    cell_length_km=0.5,
    lanes=[3, 3, 3, 3, 1, 1],  # Lane drop at cell 4 (0-based)
    free_flow_speed_kmh=100.0,
    congestion_wave_speed_kmh=20.0,
    capacity_veh_per_hour_per_lane=2000.0,
    jam_density_veh_per_km_per_lane=180.0,
)

# Ramp with simplified ALINEA
ramp = OnRampConfig(
    target_cell=2,  # Adjust to your ramp location (0-based)
    arrival_rate_profile=ramp_demand,
    alinea_enabled=True,
    alinea_gain=200.0,
    alinea_measurement_cell=3,  # Measure upstream of drop (0-based)
)

sim = CTMSimulation(
    cells=cells,
    time_step_hours=0.05,
    upstream_demand_profile=upstream_demand,
    on_ramps=[ramp],
)

result = sim.run(steps=100)
```

## Parameter Tuning Guidelines

### Initial Testing
1. Start with K_I = 200.0
2. Run simulation and observe:
   - Final metering rate (should stay in reasonable range, e.g., -500 to +2000 veh/h)
   - Density at measurement cell (should oscillate around ρ_cr = 20.0)
   - Ramp queue dynamics

### Adjustment Criteria

**If rates oscillate too much or become very negative:**
- Reduce K_I (try 100.0 or 150.0)
- This makes control less aggressive

**If control is too slow to respond:**
- Increase K_I (try 250.0 or 300.0)
- This makes control more aggressive

**If instability occurs (oscillations grow over time):**
- Reduce K_I significantly
- May need to reduce time step or adjust measurement cell

### Auto-Tuning

For automatic gain selection, use the auto-tuning feature:

```python
def evaluate_gain(K: float) -> dict:
    """Evaluate a candidate gain value."""
    # Create test simulation with gain K
    test_ramp = METANETOnRampConfig(
        target_cell=2,
        arrival_rate_profile=ramp_demand,
        alinea_enabled=True,
        alinea_gain=K,
        alinea_measurement_cell=3,
    )
    
    test_sim = METANETSimulation(
        cells=cells,
        time_step_hours=0.05,
        upstream_demand_profile=upstream_demand,
        on_ramps=[test_ramp],
    )
    
    result = test_sim.run(steps=100)
    
    # Calculate RMSE from critical density
    rho_cr = 20.0
    densities = result.densities["cell_3"]  # Measurement cell
    squared_errors = [(d - rho_cr) ** 2 for d in densities]
    rmse = (sum(squared_errors) / len(squared_errors)) ** 0.5
    
    return {'rmse': rmse}

# Run auto-tuning
best_K = sim.auto_tune_alinea_gain(
    ramp_index=0,
    simulator_fn=evaluate_gain,
    tuner_params={
        'K_min': 100.0,
        'K_max': 300.0,
        'n_grid': 20,
        'objective': 'rmse',
        'verbose': True,
    },
)

print(f"Optimal gain: {best_K:.1f}")
```

## Expected Behavior

With the recommended configuration (K_I = 200.0, measurement at cell 3):

1. **When density < ρ_cr (20.0)**:
   - Metering rate increases (more vehicles released)
   - Helps maintain flow through bottleneck

2. **When density ≈ ρ_cr (20.0)**:
   - Metering rate stable (system near equilibrium)
   - Density oscillates around critical value

3. **When density > ρ_cr (20.0)**:
   - Metering rate decreases (fewer vehicles released)
   - Can become negative (physically means no release)
   - Prevents congestion at bottleneck

## Important Notes

1. **Negative Rates**: The simplified ALINEA can produce negative rates when density >> ρ_cr. This is mathematically correct but physically meaningless. In practice, negative rates should be interpreted as "no vehicles released" (rate = 0).

2. **1-Based vs 0-Based Indexing**: 
   - Problem statement uses 1-based cell numbering (cells 1-6)
   - Python implementation uses 0-based indexing (cells 0-5)
   - Always convert when configuring: `0_based_index = 1_based_number - 1`

3. **Measurement Cell Choice**:
   - For lane drops, measure **upstream** of the drop
   - This anticipates congestion before it reaches the bottleneck
   - In this example: measure cell 4 (1-based) = cell 3 (0-based)

4. **Time Step Selection**:
   - Smaller time steps (e.g., 0.05 h = 3 min) provide more accurate control
   - Larger time steps reduce computational cost but may miss dynamics
   - Recommended: 0.05-0.1 hours for motorway applications

## References

- Problem statement: Network with v_f=100 km/h, q_max=2000 veh/h/lane, 3→1 lane drop
- Recommended K_I range: 100-300 for severe bottlenecks
- Critical density: ρ_cr = q_max / v_f = 20.0 veh/km/lane
- ALINEA simplified law: r(k) = r(k-1) + K_I · (ρ_cr - ρ_m(k-1))
