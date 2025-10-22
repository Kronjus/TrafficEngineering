# ALINEA Simplified Implementation

This document describes the simplified ALINEA ramp metering controller implementation.

## Summary

The ALINEA implementation has been simplified to use only the pure incremental control law:

**r_i(k) = r_i(k-1) + K_I · (ρ_cr - ρ_m(k-1))**

Key changes:
1. **Removed min/max rate clamping** - Control law operates without bounds
2. **Removed initial meter rate parameter** - r(0) defaults to 0.0
3. **Removed target density parameter** - Uses critical density ρ_cr = q_max / v_f
4. **Retained configurable measurement cell** - Choose which cell's density is used for control
5. **Retained automatic gain tuning** - Find optimal gain K_I using simulation-based optimization

## Simplified Control Law

### Description
The simplified ALINEA uses only the pure incremental control law without any clamping:

**r_i(k) = r_i(k-1) + K_I · (ρ_cr - ρ_m(k-1))**

where:
- **r_i(k)**: metering rate at time step k (veh/h)
- **K_I**: control gain (veh/h per veh/km/lane)
- **ρ_cr**: critical density = q_max / v_f (veh/km/lane)
- **ρ_m(k-1)**: measured density at previous time step (veh/km/lane)

### Key Features
- **No clamping**: Rates can become negative when density >> ρ_cr
- **No initial rate parameter**: r(0) = 0.0 by default
- **Uses critical density**: ρ_cr computed from cell capacity and free-flow speed
- **Simple configuration**: Only K_I and measurement cell needed

### Usage

```python
from metanet_simulation import METANETOnRampConfig, METANETSimulation

ramp = METANETOnRampConfig(
    target_cell=2,                    # Ramp merges at cell 2
    arrival_rate_profile=800.0,
    alinea_enabled=True,
    alinea_gain=200.0,                # Control gain K_I
    alinea_measurement_cell=3,        # Optional: measure at different cell
)
```

### Parameters
- `alinea_gain`: Control gain K_I (veh/h per veh/km/lane)
  - Higher values → more aggressive control
  - Typical range: 100-300 for severe bottlenecks
  - Default: 200.0

- `alinea_measurement_cell`: `Optional[Union[int, str]]`
  - Cell index (int) or cell name (str) to use for density measurement
  - Default: `None` (uses `target_cell`)
  - Can specify any valid cell in the simulation

### Deprecated Parameters
The following parameters are retained for backward compatibility but **ignored** when `alinea_enabled=True`:
- `alinea_target_density` - Use ρ_cr = q_max / v_f instead
- `alinea_min_rate` - No clamping applied
- `alinea_max_rate` - No clamping applied
- `meter_rate_veh_per_hour` - r(0) = 0.0 by default

### Example
```python
# Simplified ALINEA with default measurement cell
ramp = METANETOnRampConfig(
    target_cell=1,
    arrival_rate_profile=1000.0,
    alinea_enabled=True,
    alinea_gain=200.0,
)

# Simplified ALINEA with custom measurement cell
ramp = METANETOnRampConfig(
    target_cell=1,
    arrival_rate_profile=1000.0,
    alinea_enabled=True,
    alinea_gain=200.0,
    alinea_measurement_cell=2,  # Measure one cell downstream
)
```

## Configurable Measurement Cell

### Description
By default, ALINEA measures the density at the cell where the ramp merges (target_cell). You can measure density at any other cell, which is useful for:
- Anticipating downstream congestion
- Measuring at bottleneck locations upstream of the drop
- Multi-ramp coordination scenarios

### Usage
```python
# Measure at downstream cell
ramp = METANETOnRampConfig(
    target_cell=2,                    # Ramp merges at cell 2
    arrival_rate_profile=800.0,
    alinea_enabled=True,
    alinea_gain=200.0,
    alinea_measurement_cell=3,        # Measure density at cell 3 (downstream)
)

# Measure at named cell
ramp = METANETOnRampConfig(
    target_cell="merge_cell",
    arrival_rate_profile=800.0,
    alinea_enabled=True,
    alinea_gain=200.0,
    alinea_measurement_cell="bottleneck_cell",  # Measure at named cell
)
```

## Automatic Gain Tuning

### Description
The `auto_tune_alinea_gain()` method automatically finds the optimal gain K_I by:
- Performing grid search over candidate gain values
- Running simulations with each candidate gain
- Evaluating performance using a specified objective function (e.g., RMSE, delay)
- Selecting the gain that minimizes the objective

### Usage

```python
from metanet_simulation import METANETSimulation, METANETOnRampConfig

# Create simulation with ALINEA ramp
sim = METANETSimulation(
    cells=cells,
    time_step_hours=0.05,
    upstream_demand_profile=4000.0,
    on_ramps=[ramp],
)

# Define evaluation function
def evaluate_gain(K: float) -> dict:
    """Simulate with given gain and return metrics."""
    # Create test simulation with gain K
    test_ramp = METANETOnRampConfig(
        target_cell=2,
        arrival_rate_profile=900.0,
        alinea_enabled=True,
        alinea_gain=K,  # Test this gain
    )
    test_sim = METANETSimulation(
        cells=test_cells,
        time_step_hours=0.05,
        upstream_demand_profile=4000.0,
        on_ramps=[test_ramp],
    )
    result = test_sim.run(steps=40)
    
    # Compute objective (RMSE from critical density)
    rho_cr = 2200.0 / 100.0  # capacity / free_flow_speed
    densities = result.densities["cell_2"]
    squared_errors = [(d - rho_cr) ** 2 for d in densities]
    rmse = (sum(squared_errors) / len(squared_errors)) ** 0.5
    
    return {'rmse': rmse}

# Run auto-tuning
tuner_params = {
    'K_min': 50.0,       # Minimum gain to test
    'K_max': 300.0,      # Maximum gain to test
    'n_grid': 20,        # Number of grid points
    'objective': 'rmse', # Metric to minimize
    'verbose': True,     # Print progress
}

best_K = sim.auto_tune_alinea_gain(
    ramp_index=0,
    simulator_fn=evaluate_gain,
    tuner_params=tuner_params,
)

print(f"Optimal gain: {best_K}")
```

### Method Signature
```python
def auto_tune_alinea_gain(
    self,
    ramp_index: int,
    simulator_fn: Callable[[float], Dict[str, float]],
    tuner_params: Optional[Dict] = None,
) -> float
```

### Parameters
- `ramp_index`: Index of the ramp to tune (in `on_ramps` list)
- `simulator_fn`: Callable that takes gain K and returns metrics dict
- `tuner_params`: Dictionary with tuning configuration:
  - `K_min`: Minimum gain (default: 0.1)
  - `K_max`: Maximum gain (default: 100.0)
  - `n_grid`: Number of test points (default: 20)
  - `objective`: Metric name to minimize (default: 'rmse')
  - `verbose`: Print progress (default: False)

### Simulator Function
The simulator function must:
1. Accept a single argument: gain value (float)
2. Run a simulation with that gain
3. Return a dictionary of performance metrics

Example:
```python
def evaluate_gain(K: float) -> dict:
    # Setup test simulation
    test_cells = build_uniform_metanet_mainline(
        num_cells=4,
        cell_length_km=0.5,
        lanes=3,
        free_flow_speed_kmh=100.0,
        jam_density_veh_per_km_per_lane=160.0,
        capacity_veh_per_hour_per_lane=2200.0,
    )
    test_ramp = METANETOnRampConfig(
        target_cell=2,
        arrival_rate_profile=900.0,
        alinea_enabled=True,
        alinea_gain=K,  # Use test gain
    )
    test_sim = METANETSimulation(
        cells=test_cells,
        time_step_hours=0.05,
        upstream_demand_profile=4000.0,
        on_ramps=[test_ramp],
    )
    
    # Run simulation
    result = test_sim.run(steps=50)
    
    # Calculate metrics
    rho_cr = 2200.0 / 100.0  # critical density
    densities = result.densities["cell_2"]
    squared_errors = [(d - rho_cr) ** 2 for d in densities]
    rmse = (sum(squared_errors) / len(squared_errors)) ** 0.5
    
    return {'rmse': rmse}
```

### Objectives
Common objective functions:
- `'rmse'`: Root mean squared error from critical density ρ_cr (good for tracking)
- `'delay'`: Total vehicle delay (good for minimizing queues)
- Custom: Any metric returned by simulator_fn

Note: For simplified ALINEA, use RMSE from ρ_cr rather than an arbitrary target density.

### Full Example
See `alinea_demo.py` Scenario 4 for a complete working example.

## Testing

All new features have comprehensive test coverage:

### Measurement Cell Tests
- `test_alinea_custom_measurement_cell` - Verify custom cell selection
- `test_alinea_measurement_cell_by_name` - Verify cell selection by name
- `test_alinea_default_measurement_cell` - Verify default behavior

### Auto-Tuning Tests
- `test_auto_tune_basic` - Basic tuning with simple objective
- `test_auto_tune_different_objectives` - Multiple objective functions
- `test_auto_tune_invalid_K_range` - Validate parameter bounds
- `test_auto_tune_invalid_ramp_index` - Error handling
- `test_auto_tune_non_alinea_ramp` - Error for non-ALINEA ramps
- `test_auto_tune_with_realistic_simulator` - Simulation-based tuning

Run tests:
```bash
python -m unittest test_metanet_simulation.TestMETANETALINEAControl -v
python -m unittest test_metanet_simulation.TestALINEAAutoTuning -v
```

## Demonstration

Run the updated demo to see all features in action:
```bash
python alinea_demo.py
```

The demo includes four scenarios:
1. Fixed-rate metering (baseline)
2. Simplified ALINEA control
3. ALINEA with custom measurement cell
4. ALINEA with automatic gain tuning

## Backward Compatibility

The simplified ALINEA maintains backward compatibility:
- Deprecated parameters (`alinea_target_density`, `alinea_min_rate`, `alinea_max_rate`, `meter_rate_veh_per_hour`) are retained but **ignored** when `alinea_enabled=True`
- Existing code will run but should be updated to use the simplified API
- The new default gain is 200.0 (changed from 50.0 to match typical range for severe bottlenecks)

## Implementation Notes

### Simplified Control Law Behavior

**Important**: The simplified ALINEA does not clamp rates, which means:
- When ρ_m >> ρ_cr (high congestion), rates will **decrease** and can become **negative**
- When ρ_m << ρ_cr (low density), rates will **increase** without bound
- Negative rates are acceptable mathematically but physically meaningless
- In real systems, negative rates would be treated as zero (no vehicles released)
- For simulation analysis, observe the range of rates to ensure they remain reasonable

**Recommendation**: 
- Choose K_I such that rates remain in reasonable ranges for your scenario
- For typical motorway scenarios: K_I = 100-300 veh/h per (veh/km/lane)
- Monitor rate trajectories during simulation to detect instability
- If rates become extremely negative or positive, reduce K_I

### Measurement Cell Resolution
- During simulation initialization, cell names/indices are resolved to integer indices
- Invalid cell references raise descriptive errors
- Default measurement cell equals target cell (original behavior)

### Auto-Tuning Algorithm
- Simple grid search (robust, predictable)
- No external dependencies (uses built-in Python)
- Easily extended to more sophisticated methods (Bayesian, Nelder-Mead, etc.)

### Performance
- Grid search with n=20 and horizon=50 steps takes ~1-2 seconds
- For production use, consider:
  - Smaller grid for initial tuning
  - Caching simulation results
  - Parallel evaluation of candidates

## Future Enhancements

Potential improvements:
1. **Online adaptive tuning**: Continuously adjust gain during operation
2. **Multi-objective optimization**: Optimize multiple metrics simultaneously
3. **Bayesian optimization**: Fewer evaluations for large parameter spaces
4. **Sensitivity analysis**: Understand robustness to gain variations
5. **Rate-of-change limits**: Constrain how fast metering rate can change

## References

- Papageorgiou, M., et al. (1991). "ALINEA: A local feedback control law for on-ramp metering"
- Smaragdis, E., & Papageorgiou, M. (2003). "Series of new local ramp metering strategies"
- Problem statement requirements document

## Contact

For questions or issues, please open an issue in the repository.
