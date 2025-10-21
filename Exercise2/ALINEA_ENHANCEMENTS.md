# ALINEA Enhancements

This document describes the new features added to the ALINEA ramp metering controller implementation.

## Summary

Three major enhancements have been added to the ALINEA implementation:

1. **Configurable measurement cell** - Choose which cell's density is used for control
2. **Configurable r limits** - Set min/max bounds on metering rate (already existed, now documented)
3. **Automatic gain tuning** - Find optimal gain K_R using simulation-based optimization

## 1. Configurable Measurement Cell

### Description
By default, ALINEA measures the density at the cell where the ramp merges (target_cell). The new feature allows you to measure density at any other cell, which is useful for:
- Anticipating downstream congestion
- Measuring at bottleneck locations
- Multi-ramp coordination scenarios

### Usage

```python
from metanet_simulation import METANETOnRampConfig, METANETSimulation

ramp = METANETOnRampConfig(
    target_cell=2,                    # Ramp merges at cell 2
    arrival_rate_profile=800.0,
    meter_rate_veh_per_hour=600.0,
    alinea_enabled=True,
    alinea_gain=50.0,
    alinea_target_density=80.0,
    alinea_measurement_cell=3,        # Measure density at cell 3 (downstream)
)
```

### Parameters
- `alinea_measurement_cell`: `Optional[Union[int, str]]`
  - Cell index (int) or cell name (str) to use for density measurement
  - Default: `None` (uses `target_cell`)
  - Can specify any valid cell in the simulation

### Example
```python
# Measure at downstream cell (by index)
ramp = METANETOnRampConfig(
    target_cell=1,
    alinea_measurement_cell=2,  # Measure one cell downstream
    # ... other parameters
)

# Measure at specific cell (by name)
ramp = METANETOnRampConfig(
    target_cell="cell_1",
    alinea_measurement_cell="cell_3",  # Measure at named cell
    # ... other parameters
)
```

## 2. Configurable r Limits (r_min, r_max)

### Description
ALINEA metering rates are automatically clamped to safe bounds to prevent:
- Overly restrictive metering (too low rate causes excessive queuing)
- Ineffective metering (too high rate has no control effect)

### Usage
These parameters were already available in the implementation:

```python
ramp = METANETOnRampConfig(
    target_cell=1,
    arrival_rate_profile=800.0,
    meter_rate_veh_per_hour=600.0,
    alinea_enabled=True,
    alinea_gain=50.0,
    alinea_target_density=80.0,
    alinea_min_rate=240.0,    # Minimum metering rate (veh/h)
    alinea_max_rate=1800.0,   # Maximum metering rate (veh/h)
)
```

### Parameters
- `alinea_min_rate`: `float` (default: 240.0 veh/h)
  - Minimum allowed metering rate
  - Prevents excessive queue buildup
  
- `alinea_max_rate`: `float` (default: 2400.0 veh/h)
  - Maximum allowed metering rate
  - Ensures control remains active

### Control Law
The ALINEA control law with clipping:
```
r(k+1) = r(k) + K_R * (ρ_target - ρ_measured)
r(k+1) = clip(r(k+1), r_min, r_max)
```

## 3. Automatic Gain Tuning

### Description
The new `auto_tune_alinea_gain()` method automatically finds the optimal gain K_R by:
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
    test_sim = create_test_simulation(K)
    result = test_sim.run(steps=40)
    
    # Compute objective (e.g., RMSE from target density)
    rmse = compute_rmse(result.densities["cell_2"], target=80.0)
    
    return {'rmse': rmse, 'delay': compute_delay(result)}

# Run auto-tuning
tuner_params = {
    'K_min': 10.0,      # Minimum gain to test
    'K_max': 90.0,      # Maximum gain to test
    'n_grid': 20,       # Number of grid points
    'objective': 'rmse', # Metric to minimize
    'verbose': True,    # Print progress
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
    test_cells = build_uniform_metanet_mainline(...)
    test_ramp = METANETOnRampConfig(
        ...,
        alinea_gain=K,  # Use test gain
    )
    test_sim = METANETSimulation(...)
    
    # Run simulation
    result = test_sim.run(steps=50)
    
    # Calculate metrics
    densities = result.densities["cell_2"]
    rmse = sqrt(mean((densities - target)**2))
    delay = sum(result.ramp_queues["ramp_2"]) * time_step
    
    return {
        'rmse': rmse,
        'delay': delay,
        'total_queue': sum(result.ramp_queues["ramp_2"])
    }
```

### Objectives
Common objective functions:
- `'rmse'`: Root mean squared error from target density (good for tracking)
- `'delay'`: Total vehicle delay (good for minimizing queues)
- `'throughput'`: Maximize mainline throughput
- Custom: Any metric returned by simulator_fn

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
2. Standard ALINEA control
3. ALINEA with custom measurement cell
4. ALINEA with automatic gain tuning

## Backward Compatibility

All changes are backward compatible:
- Existing code continues to work unchanged
- New parameters have sensible defaults
- Default behavior matches original implementation

## Implementation Notes

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
