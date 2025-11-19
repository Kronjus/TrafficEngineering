# ALINEA Removal and VKT/VHT/Velocity Computation

## Summary of Changes

This update removes the ALINEA ramp metering controller from the simulation codebase and adds reliable methods to compute VKT, VHT, and per-cell velocities from simulation results.

## Changes Made

### 1. Removed ALINEA Controller

**Files Modified:**
- `Exercise2/ctm_simulation.py`
- `Exercise2/metanet_simulation.py`
- `Exercise2/ctm_simulation.ipynb`
- `Exercise2/metanet_simulation.ipynb`

**What was removed:**
- All ALINEA-related fields from `OnRampConfig` and `METANETOnRampConfig`:
  - `alinea_enabled`
  - `alinea_gain`
  - `alinea_target_density`
  - `alinea_measurement_cell` (METANET only)
  - `alinea_min_rate`
  - `alinea_max_rate`
  - `alinea_auto_tune` (METANET only)
  - `alinea_tuner_params` (METANET only)

- ALINEA control methods:
  - `CTMSimulation._apply_alinea_control()`
  - `METANETSimulation._apply_alinea_control()`
  - `METANETSimulation.auto_tune_alinea_gain()`

- ALINEA initialization and control calls in simulation loops

### 2. Enhanced SimulationResult Validation

**File Modified:** `Exercise2/ctm_simulation.py`

**Added comprehensive time-series validation** in `SimulationResult.__post_init__()`:
- Validates that all density series have length `steps` (includes initial sample at t=0)
- Validates that all flow series have length `steps - 1` (aligned with intervals)
- Validates that all ramp_queue series have length `steps`
- Validates that all ramp_flow series have length `steps - 1`
- Provides clear error messages when series lengths are inconsistent

### 3. Added VKT/VHT Computation Method

**File Modified:** `Exercise2/ctm_simulation.py`

**New method:** `SimulationResult.compute_vkt_vht(cells) -> Tuple[float, float]`

Computes Vehicle Kilometers Traveled (VKT) and Vehicle Hours Traveled (VHT):

```python
# Example usage
result = sim.run(steps=100)
vkt, vht = result.compute_vkt_vht(sim.cells)
print(f"VKT: {vkt:.1f} veh-km, VHT: {vht:.1f} veh-h")
```

**Formulas:**
- VKT = Σ (flow × dt × cell_length) over all cells and intervals
- VHT = Σ (avg_density × lanes × cell_length × dt) over all cells and intervals
  - Where avg_density = 0.5 × (ρ_t + ρ_{t+1}) (trapezoidal rule)

**Units:**
- Flow: veh/h
- Density: veh/km/lane
- Cell length: km
- Time step (dt): hours
- Result: VKT in veh-km, VHT in veh-h

### 4. Added Velocity Computation Method

**File Modified:** `Exercise2/ctm_simulation.py`

**New method:** `SimulationResult.compute_velocity_series(cells) -> Dict[str, List[float]]`

Computes per-cell velocity time series (km/h) aligned with flow intervals:

```python
# Example usage
velocities = result.compute_velocity_series(sim.cells)
for cell_name, vel_series in velocities.items():
    avg_vel = sum(vel_series) / len(vel_series)
    print(f"{cell_name}: avg velocity = {avg_vel:.1f} km/h")
```

**Formula for each interval:**
- avg_density_total = 0.5 × (ρ_t + ρ_{t+1}) × lanes  (veh/km total)
- velocity = flow / avg_density_total  (km/h)
- If avg_density_total == 0, velocity is set to 0.0

**Properties:**
- Returns velocity series for each mainline cell
- Each series has length `steps - 1` (aligned with flows/intervals)
- Handles zero-density cases safely (returns 0.0)
- Useful for identifying congestion, analyzing traffic patterns, etc.

## Migration Guide

### Updating On-Ramp Configurations

**Before (with ALINEA):**
```python
ramp = OnRampConfig(
    target_cell=2,
    arrival_rate_profile=ramp_demand,
    meter_rate_veh_per_hour=60000.0,  # Initial rate for ALINEA
    alinea_enabled=True,
    alinea_target_density=20.0,
    alinea_gain=40.0,
    alinea_min_rate=0.0,
    alinea_max_rate=2000.0,
)
```

**After (without ALINEA):**
```python
ramp = OnRampConfig(
    target_cell=2,
    arrival_rate_profile=ramp_demand,
    meter_rate_veh_per_hour=600.0,  # Fixed metering rate
    # Or None for unmetered ramp
)
```

### Computing VKT and VHT

**Before (manual calculation):**
```python
# Manual stacking and calculation required
# Prone to alignment errors
```

**After (use built-in method):**
```python
result = sim.run(steps=500)
vkt, vht = result.compute_vkt_vht(sim.cells)
print(f"VKT: {vkt:.1f}, VHT: {vht:.1f}")
```

### Computing Velocities

**New capability:**
```python
result = sim.run(steps=500)
velocities = result.compute_velocity_series(sim.cells)

# Analyze congestion
for cell_name, vel_series in velocities.items():
    congested = sum(1 for v in vel_series if 0 < v < 50)
    print(f"{cell_name}: {congested} congested intervals")
```

## Impact on Results

### VKT Changes
Without ALINEA control, VKT values may differ from previous runs because:
1. Ramp metering rates are now fixed (or unmetered) rather than dynamically adjusted
2. Previous VKT calculations may have had alignment issues (fixed by validation)
3. The simulation behavior is now deterministic (no feedback control)

### Expected Behavior
- Fixed metering: More predictable but potentially higher queuing
- Unmetered ramps: Higher flows but may cause mainline congestion
- VKT should be stable across runs with same parameters

## Testing

Run the test scripts to verify the changes:

```bash
# Basic functionality test
python3 /tmp/test_simulations.py

# VKT calculation test
python3 /tmp/test_vkt_scenario_b.py

# Usage example
python3 /tmp/example_vkt_vht_velocity.py
```

All tests should pass, demonstrating:
1. Simulations run without ALINEA parameters
2. Series length validation catches errors
3. VKT/VHT methods produce reasonable values
4. Velocity computation handles zero-density cases
5. Both CTM and METANET simulations work correctly

## Notes

- The `SimulationResult` class is shared between CTM and METANET, so improvements apply to both
- Series length validation helps catch notebook stacking issues early
- VKT/VHT formulas follow standard traffic engineering definitions
- Velocity computation uses trapezoidal rule for density averaging
- All methods are documented with comprehensive docstrings
