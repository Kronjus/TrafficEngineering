# Exercise 2 – CTM playground

This directory now contains a small but fully-featured Cell Transmission Model
(CTM) simulator.  The goal is to offer a light-weight sandbox where you can
experiment with heterogeneous mainline cells, on-ramp interactions and simple
metering strategies without pulling in a large external framework.

## Quick start

```bash
python Exercise2/ctm_simulation.py
```

Running the module directly executes `run_basic_scenario`, a three-cell freeway
with an on-ramp.  The script prints the head of a pandas `DataFrame` when
pandas is installed; otherwise the raw Python dictionaries are shown.

To use the simulator programmatically:

```python
from Exercise2.ctm_simulation import (
    build_uniform_mainline,
    CTMSimulation,
    OnRampConfig,
)

cells = build_uniform_mainline(
    num_cells=4,
    cell_length_km=0.5,
    lanes=[3, 3, 3, 2],
    free_flow_speed_kmh=100.0,
    congestion_wave_speed_kmh=20.0,
    capacity_veh_per_hour_per_lane=2200.0,
    jam_density_veh_per_km_per_lane=160.0,
)

on_ramp = OnRampConfig(
    target_cell=2,
    arrival_rate_profile=lambda step: 400 if step < 12 else 150,
    mainline_priority=0.7,
)

sim = CTMSimulation(
    cells=cells,
    time_step_hours=0.1,
    upstream_demand_profile=[1800.0] * 24,
    downstream_supply_profile=1900.0,
    on_ramps=[on_ramp],
)

result = sim.run(steps=24)
```

Use `SimulationResult` helpers to inspect the outcome:

```python
result.time_vector()       # -> list of sample times (hours)
result.interval_vector()   # -> list of (start, end) tuples for flows
result.densities["cell_0"] # -> density trace for the first cell

# Optional pandas integration
result.to_dataframe()
```

The helper `build_uniform_mainline` is convenient for creating quick scenarios,
but you can also instantiate `CellConfig` objects manually when you need full
control over individual cell parameters.