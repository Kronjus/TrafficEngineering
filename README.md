# Traffic Engineering

This repository contains traffic engineering projects and exercises, including computer vision-based traffic analysis and traffic flow simulation.

## Repository Structure

### Exercise 1: Object Detection and Tracking
Computer vision-based traffic analysis using YOLO for vehicle detection and tracking.

**Key Features:**
- Video-based vehicle detection and counting
- Georeference mapping (pixel to world coordinates)
- Region of Interest (ROI) selection
- Object tracking across frames

**Technologies:** YOLOv8, OpenCV, PyTorch, GeoPandas

See [Exercise1/README.md](Exercise1/README.md) for detailed documentation.

### Exercise 2: Traffic Flow Simulation (CTM & METANET)
Comprehensive macroscopic traffic flow simulators with CTM and METANET models.

**Key Features:**
- Cell Transmission Model (CTM) - First-order density model
- METANET - Second-order density and speed model
- On-ramp merging with queue dynamics
- ALINEA feedback ramp metering control
- Ramp metering strategies
- Time-varying demand profiles
- Lane drops and capacity bottlenecks
- Compatible APIs for model comparison

**Quick Start:**
```bash
# Run basic CTM demo
python Exercise2/ctm_simulation.py

# Run basic METANET demo
python Exercise2/metanet_simulation.py

# Run ALINEA ramp metering demo
python Exercise2/alinea_demo.py

# Run comprehensive examples
python Exercise2/examples_ctm.py
python Exercise2/examples_metanet.py

# Interactive notebook
jupyter notebook Exercise2/ctm_simulation.ipynb

# Run tests (91 tests total)
python -m unittest Exercise2.test_ctm_simulation Exercise2.test_metanet_simulation
```

See [Exercise2/README.md](Exercise2/README.md) for complete documentation.

## Installation

### Basic Dependencies
```bash
pip install -r requirements.txt
```

### Optional Dependencies

For CTM visualizations:
```bash
pip install matplotlib
```

For interactive notebooks:
```bash
pip install jupyter
```

For pandas integration (CTM results):
```bash
pip install pandas
```

## Getting Started

### CTM Simulator Quick Example

```python
from Exercise2.ctm_simulation import (
    build_uniform_mainline,
    CTMSimulation,
    OnRampConfig,
)

# Create 5-cell freeway
cells = build_uniform_mainline(
    num_cells=5,
    cell_length_km=0.5,
    lanes=3,
    free_flow_speed_kmh=100.0,
    congestion_wave_speed_kmh=20.0,
    capacity_veh_per_hour_per_lane=2200.0,
    jam_density_veh_per_km_per_lane=160.0,
)

# Add on-ramp at cell 2
on_ramp = OnRampConfig(
    target_cell=2,
    arrival_rate_profile=600.0,  # veh/h
    mainline_priority=0.7,
)

# Run simulation
sim = CTMSimulation(
    cells=cells,
    time_step_hours=0.1,
    upstream_demand_profile=5000.0,
    on_ramps=[on_ramp],
)

result = sim.run(steps=20)
print(f"Max queue: {max(result.ramp_queues['ramp_2'])} vehicles")
```

### METANET Simulator Quick Example

```python
from Exercise2.metanet_simulation import (
    build_uniform_metanet_mainline,
    METANETSimulation,
    METANETOnRampConfig,
)

# Create 5-cell freeway with METANET parameters
cells = build_uniform_metanet_mainline(
    num_cells=5,
    cell_length_km=0.5,
    lanes=3,
    free_flow_speed_kmh=100.0,
    jam_density_veh_per_km_per_lane=160.0,
    tau_s=18.0,     # Relaxation time (seconds)
    nu=60.0,        # Anticipation coefficient (km²/h)
    kappa=40.0,     # Regularization constant (veh/km/lane)
    delta=1.0,      # Equilibrium speed exponent
)

# Add on-ramp (same API as CTM)
on_ramp = METANETOnRampConfig(
    target_cell=2,
    arrival_rate_profile=600.0,
    mainline_priority=0.7,
)

# Run simulation
sim = METANETSimulation(
    cells=cells,
    time_step_hours=0.1,
    upstream_demand_profile=5000.0,
    on_ramps=[on_ramp],
)

result = sim.run(steps=20)
print(f"Max queue: {max(result.ramp_queues['ramp_2'])} vehicles")
```

### ALINEA Ramp Metering Example

```python
from Exercise2.ctm_simulation import (
    build_uniform_mainline,
    CTMSimulation,
    OnRampConfig,
)

cells = build_uniform_mainline(
    num_cells=5,
    cell_length_km=0.5,
    lanes=3,
    free_flow_speed_kmh=100.0,
    congestion_wave_speed_kmh=20.0,
    capacity_veh_per_hour_per_lane=2200.0,
    jam_density_veh_per_km_per_lane=160.0,
)

# On-ramp with ALINEA feedback control
on_ramp = OnRampConfig(
    target_cell=2,
    arrival_rate_profile=800.0,
    meter_rate_veh_per_hour=600.0,  # Initial rate
    alinea_enabled=True,             # Enable ALINEA
    alinea_gain=50.0,                # Control gain
    alinea_target_density=120.0,     # Target density (veh/km/lane)
)

sim = CTMSimulation(
    cells=cells,
    time_step_hours=0.1,
    upstream_demand_profile=5000.0,
    on_ramps=[on_ramp],
)

result = sim.run(steps=50)
print(f"Final metering rate: {on_ramp.meter_rate_veh_per_hour:.0f} veh/h")
```

## Testing

### CTM Simulator Tests
```bash
cd Exercise2
python -m unittest test_ctm_simulation -v
```

The test suite includes:
- 39 comprehensive unit tests (including ALINEA)
- Configuration validation
- Flow conservation checks
- On-ramp merging behavior
- Queue dynamics
- Profile handling
- ALINEA ramp metering control

### METANET Simulator Tests
```bash
cd Exercise2
python -m unittest test_metanet_simulation -v
```

The test suite includes:
- 52 comprehensive unit tests (including ALINEA)
- Speed dynamics validation
- Parameter usage verification
- Conservation checks
- ALINEA ramp metering control

All tests pass successfully (91 total).

## Documentation

- **Exercise 2 (CTM)**: Comprehensive documentation in [Exercise2/README.md](Exercise2/README.md)
- **API Documentation**: Inline docstrings in `Exercise2/ctm_simulation.py`
- **Examples**: Four detailed scenarios in `Exercise2/examples_ctm.py`
- **Interactive Tutorial**: Jupyter notebook at `Exercise2/ctm_simulation.ipynb`

## Features by Exercise

| Feature | Exercise 1 | Exercise 2 |
|---------|-----------|-----------|
| Computer Vision | ✓ | - |
| Object Detection | ✓ | - |
| Tracking | ✓ | - |
| Traffic Simulation | - | ✓ |
| CTM Model | - | ✓ |
| METANET Model | - | ✓ |
| On-ramps | - | ✓ |
| Ramp Metering | - | ✓ |
| ALINEA Control | - | ✓ |
| Time-varying Demand | - | ✓ |
| Unit Tests | - | ✓ (91 tests) |
| Examples | ✓ | ✓ |
| Jupyter Notebook | ✓ | ✓ |

## Technologies

### Exercise 1
- Python 3.8+
- PyTorch / TorchVision
- Ultralytics YOLO
- OpenCV
- GeoPandas, Shapely, PyProj

### Exercise 2
- Python 3.8+
- NumPy (for array operations)
- Matplotlib (optional, for visualizations)
- Pandas (optional, for DataFrame output)
- Jupyter (optional, for notebooks)

## License

See [LICENSE](LICENSE) file for details.

## Contributing

This repository is part of a traffic engineering course. For questions or contributions, please open an issue or contact the repository maintainers.

## References

### Traffic Flow Simulation (Exercise 2)

**CTM:**
- Daganzo, C. F. (1994). "The cell transmission model: A dynamic representation of highway traffic consistent with the hydrodynamic theory." *Transportation Research Part B*, 28(4), 269-287.
- Daganzo, C. F. (1995). "The cell transmission model, part II: Network traffic." *Transportation Research Part B*, 29(2), 79-93.

**METANET:**
- Papageorgiou, M., Blosseville, J. M., & Hadj-Salem, H. (1990). "Modelling and real-time control of traffic flow on the southern part of Boulevard Périphérique in Paris: Part I: Modelling." *Transportation Research Part A*, 24(5), 345-359.
- Papageorgiou, M., Blosseville, J. M., & Hadj-Salem, H. (1990). "Modelling and real-time control of traffic flow on the southern part of Boulevard Périphérique in Paris: Part II: Coordinated on-ramp metering." *Transportation Research Part A*, 24(5), 361-370.

**ALINEA Ramp Metering:**
- Papageorgiou, M., Hadj-Salem, H., & Blosseville, J. M. (1991). "ALINEA: A local feedback control law for on-ramp metering." *Transportation Research Record*, 1320, 58-67.
- Papageorgiou, M., & Kotsialos, A. (2002). "Freeway ramp metering: An overview." *IEEE Transactions on Intelligent Transportation Systems*, 3(4), 271-281.

### Computer Vision (Exercise 1)
- Ultralytics YOLOv8: https://docs.ultralytics.com/
- Buch, N., Velastin, S. A., & Orwell, J. (2011). "A review of computer vision techniques for the analysis of urban traffic." *IEEE Transactions on Intelligent Transportation Systems*, 12(3), 920-939.

## Citation

If you use this code in your research, please cite:

```
@software{traffic_engineering,
  title = {Traffic Engineering: CTM Simulator and Vision Analysis},
  author = {Repository Contributors},
  year = {2024},
  url = {https://github.com/Kronjus/TrafficEngineering}
}
```

See [CITATION.cff](CITATION.cff) for formal citation metadata.
