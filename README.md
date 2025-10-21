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

See [Exercise1/README.md](Exercise1/README.md) for detailed documentation (if available).

### Exercise 2: Cell Transmission Model (CTM) Simulator
A comprehensive traffic flow simulator based on the Cell Transmission Model.

**Key Features:**
- Mainline freeway simulation with heterogeneous cells
- On-ramp merging with queue dynamics
- Ramp metering strategies
- Time-varying demand profiles
- Lane drops and capacity bottlenecks

**Quick Start:**
```bash
# Run basic demo
python Exercise2/ctm_simulation.py

# Run comprehensive examples
python Exercise2/examples_ctm.py

# Interactive notebook
jupyter notebook Exercise2/ctm_simulation.ipynb

# Run tests
python -m unittest Exercise2.test_ctm_simulation
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

## Testing

### CTM Simulator Tests
```bash
cd Exercise2
python -m unittest test_ctm_simulation -v
```

The test suite includes:
- 32 comprehensive unit tests
- Configuration validation
- Flow conservation checks
- On-ramp merging behavior
- Queue dynamics
- Profile handling

All tests pass successfully.

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
| On-ramps | - | ✓ |
| Ramp Metering | - | ✓ |
| Time-varying Demand | - | ✓ |
| Unit Tests | ? | ✓ (32 tests) |
| Examples | ? | ✓ (4 scenarios) |
| Jupyter Notebook | ? | ✓ |

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

### CTM (Exercise 2)
- Daganzo, C. F. (1994). "The cell transmission model: A dynamic representation of highway traffic consistent with the hydrodynamic theory." *Transportation Research Part B*, 28(4), 269-287.
- Daganzo, C. F. (1995). "The cell transmission model, part II: Network traffic." *Transportation Research Part B*, 29(2), 79-93.

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
