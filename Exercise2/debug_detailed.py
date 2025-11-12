"""Add debug logging to METANET simulation."""

from metanet_simulation import (
    METANETSimulation,
    METANETCellConfig,
    exponential_equilibrium_speed,
)


# Monkey patch the run method to add logging
original_run = METANETSimulation.run

def debug_run(self, steps):
    """Run with debug logging."""
    densities = [cell.initial_density_veh_per_km_per_lane for cell in self.cells]
    speeds = []
    for idx, cell in enumerate(self.cells):
        if cell.initial_speed_kmh > 0:
            speeds.append(cell.initial_speed_kmh)
        else:
            v_eq = self.equilibrium_speed(
                densities[idx],
                cell.free_flow_speed_kmh,
                cell.critical_density_veh_per_km_per_lane,
            )
            speeds.append(max(1.0, v_eq))
    
    print("Initial state:")
    for i, (cell, d, v) in enumerate(zip(self.cells, densities, speeds)):
        print(f"  Cell {i}: density={d:.2f}, speed={v:.2f}")
    
    # Run just first 2 steps with detailed logging
    for step in range(min(2, steps)):
        print(f"\n=== Step {step} ===")
        
        # Compute flows
        for idx, cell in enumerate(self.cells):
            if idx == 0:
                upstream_demand = 3000.0  # Hard-coded for this test
                receiving = self._compute_receiving(densities[idx], cell)
                mainline_flow_in = min(upstream_demand, receiving)
                print(f"  Cell {idx} inflow: min({upstream_demand:.2f}, {receiving:.2f}) = {mainline_flow_in:.2f}")
            else:
                potential_flow = densities[idx-1] * speeds[idx-1] * self.cells[idx-1].lanes
                receiving = self._compute_receiving(densities[idx], cell)
                mainline_flow_in = min(potential_flow, receiving)
                print(f"  Cell {idx} inflow: min({potential_flow:.2f}, {receiving:.2f}) = {mainline_flow_in:.2f}")
                print(f"    (from cell {idx-1}: rho={densities[idx-1]:.2f} * v={speeds[idx-1]:.2f} * lanes={self.cells[idx-1].lanes})")
        
    # Call original
    return original_run(self, steps)

METANETSimulation.run = debug_run

# Create simple simulation
cell = METANETCellConfig(
    name="cell_0",
    length_km=0.5,
    lanes=3,
    free_flow_speed_kmh=100.0,
    jam_density_veh_per_km_per_lane=160.0,
    critical_density_veh_per_km_per_lane=33.5,
    initial_density_veh_per_km_per_lane=20.0,
)

sim = METANETSimulation(
    cells=[cell],
    time_step_hours=0.1,
    upstream_demand_profile=3000.0,
    downstream_supply_profile=5000.0,
)

result = sim.run(steps=3)
print("\n\nResult:")
print(f"Densities: {[f'{d:.2f}' for d in result.densities['cell_0'][:4]]}")
print(f"Flows: {[f'{f:.2f}' for f in result.flows['cell_0'][:3]]}")
