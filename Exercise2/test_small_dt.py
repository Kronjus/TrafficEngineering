"""Test METANET with smaller time step."""

from metanet_simulation import (
    METANETSimulation,
    build_uniform_metanet_mainline,
)

# Create cells
cells = build_uniform_metanet_mainline(
    num_cells=3,
    cell_length_km=0.5,
    lanes=3,
    free_flow_speed_kmh=100.0,
    jam_density_veh_per_km_per_lane=160.0,
    initial_density_veh_per_km_per_lane=20.0,
)

# Test with smaller time step
dt = 0.001  # 3.6 seconds, respects CFL
steps = 200  # Same total time as before

sim = METANETSimulation(
    cells=cells,
    time_step_hours=dt,
    upstream_demand_profile=3000.0,
    downstream_supply_profile=5000.0,
)

result = sim.run(steps=steps)

print(f"Time step: {dt} hours ({dt*3600:.1f} seconds)")
print(f"CFL limit: {0.5/100:.4f} hours ({0.5/100*3600:.1f} seconds)")
print(f"Steps: {steps}")
print(f"Total time: {dt*steps} hours")

print(f"\nDensities (cell_0, every 20 steps):")
for i in range(0, min(201, len(result.densities['cell_0'])), 20):
    print(f"  Step {i:3d}: {result.densities['cell_0'][i]:.2f} veh/km/lane")

print(f"\nFlows (cell_0, every 20 steps):")
for i in range(0, min(200, len(result.flows['cell_0'])), 20):
    print(f"  Step {i:3d}: {result.flows['cell_0'][i]:.2f} veh/h")

# Calculate VKT
total_vkt = 0.0
for cell in cells:
    cell_name = cell.name
    flows = result.flows[cell_name]
    for flow in flows:
        total_vkt += flow * dt * cell.length_km

print(f"\nTotal VKT: {total_vkt:.2f} veh*km")
