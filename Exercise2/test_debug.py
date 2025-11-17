from metanet_simulation import (
    METANETSimulation,
    build_uniform_metanet_mainline,
)

# Create a 2-cell chain with the first cell initially empty
cells = build_uniform_metanet_mainline(
    num_cells=2,
    cell_length_km=0.5,
    lanes=2,
    free_flow_speed_kmh=100.0,
    jam_density_veh_per_km_per_lane=150.0,
    initial_density_veh_per_km_per_lane=[0.0, 20.0],  # First cell empty
)

# Apply constant upstream demand
upstream_demand = 1000.0  # veh/h

sim = METANETSimulation(
    cells=cells,
    time_step_hours=0.01,  # Small time step
    upstream_demand_profile=upstream_demand,
    downstream_supply_profile=10000.0,  # Ample downstream capacity
)

result = sim.run(steps=3)

print("Cell 0 densities:", result.densities["cell_0"])
print("Cell 0 flows (outflows):", result.flows["cell_0"])
print("Cell 1 flows (outflows):", result.flows["cell_1"])

# Calculate vehicles in cell 0 at each step
for i, density in enumerate(result.densities["cell_0"]):
    vehicles = density * cells[0].length_km * cells[0].lanes
    print(f"Step {i}: Cell 0 density={density:.2f} veh/km/lane, vehicles={vehicles:.2f}")

# Check if the bug exists
outflow_step_0 = result.flows["cell_0"][0]
print(f"\nOutflow from cell 0 at step 0: {outflow_step_0:.2f} veh/h")
print(f"Expected inflow to cell 0 at step 0: {upstream_demand:.2f} veh/h")
print(f"Vehicles added in step 0: {upstream_demand * 0.01:.2f} vehicles")

if outflow_step_0 > 1.0:
    print("\nBUG DETECTED: Vehicles are exiting in the same step they enter!")
else:
    print("\nNo bug detected (or flow is limited by other factors)")
