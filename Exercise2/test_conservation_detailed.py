"""Detailed conservation check"""

from metanet_simulation import (
    METANETSimulation,
    build_uniform_metanet_mainline,
)

cells = build_uniform_metanet_mainline(
    num_cells=2,
    cell_length_km=1.0,
    lanes=2,
    free_flow_speed_kmh=100.0,
    jam_density_veh_per_km_per_lane=150.0,
    initial_density_veh_per_km_per_lane=[5.0, 5.0],
)

upstream_demand = 5000.0  # veh/h

sim = METANETSimulation(
    cells=cells,
    time_step_hours=0.01,
    upstream_demand_profile=upstream_demand,
    downstream_supply_profile=10000.0,
)

result = sim.run(steps=3)

dt = 0.01

print("Detailed Conservation Analysis")
print("="*70)

for step in range(len(result.flows["cell_0"])):
    print(f"\nStep {step}:")
    print(f"  Cell 0:")
    print(f"    Density: {result.densities['cell_0'][step]:.2f} veh/km/lane")
    print(f"    Vehicles: {result.densities['cell_0'][step] * 1.0 * 2:.2f}")
    if step < len(result.flows["cell_0"]):
        print(f"    Outflow: {result.flows['cell_0'][step]:.2f} veh/h")
    
    print(f"  Cell 1:")
    print(f"    Density: {result.densities['cell_1'][step]:.2f} veh/km/lane")
    print(f"    Vehicles: {result.densities['cell_1'][step] * 1.0 * 2:.2f}")
    if step < len(result.flows["cell_1"]):
        print(f"    Outflow: {result.flows['cell_1'][step]:.2f} veh/h")
    
    total = (result.densities['cell_0'][step] + result.densities['cell_1'][step]) * 1.0 * 2
    print(f"  Total vehicles in system: {total:.2f}")

# Check step-by-step conservation
print("\n" + "="*70)
print("Step-by-step Conservation:")
print("="*70)

for step in range(len(result.flows["cell_0"])):
    # What flows IN to the system (from upstream boundary)
    boundary_inflow = upstream_demand  # veh/h
    
    # What flows OUT of the system (from last cell)
    system_outflow = result.flows["cell_1"][step]  # veh/h
    
    # Net vehicles added this step
    net_added = (boundary_inflow - system_outflow) * dt
    
    # Change in total vehicles
    if step < len(result.densities["cell_0"]) - 1:
        total_before = (result.densities['cell_0'][step] + result.densities['cell_1'][step]) * 1.0 * 2
        total_after = (result.densities['cell_0'][step+1] + result.densities['cell_1'][step+1]) * 1.0 * 2
        actual_change = total_after - total_before
        
        print(f"\nStep {step}:")
        print(f"  Boundary inflow: {boundary_inflow:.2f} veh/h ({boundary_inflow * dt:.2f} veh)")
        print(f"  System outflow: {system_outflow:.2f} veh/h ({system_outflow * dt:.2f} veh)")
        print(f"  Expected net change: {net_added:.2f} veh")
        print(f"  Actual change: {actual_change:.2f} veh")
        print(f"  Error: {abs(actual_change - net_added):.2f} veh")

