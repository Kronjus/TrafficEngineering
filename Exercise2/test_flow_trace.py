"""Trace flows through the simulation"""

from metanet_simulation import (
    METANETSimulation,
    build_uniform_metanet_mainline,
)

# Simple 2-cell system
cells = build_uniform_metanet_mainline(
    num_cells=2,
    cell_length_km=1.0,
    lanes=2,
    free_flow_speed_kmh=100.0,
    jam_density_veh_per_km_per_lane=150.0,
    initial_density_veh_per_km_per_lane=[5.0, 5.0],
)

upstream_demand = 1000.0  # smaller demand for clarity

sim = METANETSimulation(
    cells=cells,
    time_step_hours=0.01,
    upstream_demand_profile=upstream_demand,
    downstream_supply_profile=10000.0,
)

result = sim.run(steps=2)

dt = 0.01

print("Flow Trace Analysis")
print("="*70)
print("\nInitial state:")
print(f"  Cell 0: {result.densities['cell_0'][0]:.2f} veh/km/lane = {result.densities['cell_0'][0] * 2:.2f} veh")
print(f"  Cell 1: {result.densities['cell_1'][0]:.2f} veh/km/lane = {result.densities['cell_1'][0] * 2:.2f} veh")
print(f"  Total: {(result.densities['cell_0'][0] + result.densities['cell_1'][0]) * 2:.2f} veh")

for step in range(2):
    print(f"\n{'='*70}")
    print(f"STEP {step}:")
    print(f"{'='*70}")
    
    # State at start of step
    veh_0_start = result.densities['cell_0'][step] * 2
    veh_1_start = result.densities['cell_1'][step] * 2
    total_start = veh_0_start + veh_1_start
    
    # Flows during step
    flow_0_out = result.flows['cell_0'][step]  # veh/h
    flow_1_out = result.flows['cell_1'][step]  # veh/h
    
    veh_0_out = flow_0_out * dt
    veh_1_out = flow_1_out * dt
    
    # Boundary flows
    boundary_in = upstream_demand * dt
    
    # State at end of step
    veh_0_end = result.densities['cell_0'][step+1] * 2
    veh_1_end = result.densities['cell_1'][step+1] * 2
    total_end = veh_0_end + veh_1_end
    
    print(f"\nState at start:")
    print(f"  Cell 0: {veh_0_start:.2f} veh")
    print(f"  Cell 1: {veh_1_start:.2f} veh")
    print(f"  Total: {total_start:.2f} veh")
    
    print(f"\nFlows during step:")
    print(f"  Boundary -> Cell 0: {boundary_in:.2f} veh")
    print(f"  Cell 0 -> Cell 1: {veh_0_out:.2f} veh")
    print(f"  Cell 1 -> Sink: {veh_1_out:.2f} veh")
    
    print(f"\nExpected state at end:")
    print(f"  Cell 0: {veh_0_start:.2f} + {boundary_in:.2f} - {veh_0_out:.2f} = {veh_0_start + boundary_in - veh_0_out:.2f} veh")
    print(f"  Cell 1: {veh_1_start:.2f} + {veh_0_out:.2f} - {veh_1_out:.2f} = {veh_1_start + veh_0_out - veh_1_out:.2f} veh")
    
    print(f"\nActual state at end:")
    print(f"  Cell 0: {veh_0_end:.2f} veh")
    print(f"  Cell 1: {veh_1_end:.2f} veh")
    print(f"  Total: {total_end:.2f} veh")
    
    print(f"\nErrors:")
    print(f"  Cell 0 error: {abs(veh_0_end - (veh_0_start + boundary_in - veh_0_out)):.2f} veh")
    print(f"  Cell 1 error: {abs(veh_1_end - (veh_1_start + veh_0_out - veh_1_out)):.2f} veh")
    print(f"  Total error: {abs(total_end - (total_start + boundary_in - veh_1_out)):.2f} veh")

