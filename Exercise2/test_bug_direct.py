"""
Direct test to demonstrate the same-step exit bug.

The bug is in lines 777-784 where max_sending includes same-step inflow:
    max_sending = (prev_current_vehicles + inflows[idx-1] * self.dt) / self.dt

This test creates a scenario where:
1. Cell 0 has some initial vehicles
2. Cell 0 receives inflow in a time step  
3. We check if the sending from cell 0 can exceed what was there at the start

The bug allows sending = (initial + inflow), but correct behavior is sending = initial only.
"""

from metanet_simulation import (
    METANETSimulation,
    build_uniform_metanet_mainline,
)

# Create a simple 2-cell chain
cells = build_uniform_metanet_mainline(
    num_cells=2,
    cell_length_km=1.0,  # 1 km cells
    lanes=2,
    free_flow_speed_kmh=100.0,
    jam_density_veh_per_km_per_lane=150.0,
    initial_density_veh_per_km_per_lane=[5.0, 5.0],  # Small initial density
)

# Large upstream demand
upstream_demand = 5000.0  # veh/h

sim = METANETSimulation(
    cells=cells,
    time_step_hours=0.01,  # 0.01 hours = 36 seconds
    upstream_demand_profile=upstream_demand,
    downstream_supply_profile=10000.0,  # Ample downstream
)

result = sim.run(steps=5)

dt = 0.01  # hours

print("="*70)
print("SAME-STEP EXIT BUG DEMONSTRATION")
print("="*70)

for step in range(len(result.flows["cell_0"])):
    # State at beginning of step
    density_start = result.densities["cell_0"][step]
    vehicles_start = density_start * cells[0].length_km * cells[0].lanes
    
    # Outflow during this step
    outflow = result.flows["cell_0"][step]
    vehicles_out = outflow * dt
    
    # Theoretical maximum based on state at start (without bug)
    max_possible_without_bug = vehicles_start / dt
    
    print(f"\nStep {step}:")
    print(f"  Vehicles at start: {vehicles_start:.2f}")
    print(f"  Outflow rate: {outflow:.2f} veh/h")
    print(f"  Vehicles leaving: {vehicles_out:.2f}")
    print(f"  Max possible (no bug): {max_possible_without_bug:.2f} veh/h")
    
    # The bug manifests when outflow exceeds what was available at start
    # (accounting for what could physically leave given the time step)
    if outflow > max_possible_without_bug + 1.0:  # +1 for tolerance
        print(f"  *** BUG DETECTED: Outflow exceeds initial availability!")
        print(f"      This indicates same-step inflow is being counted as available to exit")

# Also check conservation
print("\n" + "="*70)
print("CONSERVATION CHECK")
print("="*70)

for step in range(1, len(result.flows["cell_0"])):
    # Total system vehicles at step-1 and step
    total_prev = sum(
        result.densities[cell.name][step-1] * cell.length_km * cell.lanes
        for cell in cells
    )
    total_curr = sum(
        result.densities[cell.name][step] * cell.length_km * cell.lanes
        for cell in cells
    )
    
    # Flows
    inflow = upstream_demand  # constant
    outflow = result.flows["cell_1"][step-1]  # outflow from last cell
    
    # Expected change
    expected_change = dt * (inflow - outflow)
    actual_change = total_curr - total_prev
    
    error = abs(actual_change - expected_change)
    
    print(f"\nStep {step-1} -> {step}:")
    print(f"  Total vehicles: {total_prev:.2f} -> {total_curr:.2f}")
    print(f"  Expected change: {expected_change:.2f}")
    print(f"  Actual change: {actual_change:.2f}")
    print(f"  Error: {error:.2f}")
    
    if error > 1.0:
        print(f"  *** CONSERVATION VIOLATION!")

