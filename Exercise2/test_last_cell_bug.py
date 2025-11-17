"""
Test to show the bug specifically in the last cell outflow calculation.

Lines 857-865 compute:
    max_available_flow = (current_vehicles + inflow_vehicles) / self.dt

This means the last cell's outflow can include vehicles that just entered
in the same time step, violating the temporal discretization.
"""

from metanet_simulation import (
    METANETSimulation,
    build_uniform_metanet_mainline,
)

# Single cell to make it simple
cells = build_uniform_metanet_mainline(
    num_cells=1,
    cell_length_km=1.0,
    lanes=2,
    free_flow_speed_kmh=100.0,
    jam_density_veh_per_km_per_lane=150.0,
    initial_density_veh_per_km_per_lane=10.0,  # 20 vehicles total
)

# Constant upstream demand
upstream_demand = 2000.0  # veh/h

sim = METANETSimulation(
    cells=cells,
    time_step_hours=0.01,  # 0.01 hours
    upstream_demand_profile=upstream_demand,
    downstream_supply_profile=10000.0,  # Ample downstream
)

result = sim.run(steps=3)

dt = 0.01
print("Single Cell Test - Last Cell Bug")
print("="*60)

for step in range(len(result.flows["cell_0"])):
    density_start = result.densities["cell_0"][step]
    vehicles_start = density_start * cells[0].length_km * cells[0].lanes
    
    outflow = result.flows["cell_0"][step]
    vehicles_out = outflow * dt
    
    inflow = upstream_demand
    vehicles_in = inflow * dt
    
    # What the code computes (WITH BUG):
    # max_available_flow = (vehicles_start + vehicles_in) / dt
    computed_max_with_bug = (vehicles_start + vehicles_in) / dt
    
    # What it SHOULD compute (WITHOUT BUG):
    # max_available_flow = vehicles_start / dt
    correct_max_without_bug = vehicles_start / dt
    
    print(f"\nStep {step}:")
    print(f"  Vehicles at start: {vehicles_start:.2f}")
    print(f"  Inflow: {inflow:.2f} veh/h ({vehicles_in:.2f} vehicles)")
    print(f"  Outflow: {outflow:.2f} veh/h ({vehicles_out:.2f} vehicles)")
    print(f"  Max with bug: {computed_max_with_bug:.2f} veh/h")
    print(f"  Max without bug: {correct_max_without_bug:.2f} veh/h")
    
    # With the bug, outflow can be up to computed_max_with_bug
    # Without the bug, outflow should be limited to correct_max_without_bug
    if outflow > correct_max_without_bug + 1.0:
        print(f"  *** BUG: Outflow exceeds what was available at start!")
        print(f"      Outflow ({outflow:.2f}) > Available ({correct_max_without_bug:.2f})")

# Check net accumulation
print("\n" + "="*60)
print("Net Vehicle Accumulation Check:")
print("="*60)

for step in range(1, len(result.densities["cell_0"])):
    veh_prev = result.densities["cell_0"][step-1] * cells[0].length_km * cells[0].lanes
    veh_curr = result.densities["cell_0"][step] * cells[0].length_km * cells[0].lanes
    
    inflow = upstream_demand * dt
    outflow_veh = result.flows["cell_0"][step-1] * dt
    
    expected_curr = veh_prev + inflow - outflow_veh
    error = veh_curr - expected_curr
    
    print(f"Step {step-1} -> {step}:")
    print(f"  Expected: {expected_curr:.2f}, Actual: {veh_curr:.2f}, Error: {error:.2f}")

