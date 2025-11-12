"""Example demonstrating the diagnostic tools for METANET unit verification.

This example shows how to:
1. Run a METANET simulation with lane changes
2. Use diagnostic tools to verify units
3. Check queue reconstruction
4. Verify conservation across lane drops
"""

from metanet_simulation import (
    METANETSimulation,
    METANETOnRampConfig,
    build_uniform_metanet_mainline,
)
from diagnose_metanet_units import (
    reconstruct_queue,
    compare_queues,
    check_profile_units,
    print_unit_summary,
)


def example_basic_scenario():
    """Basic scenario with a lane drop and on-ramp."""
    print("=" * 70)
    print("Example: METANET Simulation with Lane Drop and On-Ramp")
    print("=" * 70)
    
    # Create cells with lane drop: 3 lanes -> 2 lanes -> 2 lanes
    cells = build_uniform_metanet_mainline(
        num_cells=3,
        cell_length_km=0.5,
        lanes=[3, 2, 2],  # Lane drop at cell 1
        free_flow_speed_kmh=100.0,
        jam_density_veh_per_km_per_lane=160.0,
        critical_density_veh_per_km_per_lane=33.5,
        tau_s=18.0,
        nu=60.0,
        kappa=40.0,
        delta=0.0122,
        capacity_veh_per_hour_per_lane=2200.0,
        initial_density_veh_per_km_per_lane=[20.0, 30.0, 25.0],
    )
    
    # On-ramp at the lane drop location
    on_ramp = METANETOnRampConfig(
        target_cell=1,
        arrival_rate_profile=600.0,  # veh/h
        meter_rate_veh_per_hour=500.0,  # veh/h
        mainline_priority=0.6,
        initial_queue_veh=10.0,
        name="ramp_at_lane_drop",
    )
    
    # Create simulation
    sim = METANETSimulation(
        cells=cells,
        time_step_hours=0.1,  # 6 minutes
        upstream_demand_profile=4000.0,  # veh/h
        downstream_supply_profile=3500.0,  # veh/h (bottleneck)
        on_ramps=[on_ramp],
    )
    
    print("\nSimulation setup:")
    print(f"  Time step: {sim.dt} hours ({sim.dt * 60:.1f} minutes)")
    print(f"  Cells: {[cell.lanes for cell in cells]} lanes")
    print(f"  Upstream demand: 4000 veh/h")
    print(f"  Ramp arrival: 600 veh/h, metering: 500 veh/h")
    print(f"  Initial ramp queue: {on_ramp.initial_queue_veh} vehicles")
    
    # Run simulation
    print("\nRunning simulation...")
    result = sim.run(steps=20)
    print(f"✓ Simulation completed: {result.steps} time points")
    
    # Diagnostic 1: Check profile units
    print("\n" + "=" * 70)
    print("Diagnostic 1: Profile Unit Check")
    print("=" * 70)
    check_profile_units(
        profile=on_ramp.arrival_rate_profile,
        dt=sim.dt,
        profile_name="ramp_arrival_rate"
    )
    check_profile_units(
        profile=sim.upstream_profile,
        dt=sim.dt,
        profile_name="upstream_demand"
    )
    
    # Diagnostic 2: Reconstruct and compare queue
    print("\n" + "=" * 70)
    print("Diagnostic 2: Queue Reconstruction")
    print("=" * 70)
    reconstructed = reconstruct_queue(
        arrival_rate_profile=on_ramp.arrival_rate_profile,
        ramp_flows=result.ramp_flows["ramp_at_lane_drop"],
        initial_queue=on_ramp.initial_queue_veh,
        dt=result.time_step_hours,
        ramp_name="ramp_at_lane_drop",
    )
    compare_queues(
        reconstructed_queue=reconstructed,
        reported_queue=result.ramp_queues["ramp_at_lane_drop"],
        tolerance=5.0,
        ramp_name="ramp_at_lane_drop",
    )
    
    # Diagnostic 3: Show density behavior across lane drop
    print("\n" + "=" * 70)
    print("Diagnostic 3: Density Across Lane Drop")
    print("=" * 70)
    print("\nFinal densities (veh/km/lane):")
    for i, cell in enumerate(cells):
        density = result.densities[cell.name][-1]
        total_veh = density * cell.length_km * cell.lanes
        print(f"  Cell {i} ({cell.lanes} lanes): "
              f"rho={density:.1f} veh/km/lane, "
              f"total={total_veh:.1f} veh")
    
    print("\nNote: Density is per-lane, so it can stay similar across lane drops.")
    print("Total vehicles (density * length * lanes) accounts for lane count.")
    
    # Diagnostic 4: Show flows
    print("\n" + "=" * 70)
    print("Diagnostic 4: Flows (veh/h)")
    print("=" * 70)
    print("\nFinal flows:")
    for i, cell in enumerate(cells):
        flow = result.flows[cell.name][-1]
        print(f"  Cell {i} ({cell.lanes} lanes): {flow:.0f} veh/h")
    print(f"  Ramp flow: {result.ramp_flows['ramp_at_lane_drop'][-1]:.0f} veh/h")
    
    return result


def example_wrong_units():
    """Example showing what happens when profile units are wrong."""
    print("\n" + "=" * 70)
    print("Example: Detecting Wrong Profile Units")
    print("=" * 70)
    
    # Simulate a common error: profile returns vehicles per time-step
    # instead of veh/h
    dt = 0.1  # hours
    
    # Correct profile: 600 veh/h
    correct_profile = 600.0
    
    # Wrong profile: user thinks "6 vehicles per step of 6 minutes"
    # and enters 6 as the profile value
    wrong_profile = 6.0  # This will be interpreted as 6 veh/h!
    
    print("\nScenario: User wants 60 vehicles per 6-minute step")
    print(f"Time step: {dt} hours = {dt * 60:.0f} minutes")
    print(f"Desired: 60 vehicles per step = {60.0 / dt:.0f} veh/h")
    
    print("\n--- Correct profile (600 veh/h) ---")
    check_profile_units(correct_profile, dt, "correct_profile")
    
    print("\n--- Wrong profile (60 as if per-step) ---")
    check_profile_units(wrong_profile, dt, "wrong_profile")
    
    print("\n" + "=" * 70)
    print("LESSON: Always use veh/h for profiles, not veh per time-step!")
    print("=" * 70)


if __name__ == "__main__":
    # Print unit summary
    print_unit_summary()
    
    # Run examples
    print("\n" * 2)
    result = example_basic_scenario()
    
    print("\n" * 2)
    example_wrong_units()
    
    print("\n" + "=" * 70)
    print("All diagnostics complete!")
    print("=" * 70)
