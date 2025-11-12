"""Debug script to check speed and density values in METANET."""

from metanet_simulation import (
    METANETSimulation,
    build_uniform_metanet_mainline,
    exponential_equilibrium_speed,
)


def main():
    # Create cell configuration
    num_cells = 3
    cell_length = 0.5
    lanes = 3
    free_flow_speed = 100.0
    jam_density = 160.0
    initial_density = 20.0
    
    cells = build_uniform_metanet_mainline(
        num_cells=num_cells,
        cell_length_km=cell_length,
        lanes=lanes,
        free_flow_speed_kmh=free_flow_speed,
        jam_density_veh_per_km_per_lane=jam_density,
        initial_density_veh_per_km_per_lane=initial_density,
    )
    
    # What should the initial equilibrium speed be?
    v_eq = exponential_equilibrium_speed(
        initial_density,
        free_flow_speed,
        33.5,  # default critical density
    )
    print(f"Expected equilibrium speed at density {initial_density}: {v_eq:.2f} km/h")
    
    # Run simulation
    dt = 0.1
    steps = 20
    demand = 3000.0
    
    sim = METANETSimulation(
        cells=cells,
        time_step_hours=dt,
        upstream_demand_profile=demand,
        downstream_supply_profile=5000.0,
    )
    
    # Patch the run method to capture intermediate states
    result = sim.run(steps=steps)
    
    # Print densities and check if they're reasonable
    print("\nDensities over time (cell_0, first 10 steps):")
    for i, d in enumerate(result.densities['cell_0'][:11]):
        print(f"  Step {i}: {d:.2f} veh/km/lane")
    
    print("\nFlows from cell_0 (first 10 steps):")
    for i, f in enumerate(result.flows['cell_0'][:10]):
        print(f"  Step {i}: {f:.2f} veh/h")
        # For comparison, if density was constant at 20 and speed at v_eq:
        expected_flow = 20.0 * v_eq * lanes
        print(f"    (Expected if density=20, v={v_eq:.2f}, lanes=3: {expected_flow:.2f} veh/h)")


if __name__ == "__main__":
    main()
