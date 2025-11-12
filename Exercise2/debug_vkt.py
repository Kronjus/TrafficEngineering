"""Debug script to understand VKT discrepancy between CTM and METANET."""

from ctm_simulation import (
    CTMSimulation,
    build_uniform_mainline,
)
from metanet_simulation import (
    METANETSimulation,
    build_uniform_metanet_mainline,
)


def print_flows(result, name):
    """Print all flows from a result."""
    print(f"\n{name} Flows:")
    for cell_name in sorted(result.flows.keys()):
        flows = result.flows[cell_name]
        print(f"  {cell_name}: {flows[:10]}")  # First 10 steps


def main():
    # Create identical cell configurations
    num_cells = 3
    cell_length = 0.5
    lanes = 3
    free_flow_speed = 100.0
    jam_density = 160.0
    
    ctm_cells = build_uniform_mainline(
        num_cells=num_cells,
        cell_length_km=cell_length,
        lanes=lanes,
        free_flow_speed_kmh=free_flow_speed,
        congestion_wave_speed_kmh=20.0,
        capacity_veh_per_hour_per_lane=2200.0,
        jam_density_veh_per_km_per_lane=jam_density,
        initial_density_veh_per_km_per_lane=20.0,
    )
    
    metanet_cells = build_uniform_metanet_mainline(
        num_cells=num_cells,
        cell_length_km=cell_length,
        lanes=lanes,
        free_flow_speed_kmh=free_flow_speed,
        jam_density_veh_per_km_per_lane=jam_density,
        initial_density_veh_per_km_per_lane=20.0,
    )
    
    # Run simulations with identical parameters
    dt = 0.1  # 6 minutes
    steps = 20
    demand = 3000.0
    
    print(f"Configuration:")
    print(f"  Cells: {num_cells}")
    print(f"  Length: {cell_length} km")
    print(f"  Lanes: {lanes}")
    print(f"  Time step: {dt} hours")
    print(f"  Steps: {steps}")
    print(f"  Upstream demand: {demand} veh/h")
    
    ctm_sim = CTMSimulation(
        cells=ctm_cells,
        time_step_hours=dt,
        upstream_demand_profile=demand,
        downstream_supply_profile=5000.0,
    )
    ctm_result = ctm_sim.run(steps=steps)
    
    metanet_sim = METANETSimulation(
        cells=metanet_cells,
        time_step_hours=dt,
        upstream_demand_profile=demand,
        downstream_supply_profile=5000.0,
    )
    metanet_result = metanet_sim.run(steps=steps)
    
    # Print flows
    print_flows(ctm_result, "CTM")
    print_flows(metanet_result, "METANET")
    
    # Calculate and print VKT for each cell
    print("\n\nVKT per cell:")
    ctm_total = 0.0
    metanet_total = 0.0
    
    for i, cell in enumerate(ctm_cells):
        cell_name = cell.name
        ctm_flows = ctm_result.flows[cell_name]
        metanet_flows = metanet_result.flows[cell_name]
        
        ctm_vkt = sum(flow * dt * cell.length_km for flow in ctm_flows)
        metanet_vkt = sum(flow * dt * metanet_cells[i].length_km for flow in metanet_flows)
        
        ctm_total += ctm_vkt
        metanet_total += metanet_vkt
        
        print(f"  {cell_name}:")
        print(f"    CTM VKT: {ctm_vkt:.2f}")
        print(f"    METANET VKT: {metanet_vkt:.2f}")
        print(f"    Ratio: {metanet_vkt/ctm_vkt:.2f}x")
        print(f"    Sum of flows (CTM): {sum(ctm_flows):.2f}")
        print(f"    Sum of flows (METANET): {sum(metanet_flows):.2f}")
    
    print(f"\nTotal VKT:")
    print(f"  CTM: {ctm_total:.2f}")
    print(f"  METANET: {metanet_total:.2f}")
    print(f"  Ratio: {metanet_total/ctm_total:.2f}x")
    print(f"  Difference: {metanet_total - ctm_total:.2f} ({100*(metanet_total-ctm_total)/ctm_total:.2f}%)")


if __name__ == "__main__":
    main()
