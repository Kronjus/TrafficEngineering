"""Test to compare VKT (Vehicle Kilometers Traveled) between CTM and METANET.

This test creates identical scenarios in both models and compares the total
VKT to ensure they produce equivalent results.
"""

import unittest
from ctm_simulation import (
    CTMSimulation,
    build_uniform_mainline,
)
from metanet_simulation import (
    METANETSimulation,
    build_uniform_metanet_mainline,
)


def calculate_vkt(result, cells):
    """Calculate total Vehicle Kilometers Traveled from simulation result.
    
    VKT = sum over all cells and all time steps of:
          density (veh/km/lane) * length (km) * lanes * dt (hours)
    
    This represents the total vehicle-distance traveled over the simulation period.
    """
    total_vkt = 0.0
    dt = result.time_step_hours
    
    for cell in cells:
        cell_name = cell.name
        densities = result.densities[cell_name]
        length = cell.length_km
        lanes = cell.lanes
        
        # Sum density over all time steps (skip initial, count steps only)
        # Each density value represents the average over that time step
        for i in range(1, len(densities)):
            # VKT contribution = density (veh/km/lane) * length (km) * lanes * dt (hours)
            # Units: (veh/km/lane) * km * lanes * hours = veh * hours
            # To get veh*km, we need: density * length * lanes * dt * average_speed
            # But for VKT we want: distance = density * length * lanes (total vehicles) * (distance traveled in dt)
            # Actually VKT is: sum of (vehicles in cell) * (cell length) over time
            # vehicles in cell = density * length * lanes
            # But that gives veh*km at each instant, we need to integrate over time
            # VKT = integral of (density * lanes * length) * dt * average_speed
            # But we're measuring vehicle-DISTANCE, not vehicle-TIME
            # 
            # Correct formula: VKT per cell per timestep = average vehicles * distance traveled
            # Average vehicles in cell = density (veh/km/lane) * length (km) * lanes
            # Distance traveled per vehicle in dt = average_speed * dt
            # 
            # But we don't have speed in CTM directly...
            # 
            # Alternative: VKT = sum of all flows * dt * cell_length
            # This counts vehicles moving through the system
            pass
    
    # Better approach: sum all flows through the network
    # Each flow (veh/h) * dt (hours) = vehicles passing
    # vehicles passing * cell_length = vehicle-km
    for cell_name in result.flows:
        flows = result.flows[cell_name]
        # Find corresponding cell
        cell = next(c for c in cells if c.name == cell_name)
        for flow in flows:
            # flow is in veh/h
            # flow * dt = vehicles passing in this time step
            # vehicles * cell.length_km = vehicle-km
            total_vkt += flow * dt * cell.length_km
    
    return total_vkt


class TestVKTComparison(unittest.TestCase):
    """Compare VKT between CTM and METANET simulations."""
    
    def test_vkt_simple_scenario(self):
        """Test VKT equivalence in a simple scenario without ramps."""
        # Create identical cell configurations
        # Use longer cells and smaller time step to respect CFL condition
        num_cells = 3
        cell_length = 1.0  # Longer cells for stability
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
        # Use smaller time step to respect CFL: dt < L/v = 1.0/100 = 0.01 hours
        dt = 0.005  # 18 seconds, well below CFL limit
        steps = 200  # Same total simulation time
        demand = 3000.0
        
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
        
        # Calculate VKT for both
        ctm_vkt = calculate_vkt(ctm_result, ctm_cells)
        metanet_vkt = calculate_vkt(metanet_result, metanet_cells)
        
        print(f"\nCTM VKT: {ctm_vkt:.2f}")
        print(f"METANET VKT: {metanet_vkt:.2f}")
        print(f"Difference: {metanet_vkt - ctm_vkt:.2f} ({100*(metanet_vkt-ctm_vkt)/ctm_vkt:.2f}%)")
        
        # They should be very close (within 5% due to different dynamics)
        rel_diff = abs(metanet_vkt - ctm_vkt) / ctm_vkt
        self.assertLess(rel_diff, 0.05, 
                       f"VKT difference too large: {100*rel_diff:.2f}%")


if __name__ == "__main__":
    unittest.main()
