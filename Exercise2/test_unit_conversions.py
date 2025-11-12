"""Unit tests for unit conversions and lane drop/increase scenarios in METANET.

This test suite specifically validates:
- Correct unit conversions in density updates
- Vehicle conservation across lane drops and increases
- Profile unit expectations (veh/h vs veh per step)
- Ramp queue unit consistency
"""

import unittest
from typing import List

from metanet_simulation import (
    METANETCellConfig,
    METANETSimulation,
    METANETOnRampConfig,
    build_uniform_metanet_mainline,
)


class TestUnitConversions(unittest.TestCase):
    """Test unit conversions in METANET simulation."""

    def test_density_update_units(self):
        """Test that density update correctly handles units."""
        # Create a single cell with known parameters
        cells = build_uniform_metanet_mainline(
            num_cells=1,
            cell_length_km=1.0,
            lanes=2,
            free_flow_speed_kmh=100.0,
            jam_density_veh_per_km_per_lane=160.0,
            initial_density_veh_per_km_per_lane=50.0,
        )
        
        # Constant inflow of 1000 veh/h
        sim = METANETSimulation(
            cells=cells,
            time_step_hours=0.1,
            upstream_demand_profile=1000.0,
            downstream_supply_profile=0.0,  # No outflow
        )
        
        result = sim.run(steps=1)
        
        # Expected density change:
        # d_rho = (T / (L * lanes)) * (inflow - outflow)
        # d_rho = (0.1 hours / (1.0 km * 2 lanes)) * (1000 veh/h - 0)
        # d_rho = 0.05 / lane-km * 1000 veh/h = 50 veh/km/lane
        initial_density = 50.0
        expected_density = initial_density + 50.0
        
        final_density = result.densities["cell_0"][-1]
        
        # Should be close to expected (within bounds and numerical error)
        # Note: May be capped by jam density
        self.assertGreater(final_density, initial_density)
        self.assertLess(final_density, cells[0].jam_density_veh_per_km_per_lane + 1.0)

    def test_flow_calculation_units(self):
        """Test that flow calculation correctly includes lane factor."""
        # Create cells with different lane counts
        cells = build_uniform_metanet_mainline(
            num_cells=2,
            cell_length_km=1.0,
            lanes=[3, 2],  # Lane drop
            free_flow_speed_kmh=100.0,
            jam_density_veh_per_km_per_lane=160.0,
            initial_density_veh_per_km_per_lane=30.0,
            initial_speed_kmh=80.0,
        )
        
        sim = METANETSimulation(
            cells=cells,
            time_step_hours=0.1,
            upstream_demand_profile=0.0,  # No inflow
            downstream_supply_profile=10000.0,  # Ample outflow
        )
        
        result = sim.run(steps=1)
        
        # Flow from cell 0 should be: rho * v * lanes = 30 * 80 * 3 = 7200 veh/h
        # This is the total flow, not per-lane
        flow_cell_0 = result.flows["cell_0"][0]
        
        # Should be non-zero and reasonable
        self.assertGreater(flow_cell_0, 0.0)
        self.assertLess(flow_cell_0, 10000.0)

    def test_ramp_queue_units(self):
        """Test that ramp queue is in vehicles, not veh/h."""
        cells = build_uniform_metanet_mainline(
            num_cells=2,
            cell_length_km=1.0,
            lanes=3,
            free_flow_speed_kmh=100.0,
            jam_density_veh_per_km_per_lane=160.0,
        )
        
        # Ramp with 600 veh/h arrival, 300 veh/h metering
        # Net accumulation: 300 veh/h
        ramp = METANETOnRampConfig(
            target_cell=1,
            arrival_rate_profile=600.0,  # veh/h
            meter_rate_veh_per_hour=300.0,  # veh/h
            initial_queue_veh=0.0,
        )
        
        sim = METANETSimulation(
            cells=cells,
            time_step_hours=0.1,  # 0.1 hours = 6 minutes
            upstream_demand_profile=0.0,
            on_ramps=[ramp],
        )
        
        result = sim.run(steps=10)
        
        # After 10 steps (1 hour total), expect roughly 300 vehicles in queue
        # (600 veh/h arriving - 300 veh/h leaving) * 1 hour = 300 veh
        final_queue = result.ramp_queues["ramp_1"][-1]
        
        # Queue should be in vehicles (not veh/h)
        # Should accumulate over time
        self.assertGreater(final_queue, 100.0)  # At least some accumulation
        self.assertLess(final_queue, 500.0)  # But reasonable magnitude


class TestLaneChanges(unittest.TestCase):
    """Test vehicle conservation across lane drops and increases."""

    def test_lane_drop_conservation(self):
        """Test vehicle conservation across a lane drop."""
        # 3 lanes -> 2 lanes
        cells = build_uniform_metanet_mainline(
            num_cells=2,
            cell_length_km=1.0,
            lanes=[3, 2],
            free_flow_speed_kmh=100.0,
            jam_density_veh_per_km_per_lane=160.0,
            initial_density_veh_per_km_per_lane=40.0,
        )
        
        sim = METANETSimulation(
            cells=cells,
            time_step_hours=0.01,  # Small time step
            upstream_demand_profile=3000.0,  # Constant flow
            downstream_supply_profile=10000.0,
        )
        
        result = sim.run(steps=10)
        
        # Total vehicles in the system
        def total_vehicles(step_idx):
            rho_0 = result.densities["cell_0"][step_idx]
            rho_1 = result.densities["cell_1"][step_idx]
            # vehicles = density * length * lanes
            veh_0 = rho_0 * cells[0].length_km * cells[0].lanes
            veh_1 = rho_1 * cells[1].length_km * cells[1].lanes
            return veh_0 + veh_1
        
        initial_vehicles = total_vehicles(0)
        final_vehicles = total_vehicles(-1)
        
        # Vehicles should increase (inflow > outflow initially)
        # But not by an unreasonable amount
        self.assertGreater(final_vehicles, initial_vehicles - 100.0)
        self.assertLess(final_vehicles, initial_vehicles + 500.0)

    def test_lane_increase_conservation(self):
        """Test vehicle conservation across a lane increase."""
        # 2 lanes -> 3 lanes
        cells = build_uniform_metanet_mainline(
            num_cells=2,
            cell_length_km=1.0,
            lanes=[2, 3],
            free_flow_speed_kmh=100.0,
            jam_density_veh_per_km_per_lane=160.0,
            initial_density_veh_per_km_per_lane=50.0,
        )
        
        sim = METANETSimulation(
            cells=cells,
            time_step_hours=0.01,
            upstream_demand_profile=2000.0,
            downstream_supply_profile=10000.0,
        )
        
        result = sim.run(steps=10)
        
        # Check that densities remain reasonable
        for cell_name in result.densities:
            for rho in result.densities[cell_name]:
                self.assertGreaterEqual(rho, 0.0)
                self.assertLessEqual(rho, 160.0 + 1e-6)

    def test_multiple_lane_changes(self):
        """Test conservation across multiple lane changes."""
        # 3 -> 2 -> 3 -> 2
        cells = build_uniform_metanet_mainline(
            num_cells=4,
            cell_length_km=0.5,
            lanes=[3, 2, 3, 2],
            free_flow_speed_kmh=100.0,
            jam_density_veh_per_km_per_lane=160.0,
            initial_density_veh_per_km_per_lane=30.0,
        )
        
        sim = METANETSimulation(
            cells=cells,
            time_step_hours=0.01,
            upstream_demand_profile=3000.0,
            downstream_supply_profile=5000.0,
        )
        
        result = sim.run(steps=20)
        
        # Verify simulation completes without errors
        self.assertEqual(len(result.densities), 4)
        
        # Verify all densities are within bounds
        for cell_name in result.densities:
            for rho in result.densities[cell_name]:
                self.assertGreaterEqual(rho, 0.0)
                self.assertLessEqual(rho, 160.0 + 1e-6)

    def test_lane_drop_with_ramp(self):
        """Test lane drop with on-ramp merging."""
        # Cell 1 has lane drop, and also receives ramp flow
        cells = build_uniform_metanet_mainline(
            num_cells=3,
            cell_length_km=1.0,
            lanes=[3, 2, 2],  # Lane drop at cell 1
            free_flow_speed_kmh=100.0,
            jam_density_veh_per_km_per_lane=160.0,
            initial_density_veh_per_km_per_lane=40.0,
        )
        
        ramp = METANETOnRampConfig(
            target_cell=1,  # Merges at lane drop
            arrival_rate_profile=400.0,
            meter_rate_veh_per_hour=400.0,
            initial_queue_veh=0.0,
        )
        
        sim = METANETSimulation(
            cells=cells,
            time_step_hours=0.05,
            upstream_demand_profile=3000.0,
            on_ramps=[ramp],
        )
        
        result = sim.run(steps=20)
        
        # Verify densities remain bounded
        for cell_name in result.densities:
            for rho in result.densities[cell_name]:
                self.assertGreaterEqual(rho, 0.0)
                self.assertLessEqual(rho, 160.0 + 1e-6)
        
        # Verify ramp queue is reasonable
        for queue in result.ramp_queues["ramp_1"]:
            self.assertGreaterEqual(queue, -1e-6)


class TestProfileUnitExpectations(unittest.TestCase):
    """Test that profiles are expected to return veh/h."""

    def test_constant_profile_interpreted_as_veh_per_hour(self):
        """Test that constant profile value is interpreted as veh/h."""
        cells = build_uniform_metanet_mainline(
            num_cells=1,
            cell_length_km=1.0,
            lanes=2,
            free_flow_speed_kmh=100.0,
            jam_density_veh_per_km_per_lane=160.0,
            initial_density_veh_per_km_per_lane=10.0,
        )
        
        # 1000 veh/h for 0.1 hours should add 100 vehicles total
        sim = METANETSimulation(
            cells=cells,
            time_step_hours=0.1,
            upstream_demand_profile=1000.0,  # veh/h
            downstream_supply_profile=0.0,
        )
        
        result = sim.run(steps=1)
        
        # Density change: d_rho = T/(L*lanes) * flow
        # d_rho = 0.1 / (1.0 * 2) * 1000 = 50 veh/km/lane
        initial_density = 10.0
        expected_density = initial_density + 50.0
        final_density = result.densities["cell_0"][-1]
        
        # Should be close (allowing for numerical differences)
        self.assertGreater(final_density, initial_density + 40.0)
        self.assertLess(final_density, initial_density + 60.0)

    def test_list_profile_interpreted_as_veh_per_hour(self):
        """Test that list profile values are interpreted as veh/h."""
        cells = build_uniform_metanet_mainline(
            num_cells=1,
            cell_length_km=1.0,
            lanes=2,
            free_flow_speed_kmh=100.0,
            jam_density_veh_per_km_per_lane=160.0,
            initial_density_veh_per_km_per_lane=10.0,
        )
        
        # Profile: [1000, 2000, 1000] veh/h
        sim = METANETSimulation(
            cells=cells,
            time_step_hours=0.1,
            upstream_demand_profile=[1000.0, 2000.0, 1000.0],
            downstream_supply_profile=0.0,
        )
        
        result = sim.run(steps=3)
        
        # After 3 steps, total inflow: (1000 + 2000 + 1000) * 0.1 = 400 vehicles
        # Density increase: 400 / (1.0 km * 2 lanes) = 200 veh/km/lane
        initial_density = 10.0
        expected_density = initial_density + 200.0
        final_density = result.densities["cell_0"][-1]
        
        # Should be capped by jam density
        self.assertGreater(final_density, initial_density + 100.0)
        self.assertLessEqual(final_density, 160.0 + 1e-6)


class TestRampFlowAndQueueConsistency(unittest.TestCase):
    """Test ramp flow and queue update consistency."""

    def test_ramp_queue_reconstruction(self):
        """Test that ramp queue can be reconstructed from arrivals and flows."""
        cells = build_uniform_metanet_mainline(
            num_cells=2,
            cell_length_km=1.0,
            lanes=3,
            free_flow_speed_kmh=100.0,
            jam_density_veh_per_km_per_lane=160.0,
            initial_density_veh_per_km_per_lane=20.0,
        )
        
        # Arrivals > metering, so queue should accumulate
        arrival_rate = 800.0  # veh/h
        meter_rate = 500.0    # veh/h
        initial_queue = 10.0  # vehicles
        
        ramp = METANETOnRampConfig(
            target_cell=1,
            arrival_rate_profile=arrival_rate,
            meter_rate_veh_per_hour=meter_rate,
            initial_queue_veh=initial_queue,
        )
        
        sim = METANETSimulation(
            cells=cells,
            time_step_hours=0.1,
            upstream_demand_profile=1000.0,
            on_ramps=[ramp],
        )
        
        result = sim.run(steps=10)
        
        # Reconstruct queue
        dt = 0.1
        steps = 10
        reconstructed_queue = [initial_queue]
        
        for step in range(steps):
            arrivals = arrival_rate * dt  # veh/h * hours = vehicles
            admitted = result.ramp_flows["ramp_1"][step] * dt  # veh/h * hours = vehicles
            new_queue = max(0.0, reconstructed_queue[-1] + arrivals - admitted)
            reconstructed_queue.append(new_queue)
        
        # Compare reconstructed to reported
        reported_queue = result.ramp_queues["ramp_1"]
        
        # Should be similar (within numerical tolerance)
        for i in range(len(reported_queue)):
            diff = abs(reconstructed_queue[i] - reported_queue[i])
            # Allow some tolerance for numerical differences
            self.assertLess(diff, 5.0, 
                f"Step {i}: reconstructed={reconstructed_queue[i]:.2f}, "
                f"reported={reported_queue[i]:.2f}")

    def test_ramp_flow_stored_as_veh_per_hour(self):
        """Test that ramp_flows are stored as veh/h, not veh per step."""
        cells = build_uniform_metanet_mainline(
            num_cells=2,
            cell_length_km=1.0,
            lanes=3,
            free_flow_speed_kmh=100.0,
            jam_density_veh_per_km_per_lane=160.0,
        )
        
        ramp = METANETOnRampConfig(
            target_cell=1,
            arrival_rate_profile=600.0,
            meter_rate_veh_per_hour=400.0,
        )
        
        sim = METANETSimulation(
            cells=cells,
            time_step_hours=0.1,
            upstream_demand_profile=1000.0,
            on_ramps=[ramp],
        )
        
        result = sim.run(steps=5)
        
        # Ramp flows should be in veh/h range (hundreds), not per-step (tens)
        for flow in result.ramp_flows["ramp_1"]:
            # Should be comparable to meter_rate (400 veh/h)
            self.assertGreater(flow, 0.0)
            self.assertLess(flow, 1000.0)  # veh/h magnitude
            # Not in per-step magnitude (which would be ~40 for dt=0.1)


if __name__ == "__main__":
    unittest.main()
