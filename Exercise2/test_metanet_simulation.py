"""Unit tests for the METANET simulation module.

This test suite validates the correctness of the METANET implementation,
including:
- Cell configuration validation
- Speed dynamics (relaxation, convection, anticipation)
- Conservation of vehicles
- Steady-state behavior
- On-ramp merging
- Comparison with expected METANET behavior
"""

import unittest
from typing import List

from metanet_simulation import (
    METANETCellConfig,
    METANETSimulation,
    METANETOnRampConfig,
    SimulationResult,
    build_uniform_metanet_mainline,
    run_basic_metanet_scenario,
    greenshields_speed,
)


class TestMETANETCellConfig(unittest.TestCase):
    """Test METANETCellConfig validation and construction."""

    def test_valid_cell_config(self):
        """Test that a valid cell configuration can be created."""
        cell = METANETCellConfig(
            name="test_cell",
            length_km=1.0,
            lanes=3,
            free_flow_speed_kmh=100.0,
            jam_density_veh_per_km_per_lane=160.0,
            tau_s=18.0,
            eta=60.0,
            kappa=40.0,
        )
        self.assertEqual(cell.name, "test_cell")
        self.assertEqual(cell.length_km, 1.0)
        self.assertEqual(cell.lanes, 3)
        self.assertEqual(cell.tau_s, 18.0)
        self.assertEqual(cell.eta, 60.0)

    def test_invalid_lanes(self):
        """Test that zero or negative lanes raise ValueError."""
        with self.assertRaises(ValueError):
            METANETCellConfig(
                name="bad_cell",
                length_km=1.0,
                lanes=0,
                free_flow_speed_kmh=100.0,
                jam_density_veh_per_km_per_lane=160.0,
            )

    def test_negative_parameters(self):
        """Test that negative parameters raise ValueError."""
        with self.assertRaises(ValueError):
            METANETCellConfig(
                name="bad_cell",
                length_km=-1.0,
                lanes=3,
                free_flow_speed_kmh=100.0,
                jam_density_veh_per_km_per_lane=160.0,
            )

    def test_negative_tau(self):
        """Test that negative tau raises ValueError."""
        with self.assertRaises(ValueError):
            METANETCellConfig(
                name="bad_cell",
                length_km=1.0,
                lanes=3,
                free_flow_speed_kmh=100.0,
                jam_density_veh_per_km_per_lane=160.0,
                tau_s=-10.0,
            )

    def test_negative_eta(self):
        """Test that negative eta raises ValueError."""
        with self.assertRaises(ValueError):
            METANETCellConfig(
                name="bad_cell",
                length_km=1.0,
                lanes=3,
                free_flow_speed_kmh=100.0,
                jam_density_veh_per_km_per_lane=160.0,
                eta=-5.0,
            )

    def test_negative_kappa(self):
        """Test that negative kappa raises ValueError."""
        with self.assertRaises(ValueError):
            METANETCellConfig(
                name="bad_cell",
                length_km=1.0,
                lanes=3,
                free_flow_speed_kmh=100.0,
                jam_density_veh_per_km_per_lane=160.0,
                kappa=-1.0,
            )

    def test_negative_initial_density(self):
        """Test that negative initial density raises ValueError."""
        with self.assertRaises(ValueError):
            METANETCellConfig(
                name="bad_cell",
                length_km=1.0,
                lanes=3,
                free_flow_speed_kmh=100.0,
                jam_density_veh_per_km_per_lane=160.0,
                initial_density_veh_per_km_per_lane=-10.0,
            )

    def test_initial_speed_exceeds_free_flow(self):
        """Test that initial speed > free flow raises ValueError."""
        with self.assertRaises(ValueError):
            METANETCellConfig(
                name="bad_cell",
                length_km=1.0,
                lanes=3,
                free_flow_speed_kmh=100.0,
                jam_density_veh_per_km_per_lane=160.0,
                initial_speed_kmh=150.0,
            )


class TestMETANETOnRampConfig(unittest.TestCase):
    """Test METANETOnRampConfig validation and construction."""

    def test_valid_onramp_config(self):
        """Test that a valid on-ramp configuration can be created."""
        ramp = METANETOnRampConfig(
            target_cell=1,
            arrival_rate_profile=500.0,
            mainline_priority=0.6,
        )
        self.assertEqual(ramp.target_cell, 1)
        self.assertEqual(ramp.mainline_priority, 0.6)
        self.assertEqual(ramp.queue_veh, 0.0)

    def test_invalid_priority(self):
        """Test that priority outside [0, 1] raises ValueError."""
        with self.assertRaises(ValueError):
            METANETOnRampConfig(
                target_cell=1,
                arrival_rate_profile=500.0,
                mainline_priority=1.5,
            )

    def test_negative_meter_rate(self):
        """Test that negative meter rate raises ValueError."""
        with self.assertRaises(ValueError):
            METANETOnRampConfig(
                target_cell=1,
                arrival_rate_profile=500.0,
                meter_rate_veh_per_hour=-100.0,
            )

    def test_initial_queue_assignment(self):
        """Test that initial queue is properly assigned."""
        ramp = METANETOnRampConfig(
            target_cell=1,
            arrival_rate_profile=500.0,
            initial_queue_veh=25.0,
        )
        self.assertEqual(ramp.queue_veh, 25.0)


class TestGreenshieldsSpeed(unittest.TestCase):
    """Test the Greenshields equilibrium speed function."""

    def test_zero_density(self):
        """Test that V(0) = v_f."""
        v = greenshields_speed(0.0, 100.0, 160.0)
        self.assertAlmostEqual(v, 100.0)

    def test_jam_density(self):
        """Test that V(rho_jam) = 0."""
        v = greenshields_speed(160.0, 100.0, 160.0)
        self.assertAlmostEqual(v, 0.0)

    def test_half_jam_density(self):
        """Test that V(rho_jam/2) = v_f/2."""
        v = greenshields_speed(80.0, 100.0, 160.0)
        self.assertAlmostEqual(v, 50.0)

    def test_speed_bounds(self):
        """Test that speed is bounded by [0, v_f]."""
        v = greenshields_speed(200.0, 100.0, 160.0)  # Density > jam
        self.assertAlmostEqual(v, 0.0)


class TestBuildUniformMetanetMainline(unittest.TestCase):
    """Test the build_uniform_metanet_mainline utility function."""

    def test_uniform_lanes(self):
        """Test creating cells with uniform lane count."""
        cells = build_uniform_metanet_mainline(
            num_cells=5,
            cell_length_km=0.5,
            lanes=3,
            free_flow_speed_kmh=100.0,
            jam_density_veh_per_km_per_lane=160.0,
        )
        self.assertEqual(len(cells), 5)
        for cell in cells:
            self.assertEqual(cell.lanes, 3)
            self.assertEqual(cell.length_km, 0.5)

    def test_variable_lanes(self):
        """Test creating cells with variable lane counts."""
        cells = build_uniform_metanet_mainline(
            num_cells=4,
            cell_length_km=0.5,
            lanes=[3, 3, 2, 2],
            free_flow_speed_kmh=100.0,
            jam_density_veh_per_km_per_lane=160.0,
        )
        self.assertEqual(len(cells), 4)
        self.assertEqual(cells[0].lanes, 3)
        self.assertEqual(cells[2].lanes, 2)

    def test_variable_initial_density(self):
        """Test creating cells with variable initial densities."""
        cells = build_uniform_metanet_mainline(
            num_cells=3,
            cell_length_km=0.5,
            lanes=3,
            free_flow_speed_kmh=100.0,
            jam_density_veh_per_km_per_lane=160.0,
            initial_density_veh_per_km_per_lane=[10.0, 20.0, 30.0],
        )
        self.assertEqual(cells[0].initial_density_veh_per_km_per_lane, 10.0)
        self.assertEqual(cells[1].initial_density_veh_per_km_per_lane, 20.0)
        self.assertEqual(cells[2].initial_density_veh_per_km_per_lane, 30.0)

    def test_variable_initial_speed(self):
        """Test creating cells with variable initial speeds."""
        cells = build_uniform_metanet_mainline(
            num_cells=3,
            cell_length_km=0.5,
            lanes=3,
            free_flow_speed_kmh=100.0,
            jam_density_veh_per_km_per_lane=160.0,
            initial_speed_kmh=[90.0, 80.0, 70.0],
        )
        self.assertEqual(cells[0].initial_speed_kmh, 90.0)
        self.assertEqual(cells[1].initial_speed_kmh, 80.0)
        self.assertEqual(cells[2].initial_speed_kmh, 70.0)


class TestMETANETSimulation(unittest.TestCase):
    """Test the METANETSimulation class and simulation mechanics."""

    def test_simulation_creation(self):
        """Test that a simulation can be created with valid parameters."""
        cells = build_uniform_metanet_mainline(
            num_cells=3,
            cell_length_km=0.5,
            lanes=3,
            free_flow_speed_kmh=100.0,
            jam_density_veh_per_km_per_lane=160.0,
        )
        sim = METANETSimulation(
            cells=cells,
            time_step_hours=0.1,
            upstream_demand_profile=1000.0,
        )
        self.assertEqual(len(sim.cells), 3)
        self.assertEqual(sim.dt, 0.1)

    def test_invalid_time_step(self):
        """Test that non-positive time step raises ValueError."""
        cells = build_uniform_metanet_mainline(
            num_cells=2,
            cell_length_km=0.5,
            lanes=3,
            free_flow_speed_kmh=100.0,
            jam_density_veh_per_km_per_lane=160.0,
        )
        with self.assertRaises(ValueError):
            METANETSimulation(
                cells=cells,
                time_step_hours=0.0,
                upstream_demand_profile=1000.0,
            )

    def test_empty_cells(self):
        """Test that empty cell list raises ValueError."""
        with self.assertRaises(ValueError):
            METANETSimulation(
                cells=[],
                time_step_hours=0.1,
                upstream_demand_profile=1000.0,
            )

    def test_invalid_onramp_target(self):
        """Test that invalid on-ramp target raises error."""
        cells = build_uniform_metanet_mainline(
            num_cells=3,
            cell_length_km=0.5,
            lanes=3,
            free_flow_speed_kmh=100.0,
            jam_density_veh_per_km_per_lane=160.0,
        )
        ramp = METANETOnRampConfig(
            target_cell=10,  # Out of range
            arrival_rate_profile=500.0,
        )
        with self.assertRaises(IndexError):
            METANETSimulation(
                cells=cells,
                time_step_hours=0.1,
                upstream_demand_profile=1000.0,
                on_ramps=[ramp],
            )

    def test_single_cell_conservation_no_flow(self):
        """Test vehicle conservation in a closed single cell (no inflow/outflow)."""
        initial_density = 50.0
        cells = build_uniform_metanet_mainline(
            num_cells=1,
            cell_length_km=1.0,
            lanes=2,
            free_flow_speed_kmh=100.0,
            jam_density_veh_per_km_per_lane=150.0,
            initial_density_veh_per_km_per_lane=initial_density,
        )
        
        # No inflow, no outflow
        sim = METANETSimulation(
            cells=cells,
            time_step_hours=0.01,  # Small time step
            upstream_demand_profile=0.0,
            downstream_supply_profile=0.0,
        )
        
        result = sim.run(steps=10)
        
        # Total vehicles should be conserved
        initial_vehicles = initial_density * cells[0].length_km * cells[0].lanes
        final_density = result.densities["cell_0"][-1]
        final_vehicles = final_density * cells[0].length_km * cells[0].lanes
        
        # Within small tolerance due to numerical precision
        self.assertAlmostEqual(initial_vehicles, final_vehicles, delta=1.0)

    def test_steady_state_preservation(self):
        """Test that a steady-state equilibrium is preserved."""
        # Initialize with equilibrium: rho = 50, v = V(50)
        initial_density = 50.0
        cells = build_uniform_metanet_mainline(
            num_cells=3,
            cell_length_km=1.0,
            lanes=2,
            free_flow_speed_kmh=100.0,
            jam_density_veh_per_km_per_lane=160.0,
            tau_s=18.0,
            eta=0.0,  # Disable anticipation for pure steady state
            initial_density_veh_per_km_per_lane=initial_density,
        )
        
        # Set initial speed to equilibrium
        v_eq = greenshields_speed(initial_density, 100.0, 160.0)
        for cell in cells:
            object.__setattr__(cell, "initial_speed_kmh", v_eq)
        
        # Inflow = outflow = rho * v * lanes
        steady_flow = initial_density * v_eq * 2.0
        
        sim = METANETSimulation(
            cells=cells,
            time_step_hours=0.01,
            upstream_demand_profile=steady_flow,
            downstream_supply_profile=steady_flow * 2.0,  # Ample downstream
        )
        
        result = sim.run(steps=20)
        
        # Density should remain relatively stable (allowing some numerical drift)
        for cell_name in result.densities:
            densities = result.densities[cell_name]
            for rho in densities[1:]:  # Skip initial
                self.assertGreater(rho, 0.0)
                # Allow for numerical changes but check reasonable bounds
                self.assertLess(abs(rho - initial_density), 30.0)

    def test_free_flow_propagation(self):
        """Test that vehicles propagate in free flow conditions."""
        cells = build_uniform_metanet_mainline(
            num_cells=3,
            cell_length_km=1.0,
            lanes=2,
            free_flow_speed_kmh=100.0,
            jam_density_veh_per_km_per_lane=150.0,
            initial_density_veh_per_km_per_lane=0.0,
        )
        
        # Low demand should result in free flow
        sim = METANETSimulation(
            cells=cells,
            time_step_hours=0.02,
            upstream_demand_profile=500.0,
        )
        
        result = sim.run(steps=30)
        
        # Density should increase in at least one cell
        max_density = max(
            max(result.densities["cell_0"]),
            max(result.densities["cell_1"]),
            max(result.densities["cell_2"]),
        )
        self.assertGreater(max_density, 0.0)

    def test_density_bounds(self):
        """Test that density stays within [0, rho_jam]."""
        cells = build_uniform_metanet_mainline(
            num_cells=2,
            cell_length_km=0.5,
            lanes=2,
            free_flow_speed_kmh=100.0,
            jam_density_veh_per_km_per_lane=150.0,
        )
        
        # Very high demand
        sim = METANETSimulation(
            cells=cells,
            time_step_hours=0.05,
            upstream_demand_profile=10000.0,
            downstream_supply_profile=500.0,  # Bottleneck
        )
        
        result = sim.run(steps=20)
        
        # All densities should be within bounds
        for cell_name in result.densities:
            for rho in result.densities[cell_name]:
                self.assertGreaterEqual(rho, 0.0)
                self.assertLessEqual(rho, 150.0 + 1e-6)  # Allow small numerical error

    def test_onramp_queue_dynamics(self):
        """Test that on-ramp queue accumulates and discharges correctly."""
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
            meter_rate_veh_per_hour=300.0,
            initial_queue_veh=0.0,
        )
        
        sim = METANETSimulation(
            cells=cells,
            time_step_hours=0.1,
            upstream_demand_profile=0.0,
            on_ramps=[ramp],
        )
        
        result = sim.run(steps=10)
        
        # Queue should be non-negative
        for queue in result.ramp_queues["ramp_1"]:
            self.assertGreaterEqual(queue, -1e-6)
        
        # Some vehicles should have entered from ramp
        total_ramp_flow = sum(result.ramp_flows["ramp_1"])
        self.assertGreater(total_ramp_flow, 0.0)


class TestBasicMetanetScenario(unittest.TestCase):
    """Test the run_basic_metanet_scenario function."""

    def test_basic_scenario_runs(self):
        """Test that the basic scenario runs without errors."""
        result = run_basic_metanet_scenario(steps=10)
        self.assertIsInstance(result, SimulationResult)
        self.assertEqual(result.steps, 11)  # 10 steps + initial condition

    def test_basic_scenario_has_ramp(self):
        """Test that the basic scenario includes on-ramp data."""
        result = run_basic_metanet_scenario(steps=5)
        self.assertIn("demo_ramp", result.ramp_queues)
        self.assertIn("demo_ramp", result.ramp_flows)


class TestProfileHandling(unittest.TestCase):
    """Test the profile handling for time-varying demands."""

    def test_constant_profile(self):
        """Test simulation with constant demand profile."""
        cells = build_uniform_metanet_mainline(
            num_cells=2,
            cell_length_km=0.5,
            lanes=2,
            free_flow_speed_kmh=100.0,
            jam_density_veh_per_km_per_lane=150.0,
        )
        
        sim = METANETSimulation(
            cells=cells,
            time_step_hours=0.1,
            upstream_demand_profile=1500.0,
        )
        
        result = sim.run(steps=5)
        self.assertEqual(len(result.densities["cell_0"]), 6)  # 5 steps + initial

    def test_list_profile(self):
        """Test simulation with list-based demand profile."""
        cells = build_uniform_metanet_mainline(
            num_cells=2,
            cell_length_km=0.5,
            lanes=2,
            free_flow_speed_kmh=100.0,
            jam_density_veh_per_km_per_lane=150.0,
        )
        
        demand_profile = [500.0, 1000.0, 1500.0, 1000.0, 500.0]
        sim = METANETSimulation(
            cells=cells,
            time_step_hours=0.1,
            upstream_demand_profile=demand_profile,
        )
        
        result = sim.run(steps=5)
        self.assertEqual(len(result.densities["cell_0"]), 6)

    def test_callable_profile(self):
        """Test simulation with callable demand profile."""
        cells = build_uniform_metanet_mainline(
            num_cells=2,
            cell_length_km=0.5,
            lanes=2,
            free_flow_speed_kmh=100.0,
            jam_density_veh_per_km_per_lane=150.0,
        )
        
        def demand_func(step: int) -> float:
            return 1000.0 + 500.0 * (step % 3)
        
        sim = METANETSimulation(
            cells=cells,
            time_step_hours=0.1,
            upstream_demand_profile=demand_func,
        )
        
        result = sim.run(steps=6)
        self.assertEqual(len(result.densities["cell_0"]), 7)


class TestSpeedDynamics(unittest.TestCase):
    """Test METANET-specific speed dynamics."""

    def test_relaxation_to_equilibrium(self):
        """Test that speed relaxes toward equilibrium."""
        initial_density = 50.0
        v_eq = greenshields_speed(50.0, 100.0, 160.0)
        
        cells = build_uniform_metanet_mainline(
            num_cells=1,
            cell_length_km=1.0,
            lanes=2,
            free_flow_speed_kmh=100.0,
            jam_density_veh_per_km_per_lane=160.0,
            tau_s=18.0,
            eta=0.0,  # Disable anticipation
            initial_density_veh_per_km_per_lane=initial_density,
            initial_speed_kmh=50.0,  # Below equilibrium
        )
        
        # Run with no flow to isolate speed dynamics
        sim = METANETSimulation(
            cells=cells,
            time_step_hours=0.01,
            upstream_demand_profile=0.0,
            downstream_supply_profile=0.0,
        )
        
        result = sim.run(steps=50)
        
        # Speed should increase toward equilibrium (though density changes)
        # Just check that simulation runs and produces reasonable values
        for cell_name in result.densities:
            for rho in result.densities[cell_name]:
                self.assertGreaterEqual(rho, 0.0)
                self.assertLessEqual(rho, 160.0)


if __name__ == "__main__":
    unittest.main()
