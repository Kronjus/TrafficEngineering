"""Tests for P-ALINEA controller and grid search tuning.

This test module validates:
1. P-ALINEA controller with Kp=0 behaves like standard ALINEA
2. P-ALINEA controller with Kp>0 implements proportional control
3. Grid search tuning finds reasonable (Kp, Ki) pairs
4. VHT and ramp rate penalty computation
"""

import unittest
from ctm_simulation import (
    build_uniform_mainline,
    CTMSimulation,
    OnRampConfig,
    SimulationResult,
)
from metanet_simulation import (
    build_uniform_metanet_mainline,
    METANETSimulation,
    METANETOnRampConfig,
)
from alinea_tuning import grid_search_tune_alinea, _create_kp_ki_grid


class TestPALINEA(unittest.TestCase):
    """Test P-ALINEA controller functionality."""
    
    def test_kp_zero_matches_alinea(self):
        """Test that Kp=0 behaves identically to standard ALINEA."""
        cells = build_uniform_mainline(
            num_cells=3,
            cell_length_km=0.5,
            lanes=2,
            free_flow_speed_kmh=100.0,
            congestion_wave_speed_kmh=20.0,
            capacity_veh_per_hour_per_lane=2000.0,
            jam_density_veh_per_km_per_lane=150.0,
            initial_density_veh_per_km_per_lane=30.0,
        )
        
        # Standard ALINEA (implicit Kp=0)
        ramp_alinea = OnRampConfig(
            target_cell=1,
            arrival_rate_profile=800.0,
            meter_rate_veh_per_hour=600.0,
            alinea_enabled=True,
            alinea_gain=50.0,
            alinea_target_density=80.0,
        )
        
        sim_alinea = CTMSimulation(
            cells=cells,
            time_step_hours=0.05,
            upstream_demand_profile=4000.0,
            on_ramps=[ramp_alinea],
        )
        
        result_alinea = sim_alinea.run(steps=30)
        
        # P-ALINEA with Kp=0 (explicit)
        cells2 = build_uniform_mainline(
            num_cells=3,
            cell_length_km=0.5,
            lanes=2,
            free_flow_speed_kmh=100.0,
            congestion_wave_speed_kmh=20.0,
            capacity_veh_per_hour_per_lane=2000.0,
            jam_density_veh_per_km_per_lane=150.0,
            initial_density_veh_per_km_per_lane=30.0,
        )
        
        ramp_palinea = OnRampConfig(
            target_cell=1,
            arrival_rate_profile=800.0,
            meter_rate_veh_per_hour=600.0,
            alinea_enabled=True,
            alinea_gain=50.0,
            alinea_proportional_gain=0.0,  # Explicit Kp=0
            alinea_target_density=80.0,
        )
        
        sim_palinea = CTMSimulation(
            cells=cells2,
            time_step_hours=0.05,
            upstream_demand_profile=4000.0,
            on_ramps=[ramp_palinea],
        )
        
        result_palinea = sim_palinea.run(steps=30)
        
        # Compare metering rates - they should be identical
        rates_alinea = result_alinea.ramp_meter_rates['ramp_1']
        rates_palinea = result_palinea.ramp_meter_rates['ramp_1']
        
        self.assertEqual(len(rates_alinea), len(rates_palinea))
        for r1, r2 in zip(rates_alinea, rates_palinea):
            self.assertAlmostEqual(r1, r2, places=5, 
                msg="Kp=0 should produce same rates as standard ALINEA")
    
    def test_kp_positive_affects_control(self):
        """Test that Kp>0 changes control behavior."""
        # Use conditions that create more variation in density error
        cells = build_uniform_mainline(
            num_cells=3,
            cell_length_km=0.5,
            lanes=2,
            free_flow_speed_kmh=100.0,
            congestion_wave_speed_kmh=20.0,
            capacity_veh_per_hour_per_lane=2000.0,
            jam_density_veh_per_km_per_lane=150.0,
            initial_density_veh_per_km_per_lane=50.0,  # Start further from target
        )
        
        # Use time-varying demand to create density changes
        def varying_demand(step):
            if step < 10:
                return 3500.0
            elif step < 20:
                return 5000.0  # Spike
            else:
                return 3800.0
        
        # Standard ALINEA
        ramp_alinea = OnRampConfig(
            target_cell=1,
            arrival_rate_profile=900.0,  # Higher ramp demand
            meter_rate_veh_per_hour=700.0,
            alinea_enabled=True,
            alinea_gain=50.0,
            alinea_proportional_gain=0.0,
            alinea_target_density=80.0,
        )
        
        sim_alinea = CTMSimulation(
            cells=cells,
            time_step_hours=0.05,
            upstream_demand_profile=varying_demand,
            on_ramps=[ramp_alinea],
        )
        
        result_alinea = sim_alinea.run(steps=30)
        
        # P-ALINEA with Kp>0
        cells2 = build_uniform_mainline(
            num_cells=3,
            cell_length_km=0.5,
            lanes=2,
            free_flow_speed_kmh=100.0,
            congestion_wave_speed_kmh=20.0,
            capacity_veh_per_hour_per_lane=2000.0,
            jam_density_veh_per_km_per_lane=150.0,
            initial_density_veh_per_km_per_lane=50.0,
        )
        
        ramp_palinea = OnRampConfig(
            target_cell=1,
            arrival_rate_profile=900.0,
            meter_rate_veh_per_hour=700.0,
            alinea_enabled=True,
            alinea_gain=50.0,
            alinea_proportional_gain=30.0,  # Larger Kp for more effect
            alinea_target_density=80.0,
        )
        
        sim_palinea = CTMSimulation(
            cells=cells2,
            time_step_hours=0.05,
            upstream_demand_profile=varying_demand,
            on_ramps=[ramp_palinea],
        )
        
        result_palinea = sim_palinea.run(steps=30)
        
        # Metering rates should be different
        rates_alinea = result_alinea.ramp_meter_rates['ramp_1']
        rates_palinea = result_palinea.ramp_meter_rates['ramp_1']
        
        # Check that at least some rates differ significantly
        differences = sum(1 for r1, r2 in zip(rates_alinea, rates_palinea) 
                         if abs(r1 - r2) > 5.0)  # Lower threshold
        self.assertGreater(differences, 0,
            "Kp>0 should produce different rates than Kp=0")
    
    def test_negative_kp_rejected(self):
        """Test that negative Kp is rejected."""
        with self.assertRaises(ValueError):
            OnRampConfig(
                target_cell=1,
                arrival_rate_profile=800.0,
                meter_rate_veh_per_hour=600.0,
                alinea_enabled=True,
                alinea_gain=50.0,
                alinea_proportional_gain=-10.0,  # Invalid
                alinea_target_density=80.0,
            )
    
    def test_metanet_palinea(self):
        """Test P-ALINEA with METANET simulation."""
        cells = build_uniform_metanet_mainline(
            num_cells=3,
            cell_length_km=0.5,
            lanes=2,
            free_flow_speed_kmh=100.0,
            jam_density_veh_per_km_per_lane=150.0,
            initial_density_veh_per_km_per_lane=30.0,
        )
        
        ramp = METANETOnRampConfig(
            target_cell=1,
            arrival_rate_profile=800.0,
            meter_rate_veh_per_hour=600.0,
            alinea_enabled=True,
            alinea_gain=50.0,
            alinea_proportional_gain=15.0,
            alinea_target_density=80.0,
        )
        
        sim = METANETSimulation(
            cells=cells,
            time_step_hours=0.05,
            upstream_demand_profile=4000.0,
            on_ramps=[ramp],
        )
        
        result = sim.run(steps=30)
        
        # Verify metering rates are tracked
        self.assertIn('ramp_1', result.ramp_meter_rates)
        self.assertEqual(len(result.ramp_meter_rates['ramp_1']), 31)  # Initial + 30 steps


class TestSimulationResultMetrics(unittest.TestCase):
    """Test VHT and ramp rate penalty computation."""
    
    def test_compute_vht(self):
        """Test VHT computation."""
        # Create a simple result
        result = SimulationResult(
            densities={
                'cell_0': [50.0, 60.0, 70.0],
                'cell_1': [40.0, 50.0, 60.0],
            },
            flows={'cell_0': [1000.0, 1100.0], 'cell_1': [900.0, 1000.0]},
            ramp_queues={},
            ramp_flows={},
            time_step_hours=0.1,
        )
        
        cell_lengths = {'cell_0': 1.0, 'cell_1': 1.0}
        vht = result.compute_vht(cell_lengths)
        
        # VHT should be positive
        self.assertGreater(vht, 0.0)
        
        # Manual calculation check (trapezoidal rule)
        # Cell 0: ((50+60)/2 * 1.0 * 0.1) + ((60+70)/2 * 1.0 * 0.1) = 5.5 + 6.5 = 12.0
        # Cell 1: ((40+50)/2 * 1.0 * 0.1) + ((50+60)/2 * 1.0 * 0.1) = 4.5 + 5.5 = 10.0
        # Total: 22.0
        expected = 22.0
        self.assertAlmostEqual(vht, expected, places=5)
    
    def test_compute_ramp_rate_penalty(self):
        """Test ramp rate penalty computation."""
        result = SimulationResult(
            densities={'cell_0': [50.0, 60.0, 70.0]},
            flows={'cell_0': [1000.0, 1100.0]},
            ramp_queues={'ramp_0': [10.0, 12.0, 14.0]},
            ramp_flows={'ramp_0': [500.0, 520.0]},
            ramp_meter_rates={'ramp_0': [600.0, 650.0, 700.0]},
            time_step_hours=0.1,
        )
        
        penalty = result.compute_ramp_rate_penalty()
        
        # Manual calculation: (650-600)^2 + (700-650)^2 = 2500 + 2500 = 5000
        expected = 5000.0
        self.assertAlmostEqual(penalty, expected, places=5)
    
    def test_ramp_rate_penalty_zero_for_constant(self):
        """Test that constant metering rate gives zero penalty."""
        result = SimulationResult(
            densities={'cell_0': [50.0, 60.0, 70.0]},
            flows={'cell_0': [1000.0, 1100.0]},
            ramp_queues={},
            ramp_flows={},
            ramp_meter_rates={'ramp_0': [600.0, 600.0, 600.0]},
            time_step_hours=0.1,
        )
        
        penalty = result.compute_ramp_rate_penalty()
        self.assertEqual(penalty, 0.0)


class TestGridSearchTuning(unittest.TestCase):
    """Test grid search tuning functionality."""
    
    def test_create_grid(self):
        """Test grid generation."""
        pairs = _create_kp_ki_grid(
            kp_range=(0.0, 0.2, 0.1),
            ki_range=(10.0, 30.0, 10.0),
        )
        
        # Should have 3 Kp values (0.0, 0.1, 0.2) x 3 Ki values (10, 20, 30) = 9 pairs
        self.assertEqual(len(pairs), 9)
        self.assertIn((0.0, 10.0), pairs)
        self.assertIn((0.2, 30.0), pairs)
    
    def test_grid_search_finds_reasonable_solution(self):
        """Test that grid search finds a reasonable solution."""
        cells = build_uniform_mainline(
            num_cells=3,
            cell_length_km=0.5,
            lanes=2,
            free_flow_speed_kmh=100.0,
            congestion_wave_speed_kmh=20.0,
            capacity_veh_per_hour_per_lane=2000.0,
            jam_density_veh_per_km_per_lane=150.0,
            initial_density_veh_per_km_per_lane=30.0,
        )
        
        ramp = OnRampConfig(
            target_cell=1,
            arrival_rate_profile=800.0,
            meter_rate_veh_per_hour=600.0,
            alinea_enabled=True,
            alinea_gain=50.0,  # Will be overridden
            alinea_proportional_gain=0.0,  # Will be overridden
            alinea_target_density=80.0,
        )
        
        # Small grid for fast test
        results = grid_search_tune_alinea(
            cells=cells,
            ramp_config=ramp,
            simulation_class=CTMSimulation,
            upstream_demand_profile=4000.0,
            time_step_hours=0.05,
            steps=20,  # Short simulation
            kp_range=(0.0, 0.2, 0.1),  # 3 values
            ki_range=(30.0, 50.0, 10.0),  # 3 values
            verbose=False,
        )
        
        # Verify result structure
        self.assertIn('Kp', results)
        self.assertIn('Ki', results)
        self.assertIn('J', results)
        self.assertIn('VHT', results)
        self.assertIn('ramp_penalty', results)
        self.assertIn('all_results', results)
        
        # Verify valid gains
        self.assertGreaterEqual(results['Kp'], 0.0)
        self.assertGreaterEqual(results['Ki'], 0.0)
        
        # Verify all tested combinations recorded
        self.assertEqual(len(results['all_results']), 9)
    
    def test_grid_search_with_metanet(self):
        """Test grid search with METANET simulation."""
        cells = build_uniform_metanet_mainline(
            num_cells=3,
            cell_length_km=0.5,
            lanes=2,
            free_flow_speed_kmh=100.0,
            jam_density_veh_per_km_per_lane=150.0,
            initial_density_veh_per_km_per_lane=30.0,
        )
        
        ramp = METANETOnRampConfig(
            target_cell=1,
            arrival_rate_profile=800.0,
            meter_rate_veh_per_hour=600.0,
            alinea_enabled=True,
            alinea_gain=50.0,
            alinea_proportional_gain=0.0,
            alinea_target_density=80.0,
        )
        
        results = grid_search_tune_alinea(
            cells=cells,
            ramp_config=ramp,
            simulation_class=METANETSimulation,
            upstream_demand_profile=4000.0,
            time_step_hours=0.05,
            steps=20,
            kp_range=(0.0, 0.1, 0.1),  # 2 values
            ki_range=(40.0, 50.0, 10.0),  # 2 values
            verbose=False,
        )
        
        # Verify successful tuning
        self.assertIsNotNone(results)
        self.assertEqual(len(results['all_results']), 4)


if __name__ == '__main__':
    unittest.main()
