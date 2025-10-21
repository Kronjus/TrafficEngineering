"""Demonstration of ALINEA ramp metering control.

This script compares scenarios with and without ALINEA control to show
the benefits of feedback-based ramp metering. It also demonstrates the
new features: configurable measurement cell and automatic gain tuning.

Usage:
    python alinea_demo.py
"""

from ctm_simulation import (
    build_uniform_mainline,
    CTMSimulation,
    OnRampConfig,
)
from metanet_simulation import (
    build_uniform_metanet_mainline,
    METANETSimulation,
    METANETOnRampConfig,
)


def run_without_alinea():
    """Run scenario with fixed-rate metering."""
    print("\n" + "=" * 70)
    print("Scenario 1: Fixed-Rate Ramp Metering (No ALINEA)")
    print("=" * 70)
    
    cells = build_uniform_mainline(
        num_cells=5,
        cell_length_km=0.5,
        lanes=3,
        free_flow_speed_kmh=100.0,
        congestion_wave_speed_kmh=20.0,
        capacity_veh_per_hour_per_lane=2200.0,
        jam_density_veh_per_km_per_lane=160.0,
        initial_density_veh_per_km_per_lane=20.0,
    )

    # High demand on-ramp with fixed metering rate
    on_ramp = OnRampConfig(
        target_cell=2,
        arrival_rate_profile=1200.0,  # High arrival demand
        meter_rate_veh_per_hour=800.0,  # Fixed metering rate
        mainline_priority=0.7,
    )

    sim = CTMSimulation(
        cells=cells,
        time_step_hours=0.05,  # 3 minutes
        upstream_demand_profile=5000.0,  # High mainline demand
        on_ramps=[on_ramp],
    )

    result = sim.run(steps=60)  # 3 hours of simulation
    
    # Extract key metrics
    max_queue = max(result.ramp_queues["ramp_2"])
    avg_queue = sum(result.ramp_queues["ramp_2"]) / len(result.ramp_queues["ramp_2"])
    max_density_cell2 = max(result.densities["cell_2"])
    
    print(f"\nResults:")
    print(f"  Maximum ramp queue: {max_queue:.1f} vehicles")
    print(f"  Average ramp queue: {avg_queue:.1f} vehicles")
    print(f"  Maximum density at merge cell: {max_density_cell2:.1f} veh/km/lane")
    print(f"  Metering rate (fixed): 800 veh/h")
    
    return result


def run_with_alinea():
    """Run scenario with ALINEA feedback control."""
    print("\n" + "=" * 70)
    print("Scenario 2: ALINEA Feedback Ramp Metering")
    print("=" * 70)
    
    cells = build_uniform_mainline(
        num_cells=5,
        cell_length_km=0.5,
        lanes=3,
        free_flow_speed_kmh=100.0,
        congestion_wave_speed_kmh=20.0,
        capacity_veh_per_hour_per_lane=2200.0,
        jam_density_veh_per_km_per_lane=160.0,
        initial_density_veh_per_km_per_lane=20.0,
    )

    # Same demand, but with ALINEA control
    on_ramp = OnRampConfig(
        target_cell=2,
        arrival_rate_profile=1200.0,  # Same high arrival demand
        meter_rate_veh_per_hour=800.0,  # Initial rate (will be adjusted)
        mainline_priority=0.7,
        alinea_enabled=True,
        alinea_gain=60.0,  # Control gain
        alinea_target_density=100.0,  # Target density (veh/km/lane)
        alinea_min_rate=240.0,
        alinea_max_rate=1800.0,
    )

    sim = CTMSimulation(
        cells=cells,
        time_step_hours=0.05,  # 3 minutes
        upstream_demand_profile=5000.0,  # Same high mainline demand
        on_ramps=[on_ramp],
    )

    result = sim.run(steps=60)  # 3 hours of simulation
    
    # Extract key metrics
    max_queue = max(result.ramp_queues["ramp_2"])
    avg_queue = sum(result.ramp_queues["ramp_2"]) / len(result.ramp_queues["ramp_2"])
    max_density_cell2 = max(result.densities["cell_2"])
    final_rate = on_ramp.meter_rate_veh_per_hour
    
    print(f"\nResults:")
    print(f"  Maximum ramp queue: {max_queue:.1f} vehicles")
    print(f"  Average ramp queue: {avg_queue:.1f} vehicles")
    print(f"  Maximum density at merge cell: {max_density_cell2:.1f} veh/km/lane")
    print(f"  Final metering rate: {final_rate:.1f} veh/h")
    print(f"  Target density: {on_ramp.alinea_target_density:.1f} veh/km/lane")
    
    return result


def run_with_custom_measurement_cell():
    """Run scenario with ALINEA using a custom measurement cell."""
    print("\n" + "=" * 70)
    print("Scenario 3: ALINEA with Custom Measurement Cell")
    print("=" * 70)
    print("Demonstrates measuring density downstream of merge point.")
    
    cells = build_uniform_metanet_mainline(
        num_cells=5,
        cell_length_km=0.5,
        lanes=3,
        free_flow_speed_kmh=100.0,
        jam_density_veh_per_km_per_lane=160.0,
        initial_density_veh_per_km_per_lane=30.0,
    )

    # Ramp merges at cell 2, but measures density at cell 3 (downstream)
    on_ramp = METANETOnRampConfig(
        target_cell=2,
        arrival_rate_profile=1000.0,
        meter_rate_veh_per_hour=700.0,
        alinea_enabled=True,
        alinea_gain=55.0,
        alinea_target_density=90.0,
        alinea_measurement_cell=3,  # Measure downstream cell
        alinea_min_rate=240.0,
        alinea_max_rate=1800.0,
    )

    sim = METANETSimulation(
        cells=cells,
        time_step_hours=0.05,
        upstream_demand_profile=4500.0,
        on_ramps=[on_ramp],
    )

    result = sim.run(steps=60)
    
    max_queue = max(result.ramp_queues["ramp_2"])
    avg_queue = sum(result.ramp_queues["ramp_2"]) / len(result.ramp_queues["ramp_2"])
    max_density_cell3 = max(result.densities["cell_3"])
    final_rate = on_ramp.meter_rate_veh_per_hour
    
    print(f"\nResults:")
    print(f"  Merge cell: 2, Measurement cell: {on_ramp.alinea_measurement_cell}")
    print(f"  Maximum ramp queue: {max_queue:.1f} vehicles")
    print(f"  Average ramp queue: {avg_queue:.1f} vehicles")
    print(f"  Maximum density at measurement cell: {max_density_cell3:.1f} veh/km/lane")
    print(f"  Final metering rate: {final_rate:.1f} veh/h")
    print(f"  Target density: {on_ramp.alinea_target_density:.1f} veh/km/lane")
    
    return result


def run_with_auto_tuning():
    """Run scenario with automatic ALINEA gain tuning."""
    print("\n" + "=" * 70)
    print("Scenario 4: ALINEA with Automatic Gain Tuning")
    print("=" * 70)
    print("Demonstrates auto-tuning to find optimal gain K_R.")
    
    # Create base scenario
    cells = build_uniform_metanet_mainline(
        num_cells=4,
        cell_length_km=0.5,
        lanes=3,
        free_flow_speed_kmh=100.0,
        jam_density_veh_per_km_per_lane=160.0,
        initial_density_veh_per_km_per_lane=40.0,
    )

    on_ramp = METANETOnRampConfig(
        target_cell=2,
        arrival_rate_profile=900.0,
        meter_rate_veh_per_hour=650.0,
        alinea_enabled=True,
        alinea_gain=30.0,  # Initial guess (will be tuned)
        alinea_target_density=80.0,
        alinea_min_rate=240.0,
        alinea_max_rate=1800.0,
    )

    sim = METANETSimulation(
        cells=cells,
        time_step_hours=0.05,
        upstream_demand_profile=4000.0,
        on_ramps=[on_ramp],
    )

    # Define simulator function for tuning
    def evaluate_gain(K: float) -> dict:
        """Run simulation with given gain and return performance metrics."""
        test_cells = build_uniform_metanet_mainline(
            num_cells=4,
            cell_length_km=0.5,
            lanes=3,
            free_flow_speed_kmh=100.0,
            jam_density_veh_per_km_per_lane=160.0,
            initial_density_veh_per_km_per_lane=40.0,
        )

        test_ramp = METANETOnRampConfig(
            target_cell=2,
            arrival_rate_profile=900.0,
            meter_rate_veh_per_hour=650.0,
            alinea_enabled=True,
            alinea_gain=K,  # Test this gain
            alinea_target_density=80.0,
            alinea_min_rate=240.0,
            alinea_max_rate=1800.0,
        )

        test_sim = METANETSimulation(
            cells=test_cells,
            time_step_hours=0.05,
            upstream_demand_profile=4000.0,
            on_ramps=[test_ramp],
        )

        # Run short evaluation
        result = test_sim.run(steps=40)

        # Calculate RMSE from target density
        target_density = 80.0
        densities = result.densities["cell_2"]
        squared_errors = [(d - target_density) ** 2 for d in densities]
        rmse = (sum(squared_errors) / len(squared_errors)) ** 0.5

        return {'rmse': rmse}

    print("\nRunning auto-tuner...")
    initial_gain = on_ramp.alinea_gain
    
    tuner_params = {
        'K_min': 10.0,
        'K_max': 90.0,
        'n_grid': 10,
        'objective': 'rmse',
        'verbose': True,  # Show tuning progress
    }

    best_K = sim.auto_tune_alinea_gain(
        ramp_index=0,
        simulator_fn=evaluate_gain,
        tuner_params=tuner_params,
    )

    print(f"\nTuning complete:")
    print(f"  Initial gain: {initial_gain:.1f}")
    print(f"  Optimal gain: {best_K:.1f}")
    
    # Run final simulation with tuned gain
    print("\nRunning final simulation with tuned gain...")
    result = sim.run(steps=60)
    
    max_queue = max(result.ramp_queues["ramp_2"])
    avg_queue = sum(result.ramp_queues["ramp_2"]) / len(result.ramp_queues["ramp_2"])
    max_density = max(result.densities["cell_2"])
    
    print(f"\nFinal Results:")
    print(f"  Maximum ramp queue: {max_queue:.1f} vehicles")
    print(f"  Average ramp queue: {avg_queue:.1f} vehicles")
    print(f"  Maximum density at merge cell: {max_density:.1f} veh/km/lane")
    print(f"  Target density: {on_ramp.alinea_target_density:.1f} veh/km/lane")
    
    return result


def run_with_slew_limit():
    """Run scenario with ALINEA and adjustable slew rate limiting."""
    print("\n" + "=" * 70)
    print("Scenario 5: ALINEA with Slew Rate Limiting")
    print("=" * 70)
    print("Demonstrates smooth rate changes with slew limit constraint.")
    
    cells = build_uniform_mainline(
        num_cells=4,
        cell_length_km=0.5,
        lanes=3,
        free_flow_speed_kmh=100.0,
        congestion_wave_speed_kmh=20.0,
        capacity_veh_per_hour_per_lane=2200.0,
        jam_density_veh_per_km_per_lane=160.0,
        initial_density_veh_per_km_per_lane=120.0,  # High initial density
    )

    # ALINEA with slew limit for smooth transitions
    on_ramp = OnRampConfig(
        target_cell=2,
        arrival_rate_profile=900.0,
        meter_rate_veh_per_hour=1000.0,  # Start high
        mainline_priority=0.7,
        alinea_enabled=True,
        alinea_gain=150.0,  # High gain for aggressive control
        alinea_target_density=70.0,  # Lower target
        alinea_min_rate=240.0,
        alinea_max_rate=1800.0,
        alinea_slew_limit=400.0,  # Limit rate changes to 400 veh/h²
    )

    sim = CTMSimulation(
        cells=cells,
        time_step_hours=0.05,  # 3 minutes
        upstream_demand_profile=4500.0,
        on_ramps=[on_ramp],
    )

    print(f"\nInitial conditions:")
    print(f"  Initial metering rate: {on_ramp.meter_rate_veh_per_hour:.1f} veh/h")
    print(f"  Initial density: 120 veh/km/lane")
    print(f"  Target density: {on_ramp.alinea_target_density:.1f} veh/km/lane")
    print(f"  Slew limit: {on_ramp.alinea_slew_limit:.1f} veh/h²")
    print(f"  Max change per step: {on_ramp.alinea_slew_limit * sim.dt:.1f} veh/h")
    
    # Track rate evolution
    rates = [on_ramp.meter_rate_veh_per_hour]
    result = sim.run(steps=20)
    
    # Collect rate history (approximate from final values)
    # In real usage, you'd store rates at each step
    final_rate = on_ramp.meter_rate_veh_per_hour
    
    max_queue = max(result.ramp_queues["ramp_2"])
    avg_queue = sum(result.ramp_queues["ramp_2"]) / len(result.ramp_queues["ramp_2"])
    max_density_cell2 = max(result.densities["cell_2"])
    
    print(f"\nResults after 20 steps:")
    print(f"  Final metering rate: {final_rate:.1f} veh/h")
    print(f"  Rate change: {final_rate - rates[0]:.1f} veh/h")
    print(f"  Maximum ramp queue: {max_queue:.1f} vehicles")
    print(f"  Average ramp queue: {avg_queue:.1f} vehicles")
    print(f"  Maximum density at merge cell: {max_density_cell2:.1f} veh/km/lane")
    
    print(f"\nSlew limit effect:")
    print(f"  Without slew limit: Rate could change by ~{150.0 * (70.0 - 120.0):.0f} veh/h instantly")
    print(f"  With slew limit: Rate changes gradually by max {on_ramp.alinea_slew_limit * sim.dt:.1f} veh/h per step")
    print(f"  Result: Smoother control action, better driver compliance")
    
    return result


def compare_scenarios():
    """Compare fixed metering vs ALINEA control."""
    print("\n" + "=" * 70)
    print("ALINEA Ramp Metering Demonstration")
    print("=" * 70)
    print("\nThis demo compares fixed-rate metering with ALINEA feedback control")
    print("and demonstrates new features: custom measurement cells, auto-tuning,")
    print("and slew rate limiting.")
    
    result_fixed = run_without_alinea()
    result_alinea = run_with_alinea()
    result_custom = run_with_custom_measurement_cell()
    result_tuned = run_with_auto_tuning()
    result_slew = run_with_slew_limit()
    
    print("\n" + "=" * 70)
    print("Summary Comparison")
    print("=" * 70)
    
    fixed_max_queue = max(result_fixed.ramp_queues["ramp_2"])
    alinea_max_queue = max(result_alinea.ramp_queues["ramp_2"])
    
    fixed_max_density = max(result_fixed.densities["cell_2"])
    alinea_max_density = max(result_alinea.densities["cell_2"])
    
    print(f"\nRamp Queue (vehicles):")
    print(f"  Fixed metering:    {fixed_max_queue:6.1f}")
    print(f"  ALINEA control:    {alinea_max_queue:6.1f}")
    
    print(f"\nMerge Cell Density (veh/km/lane):")
    print(f"  Fixed metering:    {fixed_max_density:6.1f}")
    print(f"  ALINEA control:    {alinea_max_density:6.1f}")
    
    print("\nNew ALINEA Features Demonstrated:")
    print("  ✓ Configurable measurement cell (Scenario 3)")
    print("    - Allows measuring density at a different cell than merge point")
    print("    - Useful for anticipating downstream congestion")
    print("\n  ✓ Automatic gain tuning (Scenario 4)")
    print("    - Grid search over K values to find optimal gain")
    print("    - Minimizes density tracking error (RMSE)")
    print("    - Can use custom objective functions")
    print("\n  ✓ Adjustable slew rate limiting (Scenario 5)")
    print("    - Constrains how quickly metering rate can change")
    print("    - Prevents abrupt changes and improves stability")
    print("    - Mimics physical limitations of metering systems")
    print("\n  ✓ Configurable r_min and r_max limits")
    print("    - Already demonstrated in all ALINEA scenarios")
    print("    - Ensures safe metering rate bounds")
    
    print("\nALINEA Benefits:")
    print("  • Prevents mainline congestion by adjusting to traffic conditions")
    print("  • Balances ramp queue growth with mainline capacity")
    print("  • Automatically adapts to changing demand patterns")
    print("  • Maintains target density for optimal throughput")
    
    print("\n" + "=" * 70)
    print("To visualize results, modify this script to plot densities and queues")
    print("over time using matplotlib.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    compare_scenarios()
