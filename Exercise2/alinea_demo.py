"""Demonstration of ALINEA ramp metering control.

This script compares scenarios with and without ALINEA control to show
the benefits of feedback-based ramp metering.

Usage:
    python alinea_demo.py
"""

from ctm_simulation import (
    build_uniform_mainline,
    CTMSimulation,
    OnRampConfig,
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


def compare_scenarios():
    """Compare fixed metering vs ALINEA control."""
    print("\n" + "=" * 70)
    print("ALINEA Ramp Metering Demonstration")
    print("=" * 70)
    print("\nThis demo compares fixed-rate metering with ALINEA feedback control.")
    print("Both scenarios have high mainline and ramp demand to stress the system.")
    
    result_fixed = run_without_alinea()
    result_alinea = run_with_alinea()
    
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
