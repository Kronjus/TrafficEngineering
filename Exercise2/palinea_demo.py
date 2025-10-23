"""Demonstration of P-ALINEA controller and grid search tuning.

This script demonstrates:
1. Standard ALINEA (Kp=0)
2. P-ALINEA with proportional gain (Kp>0)
3. Grid search tuning to find optimal (Kp, Ki) gains
4. Comparison of objective J = VHT + sum(dr^2)

Usage:
    python palinea_demo.py
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
from alinea_tuning import grid_search_tune_alinea


def run_standard_alinea():
    """Run scenario with standard ALINEA (Kp=0)."""
    print("\n" + "=" * 70)
    print("Scenario 1: Standard ALINEA (Kp=0, Ki=50)")
    print("=" * 70)
    
    cells = build_uniform_mainline(
        num_cells=5,
        cell_length_km=0.5,
        lanes=3,
        free_flow_speed_kmh=100.0,
        congestion_wave_speed_kmh=20.0,
        capacity_veh_per_hour_per_lane=2200.0,
        jam_density_veh_per_km_per_lane=160.0,
        initial_density_veh_per_km_per_lane=40.0,
    )

    on_ramp = OnRampConfig(
        target_cell=2,
        arrival_rate_profile=1200.0,
        meter_rate_veh_per_hour=800.0,
        mainline_priority=0.7,
        alinea_enabled=True,
        alinea_gain=50.0,  # Ki
        alinea_proportional_gain=0.0,  # Kp = 0 (standard ALINEA)
        alinea_target_density=100.0,
        alinea_min_rate=240.0,
        alinea_max_rate=1800.0,
    )

    sim = CTMSimulation(
        cells=cells,
        time_step_hours=0.05,
        upstream_demand_profile=5000.0,
        on_ramps=[on_ramp],
    )

    result = sim.run(steps=60)
    
    # Compute metrics
    cell_lengths = {cell.name: cell.length_km for cell in cells}
    vht = result.compute_vht(cell_lengths)
    ramp_penalty = result.compute_ramp_rate_penalty()
    J = vht + ramp_penalty
    
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
    print(f"\n  Performance Metrics:")
    print(f"    VHT: {vht:.2f} veh-hours")
    print(f"    Ramp rate penalty: {ramp_penalty:.2f}")
    print(f"    Objective J = VHT + penalty: {J:.2f}")
    
    return result, J


def run_palinea_fixed_gains():
    """Run scenario with P-ALINEA using fixed gains."""
    print("\n" + "=" * 70)
    print("Scenario 2: P-ALINEA with Fixed Gains (Kp=0.3, Ki=50)")
    print("=" * 70)
    
    cells = build_uniform_mainline(
        num_cells=5,
        cell_length_km=0.5,
        lanes=3,
        free_flow_speed_kmh=100.0,
        congestion_wave_speed_kmh=20.0,
        capacity_veh_per_hour_per_lane=2200.0,
        jam_density_veh_per_km_per_lane=160.0,
        initial_density_veh_per_km_per_lane=40.0,
    )

    on_ramp = OnRampConfig(
        target_cell=2,
        arrival_rate_profile=1200.0,
        meter_rate_veh_per_hour=800.0,
        mainline_priority=0.7,
        alinea_enabled=True,
        alinea_gain=50.0,  # Ki
        alinea_proportional_gain=0.3,  # Kp > 0 (P-ALINEA)
        alinea_target_density=100.0,
        alinea_min_rate=240.0,
        alinea_max_rate=1800.0,
    )

    sim = CTMSimulation(
        cells=cells,
        time_step_hours=0.05,
        upstream_demand_profile=5000.0,
        on_ramps=[on_ramp],
    )

    result = sim.run(steps=60)
    
    # Compute metrics
    cell_lengths = {cell.name: cell.length_km for cell in cells}
    vht = result.compute_vht(cell_lengths)
    ramp_penalty = result.compute_ramp_rate_penalty()
    J = vht + ramp_penalty
    
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
    print(f"\n  Performance Metrics:")
    print(f"    VHT: {vht:.2f} veh-hours")
    print(f"    Ramp rate penalty: {ramp_penalty:.2f}")
    print(f"    Objective J = VHT + penalty: {J:.2f}")
    
    return result, J


def run_grid_search_tuning():
    """Run grid search to find optimal (Kp, Ki) gains."""
    print("\n" + "=" * 70)
    print("Scenario 3: Grid Search Tuning of P-ALINEA")
    print("=" * 70)
    print("Searching for optimal (Kp, Ki) to minimize J = VHT + sum(dr^2)")
    
    cells = build_uniform_mainline(
        num_cells=5,
        cell_length_km=0.5,
        lanes=3,
        free_flow_speed_kmh=100.0,
        congestion_wave_speed_kmh=20.0,
        capacity_veh_per_hour_per_lane=2200.0,
        jam_density_veh_per_km_per_lane=160.0,
        initial_density_veh_per_km_per_lane=40.0,
    )

    base_ramp = OnRampConfig(
        target_cell=2,
        arrival_rate_profile=1200.0,
        meter_rate_veh_per_hour=800.0,
        mainline_priority=0.7,
        alinea_enabled=True,
        alinea_gain=50.0,  # Will be tuned
        alinea_proportional_gain=0.0,  # Will be tuned
        alinea_target_density=100.0,
        alinea_min_rate=240.0,
        alinea_max_rate=1800.0,
    )
    
    # Run grid search
    print("\n")
    best_params = grid_search_tune_alinea(
        cells=cells,
        ramp_config=base_ramp,
        simulation_class=CTMSimulation,
        upstream_demand_profile=5000.0,
        time_step_hours=0.05,
        steps=60,
        kp_range=(0.0, 1.0, 0.2),  # Test Kp = 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
        ki_range=(20.0, 80.0, 20.0),  # Test Ki = 20, 40, 60, 80
        verbose=True,
    )
    
    print(f"\nBest configuration found:")
    print(f"  Kp (proportional gain): {best_params['Kp']:.3f}")
    print(f"  Ki (integral gain): {best_params['Ki']:.3f}")
    print(f"  Objective J: {best_params['J']:.2f}")
    print(f"  VHT: {best_params['VHT']:.2f}")
    print(f"  Ramp penalty: {best_params['ramp_penalty']:.2f}")
    
    return best_params


def run_with_tuned_gains(best_params):
    """Run simulation with the tuned gains."""
    print("\n" + "=" * 70)
    print("Scenario 4: P-ALINEA with Tuned Gains")
    print("=" * 70)
    print(f"Using tuned gains: Kp={best_params['Kp']:.3f}, Ki={best_params['Ki']:.3f}")
    
    cells = build_uniform_mainline(
        num_cells=5,
        cell_length_km=0.5,
        lanes=3,
        free_flow_speed_kmh=100.0,
        congestion_wave_speed_kmh=20.0,
        capacity_veh_per_hour_per_lane=2200.0,
        jam_density_veh_per_km_per_lane=160.0,
        initial_density_veh_per_km_per_lane=40.0,
    )

    on_ramp = OnRampConfig(
        target_cell=2,
        arrival_rate_profile=1200.0,
        meter_rate_veh_per_hour=800.0,
        mainline_priority=0.7,
        alinea_enabled=True,
        alinea_gain=best_params['Ki'],
        alinea_proportional_gain=best_params['Kp'],
        alinea_target_density=100.0,
        alinea_min_rate=240.0,
        alinea_max_rate=1800.0,
    )

    sim = CTMSimulation(
        cells=cells,
        time_step_hours=0.05,
        upstream_demand_profile=5000.0,
        on_ramps=[on_ramp],
    )

    result = sim.run(steps=60)
    
    max_queue = max(result.ramp_queues["ramp_2"])
    avg_queue = sum(result.ramp_queues["ramp_2"]) / len(result.ramp_queues["ramp_2"])
    max_density_cell2 = max(result.densities["cell_2"])
    final_rate = on_ramp.meter_rate_veh_per_hour
    
    print(f"\nResults:")
    print(f"  Maximum ramp queue: {max_queue:.1f} vehicles")
    print(f"  Average ramp queue: {avg_queue:.1f} vehicles")
    print(f"  Maximum density at merge cell: {max_density_cell2:.1f} veh/km/lane")
    print(f"  Final metering rate: {final_rate:.1f} veh/h")
    print(f"  Objective J: {best_params['J']:.2f}")
    
    return result


def compare_all_scenarios():
    """Compare all scenarios."""
    print("\n" + "=" * 70)
    print("P-ALINEA DEMONSTRATION")
    print("=" * 70)
    print("\nThis demo shows:")
    print("1. Standard ALINEA (Kp=0): Integral-only control")
    print("2. P-ALINEA with fixed gains: PI control with manual tuning")
    print("3. Grid search tuning: Automatic optimization of (Kp, Ki)")
    print("4. P-ALINEA with tuned gains: Using the optimal configuration")
    
    # Run all scenarios
    result_alinea, J_alinea = run_standard_alinea()
    result_palinea, J_palinea = run_palinea_fixed_gains()
    best_params = run_grid_search_tuning()
    result_tuned = run_with_tuned_gains(best_params)
    
    # Final comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    
    print(f"\nObjective J (lower is better):")
    print(f"  Standard ALINEA (Kp=0):        {J_alinea:.2f}")
    print(f"  P-ALINEA (Kp=0.3):             {J_palinea:.2f}")
    print(f"  P-ALINEA (tuned):              {best_params['J']:.2f}")
    
    improvement_fixed = ((J_alinea - J_palinea) / J_alinea) * 100
    improvement_tuned = ((J_alinea - best_params['J']) / J_alinea) * 100
    
    print(f"\nImprovement vs. standard ALINEA:")
    print(f"  P-ALINEA (fixed):  {improvement_fixed:+.1f}%")
    print(f"  P-ALINEA (tuned):  {improvement_tuned:+.1f}%")
    
    print("\nKey Features Demonstrated:")
    print("  ✓ P-ALINEA: Optional proportional term (Kp parameter)")
    print("    - Kp=0: Standard ALINEA (backward compatible)")
    print("    - Kp>0: P-ALINEA with anticipatory control")
    print()
    print("  ✓ Grid Search Tuning: Automatic (Kp, Ki) optimization")
    print("    - Objective: J = VHT + sum_k(r(k)-r(k-1))^2")
    print("    - Balances travel time and smooth metering")
    print()
    print("  ✓ Performance Metrics:")
    print("    - VHT: Vehicle Hours Traveled (congestion measure)")
    print("    - Ramp penalty: Penalizes rapid rate changes")
    print()
    print("  ✓ Metering Rate Tracking:")
    print("    - Full history stored for analysis")
    print("    - Used to compute ramp rate penalty")
    
    print("\n" + "=" * 70)
    print("To save tuning results, use:")
    print("  from alinea_tuning import save_tuning_results_csv")
    print("  save_tuning_results_csv(best_params, 'tuning_results.csv')")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    compare_all_scenarios()
