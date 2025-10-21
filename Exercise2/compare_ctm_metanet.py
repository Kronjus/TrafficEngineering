"""Comparison script for CTM and METANET simulations.

This module provides utilities to compare CTM and METANET simulations
side-by-side under similar conditions. It helps validate the METANET
implementation and understand the differences between the two models.
"""

from typing import Tuple

from ctm_simulation import (
    CTMSimulation,
    CellConfig,
    OnRampConfig,
    SimulationResult,
    build_uniform_mainline,
)
from metanet_simulation import (
    METANETSimulation,
    METANETCellConfig,
    METANETOnRampConfig,
    build_uniform_metanet_mainline,
)

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def create_comparable_cells(
    num_cells: int = 5,
    cell_length_km: float = 0.5,
    lanes: int = 3,
    free_flow_speed_kmh: float = 100.0,
    jam_density: float = 160.0,
) -> Tuple[list, list]:
    """Create comparable CTM and METANET cell configurations.
    
    Parameters
    ----------
    num_cells : int
        Number of cells to create
    cell_length_km : float
        Length of each cell in km
    lanes : int
        Number of lanes per cell
    free_flow_speed_kmh : float
        Free-flow speed in km/h
    jam_density : float
        Jam density in veh/km/lane
    
    Returns
    -------
    tuple
        (ctm_cells, metanet_cells) - Lists of configured cells
    """
    # CTM cells
    ctm_cells = build_uniform_mainline(
        num_cells=num_cells,
        cell_length_km=cell_length_km,
        lanes=lanes,
        free_flow_speed_kmh=free_flow_speed_kmh,
        congestion_wave_speed_kmh=20.0,
        capacity_veh_per_hour_per_lane=2200.0,
        jam_density_veh_per_km_per_lane=jam_density,
    )
    
    # METANET cells
    metanet_cells = build_uniform_metanet_mainline(
        num_cells=num_cells,
        cell_length_km=cell_length_km,
        lanes=lanes,
        free_flow_speed_kmh=free_flow_speed_kmh,
        jam_density_veh_per_km_per_lane=jam_density,
        tau_s=18.0,
        eta=60.0,
        kappa=40.0,
        capacity_veh_per_hour_per_lane=2200.0,
    )
    
    return ctm_cells, metanet_cells


def compare_basic_scenario(steps: int = 30) -> Tuple[SimulationResult, SimulationResult]:
    """Compare CTM and METANET on a basic scenario.
    
    Parameters
    ----------
    steps : int
        Number of simulation steps
    
    Returns
    -------
    tuple
        (ctm_result, metanet_result)
    """
    print("\n" + "="*70)
    print("COMPARISON: Basic Freeway Scenario")
    print("="*70)
    
    # Time-varying demand profile
    def demand_profile(step: int) -> float:
        if step < 10:
            return 3000.0
        elif step < 20:
            return 6000.0  # Peak
        else:
            return 4000.0
    
    ctm_cells, metanet_cells = create_comparable_cells(num_cells=5)
    
    # CTM simulation
    print("\nRunning CTM simulation...")
    ctm_sim = CTMSimulation(
        cells=ctm_cells,
        time_step_hours=0.05,
        upstream_demand_profile=demand_profile,
        downstream_supply_profile=7000.0,
    )
    ctm_result = ctm_sim.run(steps=steps)
    
    # METANET simulation
    print("Running METANET simulation...")
    metanet_sim = METANETSimulation(
        cells=metanet_cells,
        time_step_hours=0.05,
        upstream_demand_profile=demand_profile,
        downstream_supply_profile=7000.0,
    )
    metanet_result = metanet_sim.run(steps=steps)
    
    # Print comparison statistics
    print("\nComparison Statistics:")
    print(f"  Time steps: {steps}")
    print(f"  Duration: {ctm_result.duration_hours:.2f} hours")
    
    # Compare final densities
    print("\nFinal densities (veh/km/lane):")
    print("  Cell    CTM     METANET   Difference")
    print("  " + "-"*40)
    for i in range(len(ctm_cells)):
        cell_name = f"cell_{i}"
        ctm_final = ctm_result.densities[cell_name][-1]
        metanet_final = metanet_result.densities[cell_name][-1]
        diff = metanet_final - ctm_final
        print(f"  {cell_name}   {ctm_final:6.2f}  {metanet_final:6.2f}  {diff:+7.2f}")
    
    if HAS_MATPLOTLIB:
        plot_comparison(ctm_result, metanet_result, "basic_scenario")
    
    return ctm_result, metanet_result


def compare_onramp_scenario(steps: int = 30) -> Tuple[SimulationResult, SimulationResult]:
    """Compare CTM and METANET with on-ramp merging.
    
    Parameters
    ----------
    steps : int
        Number of simulation steps
    
    Returns
    -------
    tuple
        (ctm_result, metanet_result)
    """
    print("\n" + "="*70)
    print("COMPARISON: On-Ramp Merging Scenario")
    print("="*70)
    
    ctm_cells, metanet_cells = create_comparable_cells(
        num_cells=4,
        lanes=3
    )
    
    # Create comparable on-ramps
    ctm_ramp = OnRampConfig(
        target_cell=2,
        arrival_rate_profile=800.0,
        meter_rate_veh_per_hour=500.0,
        mainline_priority=0.6,
        initial_queue_veh=10.0,
    )
    
    metanet_ramp = METANETOnRampConfig(
        target_cell=2,
        arrival_rate_profile=800.0,
        meter_rate_veh_per_hour=500.0,
        mainline_priority=0.6,
        initial_queue_veh=10.0,
    )
    
    # CTM simulation
    print("\nRunning CTM simulation...")
    ctm_sim = CTMSimulation(
        cells=ctm_cells,
        time_step_hours=0.1,
        upstream_demand_profile=4000.0,
        downstream_supply_profile=5000.0,
        on_ramps=[ctm_ramp],
    )
    ctm_result = ctm_sim.run(steps=steps)
    
    # METANET simulation
    print("Running METANET simulation...")
    metanet_sim = METANETSimulation(
        cells=metanet_cells,
        time_step_hours=0.1,
        upstream_demand_profile=4000.0,
        downstream_supply_profile=5000.0,
        on_ramps=[metanet_ramp],
    )
    metanet_result = metanet_sim.run(steps=steps)
    
    # Print comparison
    print("\nRamp Queue Comparison:")
    ctm_ramp_name = list(ctm_result.ramp_queues.keys())[0]
    metanet_ramp_name = list(metanet_result.ramp_queues.keys())[0]
    
    ctm_max_queue = max(ctm_result.ramp_queues[ctm_ramp_name])
    metanet_max_queue = max(metanet_result.ramp_queues[metanet_ramp_name])
    
    ctm_total_flow = sum(ctm_result.ramp_flows[ctm_ramp_name])
    metanet_total_flow = sum(metanet_result.ramp_flows[metanet_ramp_name])
    
    print(f"  Max queue - CTM: {ctm_max_queue:.1f} veh, METANET: {metanet_max_queue:.1f} veh")
    print(f"  Total ramp flow - CTM: {ctm_total_flow:.0f}, METANET: {metanet_total_flow:.0f}")
    
    if HAS_MATPLOTLIB:
        plot_comparison(ctm_result, metanet_result, "onramp_scenario")
    
    return ctm_result, metanet_result


def compare_conservation(steps: int = 20) -> None:
    """Test and compare conservation properties in both models.
    
    Parameters
    ----------
    steps : int
        Number of simulation steps
    """
    print("\n" + "="*70)
    print("COMPARISON: Conservation of Vehicles")
    print("="*70)
    
    initial_density = 50.0
    
    # CTM setup
    ctm_cells = build_uniform_mainline(
        num_cells=5,
        cell_length_km=1.0,
        lanes=2,
        free_flow_speed_kmh=100.0,
        congestion_wave_speed_kmh=20.0,
        capacity_veh_per_hour_per_lane=2200.0,
        jam_density_veh_per_km_per_lane=160.0,
        initial_density_veh_per_km_per_lane=initial_density,
    )
    
    # METANET setup
    metanet_cells = build_uniform_metanet_mainline(
        num_cells=5,
        cell_length_km=1.0,
        lanes=2,
        free_flow_speed_kmh=100.0,
        jam_density_veh_per_km_per_lane=160.0,
        initial_density_veh_per_km_per_lane=initial_density,
    )
    
    # Closed system: no inflow, no outflow
    print("\nRunning closed system simulations (no inflow/outflow)...")
    
    ctm_sim = CTMSimulation(
        cells=ctm_cells,
        time_step_hours=0.05,
        upstream_demand_profile=0.0,
        downstream_supply_profile=0.0,
    )
    ctm_result = ctm_sim.run(steps=steps)
    
    metanet_sim = METANETSimulation(
        cells=metanet_cells,
        time_step_hours=0.05,
        upstream_demand_profile=0.0,
        downstream_supply_profile=0.0,
    )
    metanet_result = metanet_sim.run(steps=steps)
    
    # Calculate total vehicles
    def total_vehicles(result, cells):
        total = 0.0
        for idx, cell_name in enumerate(sorted(result.densities.keys())):
            final_density = result.densities[cell_name][-1]
            cell = cells[idx]
            total += final_density * cell.length_km * cell.lanes
        return total
    
    ctm_initial = initial_density * sum(c.length_km * c.lanes for c in ctm_cells)
    ctm_final = total_vehicles(ctm_result, ctm_cells)
    
    metanet_initial = initial_density * sum(c.length_km * c.lanes for c in metanet_cells)
    metanet_final = total_vehicles(metanet_result, metanet_cells)
    
    print("\nConservation Results:")
    print(f"  CTM:")
    print(f"    Initial vehicles: {ctm_initial:.2f}")
    print(f"    Final vehicles:   {ctm_final:.2f}")
    print(f"    Change:           {ctm_final - ctm_initial:+.2f} ({100*(ctm_final-ctm_initial)/ctm_initial:+.2f}%)")
    
    print(f"  METANET:")
    print(f"    Initial vehicles: {metanet_initial:.2f}")
    print(f"    Final vehicles:   {metanet_final:.2f}")
    print(f"    Change:           {metanet_final - metanet_initial:+.2f} ({100*(metanet_final-metanet_initial)/metanet_initial:+.2f}%)")


def plot_comparison(ctm_result, metanet_result, scenario_name):
    """Plot side-by-side comparison of CTM and METANET results.
    
    Parameters
    ----------
    ctm_result : SimulationResult
        CTM simulation results
    metanet_result : SimulationResult
        METANET simulation results
    scenario_name : str
        Name for the output file
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ctm_time = ctm_result.time_vector()
    metanet_time = metanet_result.time_vector()
    
    # CTM densities
    ax = axes[0, 0]
    for cell_name in sorted(ctm_result.densities.keys()):
        ax.plot(ctm_time, ctm_result.densities[cell_name], label=cell_name, linewidth=2)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Density (veh/km/lane)')
    ax.set_title('CTM: Cell Densities')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # METANET densities
    ax = axes[0, 1]
    for cell_name in sorted(metanet_result.densities.keys()):
        ax.plot(metanet_time, metanet_result.densities[cell_name], label=cell_name, linewidth=2)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Density (veh/km/lane)')
    ax.set_title('METANET: Cell Densities')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # CTM flows
    ax = axes[1, 0]
    ctm_intervals = ctm_result.interval_vector()
    ctm_interval_times = [t[0] for t in ctm_intervals]
    for cell_name in sorted(ctm_result.flows.keys()):
        ax.plot(ctm_interval_times, ctm_result.flows[cell_name], label=cell_name, linewidth=2)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Flow (veh/h)')
    ax.set_title('CTM: Cell Outflows')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # METANET flows
    ax = axes[1, 1]
    metanet_intervals = metanet_result.interval_vector()
    metanet_interval_times = [t[0] for t in metanet_intervals]
    for cell_name in sorted(metanet_result.flows.keys()):
        ax.plot(metanet_interval_times, metanet_result.flows[cell_name], label=cell_name, linewidth=2)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Flow (veh/h)')
    ax.set_title('METANET: Cell Outflows')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'/tmp/comparison_{scenario_name}.png'
    plt.savefig(filename, dpi=100)
    print(f"\nComparison plot saved to: {filename}")
    plt.close()


def main():
    """Run all comparison scenarios."""
    print("\n" + "="*70)
    print("CTM vs METANET COMPARISON")
    print("="*70)
    print("\nThis script compares CTM and METANET simulations under")
    print("comparable conditions to validate the METANET implementation")
    print("and understand model differences.")
    
    # Run comparisons
    compare_basic_scenario(steps=30)
    compare_onramp_scenario(steps=30)
    compare_conservation(steps=20)
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETED")
    print("="*70)
    
    if HAS_MATPLOTLIB:
        print("\nComparison plots have been saved to /tmp/")
    else:
        print("\nTip: Install matplotlib to generate comparison plots:")
        print("  pip install matplotlib")


if __name__ == "__main__":
    main()
