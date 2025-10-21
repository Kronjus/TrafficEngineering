"""Example scenarios for the METANET simulation.

This module demonstrates various use cases of the METANET macroscopic traffic
flow simulator, including:
- Basic freeway simulation with time-varying demand
- On-ramp merging with queue dynamics
- Lane drops and capacity bottlenecks
- Ramp metering strategies
- Speed dynamics and wave propagation

Run this script directly to execute all examples, or import individual
functions to run specific scenarios.
"""

from metanet_simulation import (
    METANETCellConfig,
    METANETSimulation,
    METANETOnRampConfig,
    build_uniform_metanet_mainline,
)

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: matplotlib not available. Install with 'pip install matplotlib' for visualizations.")


def example_1_basic_freeway():
    """Example 1: Basic freeway with time-varying demand.
    
    Demonstrates:
    - Simple 5-cell mainline freeway
    - Time-varying upstream demand (triangular profile)
    - No on-ramps
    - METANET speed dynamics and congestion wave propagation
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Freeway with Time-Varying Demand")
    print("="*70)
    
    # Create a 5-cell mainline
    cells = build_uniform_metanet_mainline(
        num_cells=5,
        cell_length_km=0.5,
        lanes=3,
        free_flow_speed_kmh=100.0,
        jam_density_veh_per_km_per_lane=160.0,
        tau_s=18.0,
        eta=60.0,
        kappa=40.0,
        capacity_veh_per_hour_per_lane=2200.0,
    )
    
    # Time-varying demand: ramp up, peak, ramp down
    def triangular_demand(step: int) -> float:
        peak = 6000.0  # Near capacity
        if step < 10:
            return peak * (step / 10.0)
        elif step < 20:
            return peak
        else:
            return max(0.0, peak - 300.0 * (step - 20))
    
    sim = METANETSimulation(
        cells=cells,
        time_step_hours=0.05,  # 3 minutes
        upstream_demand_profile=triangular_demand,
        downstream_supply_profile=7000.0,  # High downstream capacity
    )
    
    result = sim.run(steps=40)
    
    print(f"\nSimulated duration: {result.duration_hours:.2f} hours")
    print(f"Number of cells: {len(cells)}")
    print(f"Time step: {result.time_step_hours * 60:.1f} minutes")
    
    # Print summary statistics
    print("\nFinal densities (veh/km/lane):")
    for cell_name in sorted(result.densities.keys()):
        final_density = result.densities[cell_name][-1]
        print(f"  {cell_name}: {final_density:.2f}")
    
    if HAS_MATPLOTLIB:
        plot_example_1(result)
    
    return result


def example_2_onramp_merge():
    """Example 2: Freeway with single on-ramp.
    
    Demonstrates:
    - 4-cell mainline with lane drop
    - On-ramp merging at cell 2
    - Time-varying ramp demand
    - Queue dynamics without metering
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Freeway with On-Ramp Merge")
    print("="*70)
    
    # Create mainline with lane drop
    cells = build_uniform_metanet_mainline(
        num_cells=4,
        cell_length_km=0.5,
        lanes=[3, 3, 2, 2],  # Lane drop at cell 2
        free_flow_speed_kmh=100.0,
        jam_density_veh_per_km_per_lane=160.0,
        tau_s=18.0,
        eta=60.0,
        capacity_veh_per_hour_per_lane=2200.0,
    )
    
    # On-ramp with time-varying demand (rush hour pattern)
    def ramp_demand(step: int) -> float:
        if step < 10:
            return 200.0
        elif step < 25:
            return 800.0  # Heavy ramp demand
        else:
            return 300.0
    
    on_ramp = METANETOnRampConfig(
        target_cell=2,
        arrival_rate_profile=ramp_demand,
        mainline_priority=0.6,  # Favor mainline
        initial_queue_veh=5.0,
    )
    
    sim = METANETSimulation(
        cells=cells,
        time_step_hours=0.05,
        upstream_demand_profile=4500.0,  # Moderate mainline demand
        downstream_supply_profile=5000.0,
        on_ramps=[on_ramp],
    )
    
    result = sim.run(steps=35)
    
    print(f"\nSimulated duration: {result.duration_hours:.2f} hours")
    print(f"Lane configuration: {[c.lanes for c in cells]}")
    
    # Print ramp statistics
    ramp_name = list(result.ramp_queues.keys())[0]
    max_queue = max(result.ramp_queues[ramp_name])
    avg_queue = sum(result.ramp_queues[ramp_name]) / len(result.ramp_queues[ramp_name])
    total_ramp_flow = sum(result.ramp_flows[ramp_name]) * result.time_step_hours
    
    print(f"\nOn-ramp statistics:")
    print(f"  Max queue: {max_queue:.1f} vehicles")
    print(f"  Avg queue: {avg_queue:.1f} vehicles")
    print(f"  Total vehicles merged: {total_ramp_flow:.0f} vehicles")
    
    if HAS_MATPLOTLIB:
        plot_example_2(result, cells)
    
    return result


def example_3_ramp_metering():
    """Example 3: On-ramp with metering control.
    
    Demonstrates:
    - Ramp metering to limit merge flow
    - Comparison of metered vs unmetered scenarios
    - Queue buildup due to metering
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Ramp Metering Strategy")
    print("="*70)
    
    cells = build_uniform_metanet_mainline(
        num_cells=3,
        cell_length_km=0.5,
        lanes=3,
        free_flow_speed_kmh=100.0,
        jam_density_veh_per_km_per_lane=160.0,
        tau_s=18.0,
        eta=60.0,
        capacity_veh_per_hour_per_lane=2200.0,
    )
    
    # High constant ramp demand
    ramp_demand = 1200.0
    
    print("\nScenario A: No metering")
    ramp_no_meter = METANETOnRampConfig(
        target_cell=1,
        arrival_rate_profile=ramp_demand,
        meter_rate_veh_per_hour=None,  # No metering
        mainline_priority=0.5,
    )
    
    sim_no_meter = METANETSimulation(
        cells=cells,
        time_step_hours=0.1,
        upstream_demand_profile=5000.0,
        on_ramps=[ramp_no_meter],
    )
    
    result_no_meter = sim_no_meter.run(steps=20)
    
    print("\nScenario B: With metering (600 veh/h)")
    ramp_with_meter = METANETOnRampConfig(
        target_cell=1,
        arrival_rate_profile=ramp_demand,
        meter_rate_veh_per_hour=600.0,  # Metering rate
        mainline_priority=0.5,
    )
    
    sim_with_meter = METANETSimulation(
        cells=cells,
        time_step_hours=0.1,
        upstream_demand_profile=5000.0,
        on_ramps=[ramp_with_meter],
    )
    
    result_with_meter = sim_with_meter.run(steps=20)
    
    # Compare results
    ramp_name = list(result_no_meter.ramp_queues.keys())[0]
    
    print(f"\nComparison:")
    print(f"  No metering - Final queue: {result_no_meter.ramp_queues[ramp_name][-1]:.1f} veh")
    print(f"  With metering - Final queue: {result_with_meter.ramp_queues[ramp_name][-1]:.1f} veh")
    
    total_flow_no_meter = sum(result_no_meter.ramp_flows[ramp_name])
    total_flow_with_meter = sum(result_with_meter.ramp_flows[ramp_name])
    
    print(f"  No metering - Total ramp flow: {total_flow_no_meter:.0f} veh")
    print(f"  With metering - Total ramp flow: {total_flow_with_meter:.0f} veh")
    
    if HAS_MATPLOTLIB:
        plot_example_3(result_no_meter, result_with_meter)
    
    return result_no_meter, result_with_meter


def example_4_speed_dynamics():
    """Example 4: METANET speed dynamics demonstration.
    
    Demonstrates:
    - Relaxation to equilibrium speed
    - Convection term effects
    - Anticipation term effects
    - Stop-and-go wave formation
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: METANET Speed Dynamics")
    print("="*70)
    
    # Create longer mainline to observe wave propagation
    cells = build_uniform_metanet_mainline(
        num_cells=10,
        cell_length_km=0.4,
        lanes=3,
        free_flow_speed_kmh=110.0,
        jam_density_veh_per_km_per_lane=165.0,
        tau_s=18.0,
        eta=60.0,
        kappa=40.0,
        capacity_veh_per_hour_per_lane=2100.0,
        initial_density_veh_per_km_per_lane=80.0,  # Moderate initial density
    )
    
    # Pulse demand to trigger wave
    def pulse_demand(step: int) -> float:
        if 5 <= step < 15:
            return 7000.0  # High demand pulse
        else:
            return 4000.0  # Normal demand
    
    sim = METANETSimulation(
        cells=cells,
        time_step_hours=0.05,
        upstream_demand_profile=pulse_demand,
        downstream_supply_profile=6000.0,
    )
    
    result = sim.run(steps=40)
    
    print(f"\nSimulated duration: {result.duration_hours:.2f} hours")
    print(f"Mainline length: {sum(c.length_km for c in cells):.1f} km")
    
    # Print density range
    print("\nDensity ranges (veh/km/lane):")
    for cell_name in sorted(result.densities.keys()):
        densities = result.densities[cell_name]
        print(f"  {cell_name}: {min(densities):.1f} - {max(densities):.1f}")
    
    if HAS_MATPLOTLIB:
        plot_example_4(result, cells)
    
    return result


def example_5_multiple_ramps():
    """Example 5: Complex scenario with multiple on-ramps.
    
    Demonstrates:
    - Multiple on-ramps at different locations
    - Lane additions and drops
    - Variable capacity sections
    - Coordinated metering
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Complex Multi-Ramp Scenario")
    print("="*70)
    
    # Create cells with varying lane counts
    lane_config = [2, 2, 3, 3, 3, 3, 3, 2, 2, 2]
    cells = build_uniform_metanet_mainline(
        num_cells=10,
        cell_length_km=0.4,
        lanes=lane_config,
        free_flow_speed_kmh=110.0,
        jam_density_veh_per_km_per_lane=165.0,
        tau_s=18.0,
        eta=60.0,
        kappa=40.0,
        capacity_veh_per_hour_per_lane=2100.0,
    )
    
    # Two on-ramps
    ramp_1 = METANETOnRampConfig(
        target_cell=2,
        arrival_rate_profile=400.0,
        mainline_priority=0.7,
        name="early_ramp",
    )
    
    ramp_2 = METANETOnRampConfig(
        target_cell=6,
        arrival_rate_profile=600.0,
        meter_rate_veh_per_hour=500.0,
        mainline_priority=0.65,
        name="late_ramp",
    )
    
    # Time-varying upstream demand
    def peak_hour_demand(step: int) -> float:
        if step < 5:
            return 3000.0
        elif step < 20:
            return 5500.0  # Peak
        else:
            return 3500.0
    
    sim = METANETSimulation(
        cells=cells,
        time_step_hours=0.05,
        upstream_demand_profile=peak_hour_demand,
        downstream_supply_profile=6000.0,
        on_ramps=[ramp_1, ramp_2],
    )
    
    result = sim.run(steps=30)
    
    print(f"\nSimulated duration: {result.duration_hours:.2f} hours")
    print(f"Mainline length: {sum(c.length_km for c in cells):.1f} km")
    print(f"Lane configuration: {lane_config}")
    
    # Print statistics for both ramps
    for ramp_name in sorted(result.ramp_queues.keys()):
        max_queue = max(result.ramp_queues[ramp_name])
        total_flow = sum(result.ramp_flows[ramp_name]) * result.time_step_hours
        print(f"\n{ramp_name}:")
        print(f"  Max queue: {max_queue:.1f} vehicles")
        print(f"  Total vehicles merged: {total_flow:.0f} vehicles")
    
    if HAS_MATPLOTLIB:
        plot_example_5(result, cells)
    
    return result


# Plotting functions
def plot_example_1(result):
    """Create visualizations for Example 1."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    time = result.time_vector()
    
    # Plot densities
    ax = axes[0]
    for cell_name in sorted(result.densities.keys()):
        ax.plot(time, result.densities[cell_name], label=cell_name, linewidth=2)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Density (veh/km/lane)')
    ax.set_title('METANET: Cell Densities Over Time')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Plot flows
    ax = axes[1]
    intervals = result.interval_vector()
    interval_times = [t[0] for t in intervals]
    
    for cell_name in sorted(result.flows.keys()):
        ax.plot(interval_times, result.flows[cell_name], label=cell_name, linewidth=2)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Flow (veh/h)')
    ax.set_title('METANET: Cell Outflows Over Time')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/metanet_example_1.png', dpi=100)
    print("\nPlot saved to: /tmp/metanet_example_1.png")
    plt.close()


def plot_example_2(result, cells):
    """Create visualizations for Example 2."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    time = result.time_vector()
    
    # Plot densities with lane info
    ax = axes[0]
    for idx, cell_name in enumerate(sorted(result.densities.keys())):
        lane_count = cells[idx].lanes
        label = f"{cell_name} ({lane_count}L)"
        ax.plot(time, result.densities[cell_name], label=label, linewidth=2)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Density (veh/km/lane)')
    ax.set_title('METANET: Cell Densities (with Lane Configuration)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Plot ramp queue and flow
    ax = axes[1]
    ramp_name = list(result.ramp_queues.keys())[0]
    ax.plot(time, result.ramp_queues[ramp_name], 'r-', linewidth=2, label='Queue')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Queue Length (vehicles)', color='r')
    ax.tick_params(axis='y', labelcolor='r')
    ax.grid(True, alpha=0.3)
    
    ax2 = ax.twinx()
    intervals = result.interval_vector()
    interval_times = [t[0] for t in intervals]
    ax2.plot(interval_times, result.ramp_flows[ramp_name], 'b-', linewidth=2, label='Flow')
    ax2.set_ylabel('Ramp Flow (veh/h)', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    ax.set_title('METANET: On-Ramp Queue and Flow')
    
    # Plot mainline flows
    ax = axes[2]
    for cell_name in sorted(result.flows.keys()):
        ax.plot(interval_times, result.flows[cell_name], label=cell_name, linewidth=2)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Flow (veh/h)')
    ax.set_title('METANET: Mainline Outflows')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/metanet_example_2.png', dpi=100)
    print("\nPlot saved to: /tmp/metanet_example_2.png")
    plt.close()


def plot_example_3(result_no_meter, result_with_meter):
    """Create visualizations for Example 3."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    time_no = result_no_meter.time_vector()
    time_with = result_with_meter.time_vector()
    
    ramp_name_no = list(result_no_meter.ramp_queues.keys())[0]
    ramp_name_with = list(result_with_meter.ramp_queues.keys())[0]
    
    # Densities comparison
    ax = axes[0, 0]
    for cell_name in sorted(result_no_meter.densities.keys()):
        ax.plot(time_no, result_no_meter.densities[cell_name], '--', label=f"{cell_name} (no meter)")
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Density (veh/km/lane)')
    ax.set_title('METANET Densities - No Metering')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    for cell_name in sorted(result_with_meter.densities.keys()):
        ax.plot(time_with, result_with_meter.densities[cell_name], '-', label=f"{cell_name} (metered)")
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Density (veh/km/lane)')
    ax.set_title('METANET Densities - With Metering')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Queue comparison
    ax = axes[1, 0]
    ax.plot(time_no, result_no_meter.ramp_queues[ramp_name_no], 'r-', linewidth=2, label='No metering')
    ax.plot(time_with, result_with_meter.ramp_queues[ramp_name_with], 'b-', linewidth=2, label='With metering')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Queue Length (vehicles)')
    ax.set_title('METANET: On-Ramp Queue Comparison')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Flow comparison
    ax = axes[1, 1]
    intervals_no = result_no_meter.interval_vector()
    intervals_with = result_with_meter.interval_vector()
    interval_times_no = [t[0] for t in intervals_no]
    interval_times_with = [t[0] for t in intervals_with]
    
    ax.plot(interval_times_no, result_no_meter.ramp_flows[ramp_name_no], 'r-', linewidth=2, label='No metering')
    ax.plot(interval_times_with, result_with_meter.ramp_flows[ramp_name_with], 'b-', linewidth=2, label='With metering')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Ramp Flow (veh/h)')
    ax.set_title('METANET: On-Ramp Flow Comparison')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/metanet_example_3.png', dpi=100)
    print("\nPlot saved to: /tmp/metanet_example_3.png")
    plt.close()


def plot_example_4(result, cells):
    """Create visualizations for Example 4."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    time = result.time_vector()
    intervals = result.interval_vector()
    interval_times = [t[0] for t in intervals]
    
    # Space-time density plot
    ax = fig.add_subplot(gs[0, :])
    cell_names = sorted(result.densities.keys())
    for idx, cell_name in enumerate(cell_names):
        color = plt.cm.viridis(idx / len(cell_names))
        ax.plot(time, result.densities[cell_name], color=color, linewidth=2, label=cell_name)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Density (veh/km/lane)')
    ax.set_title('METANET: Mainline Densities - Wave Propagation')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Flows at selected locations
    ax = fig.add_subplot(gs[1, 0])
    selected_cells = ['cell_0', 'cell_4', 'cell_9']
    for cell_name in selected_cells:
        if cell_name in result.flows:
            ax.plot(interval_times, result.flows[cell_name], linewidth=2, label=cell_name)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Flow (veh/h)')
    ax.set_title('METANET: Selected Mainline Outflows')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Average density over time
    ax = fig.add_subplot(gs[1, 1])
    avg_densities = []
    for t_idx in range(len(time)):
        avg = sum(result.densities[cn][t_idx] for cn in cell_names) / len(cell_names)
        avg_densities.append(avg)
    ax.plot(time, avg_densities, 'b-', linewidth=2)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Average Density (veh/km/lane)')
    ax.set_title('METANET: Network-Wide Average Density')
    ax.grid(True, alpha=0.3)
    
    # Heatmap-style density plot
    ax = fig.add_subplot(gs[2, :])
    densities_matrix = []
    for cell_name in cell_names:
        densities_matrix.append(result.densities[cell_name])
    
    im = ax.imshow(densities_matrix, aspect='auto', cmap='RdYlGn_r', 
                   extent=[0, result.duration_hours, len(cells), 0])
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Cell Index')
    ax.set_title('METANET: Space-Time Density Heatmap')
    ax.set_yticks(range(len(cells)))
    ax.set_yticklabels([f'cell_{i}' for i in range(len(cells))])
    plt.colorbar(im, ax=ax, label='Density (veh/km/lane)')
    
    plt.savefig('/tmp/metanet_example_4.png', dpi=100)
    print("\nPlot saved to: /tmp/metanet_example_4.png")
    plt.close()


def plot_example_5(result, cells):
    """Create visualizations for Example 5."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    time = result.time_vector()
    intervals = result.interval_vector()
    interval_times = [t[0] for t in intervals]
    
    # Mainline densities
    ax = fig.add_subplot(gs[0, :])
    cell_names = sorted(result.densities.keys())
    for idx, cell_name in enumerate(cell_names):
        color = plt.cm.viridis(idx / len(cell_names))
        ax.plot(time, result.densities[cell_name], color=color, linewidth=2, label=cell_name)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Density (veh/km/lane)')
    ax.set_title('METANET: Mainline Densities (10 cells, 2 ramps)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Ramp queues
    ax = fig.add_subplot(gs[1, 0])
    for ramp_name in sorted(result.ramp_queues.keys()):
        ax.plot(time, result.ramp_queues[ramp_name], linewidth=2, label=ramp_name)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Queue Length (vehicles)')
    ax.set_title('METANET: On-Ramp Queues')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Ramp flows
    ax = fig.add_subplot(gs[1, 1])
    for ramp_name in sorted(result.ramp_flows.keys()):
        ax.plot(interval_times, result.ramp_flows[ramp_name], linewidth=2, label=ramp_name)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Flow (veh/h)')
    ax.set_title('METANET: On-Ramp Flows')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Mainline flows at selected locations
    ax = fig.add_subplot(gs[2, 0])
    selected_cells = ['cell_0', 'cell_4', 'cell_9']
    for cell_name in selected_cells:
        if cell_name in result.flows:
            ax.plot(interval_times, result.flows[cell_name], linewidth=2, label=cell_name)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Flow (veh/h)')
    ax.set_title('METANET: Selected Mainline Outflows')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Lane configuration diagram
    ax = fig.add_subplot(gs[2, 1])
    cell_positions = range(len(cells))
    lane_counts = [c.lanes for c in cells]
    ax.bar(cell_positions, lane_counts, color='steelblue', edgecolor='black')
    ax.set_xlabel('Cell Index')
    ax.set_ylabel('Number of Lanes')
    ax.set_title('Lane Configuration')
    ax.set_xticks(cell_positions)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.savefig('/tmp/metanet_example_5.png', dpi=100)
    print("\nPlot saved to: /tmp/metanet_example_5.png")
    plt.close()


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("METANET SIMULATION EXAMPLES")
    print("="*70)
    print("\nThis script demonstrates various features of the METANET simulator.")
    print("Examples include basic freeway operations, on-ramp merging,")
    print("ramp metering, speed dynamics, and complex multi-feature scenarios.")
    
    # Run all examples
    result1 = example_1_basic_freeway()
    result2 = example_2_onramp_merge()
    result3_no, result3_with = example_3_ramp_metering()
    result4 = example_4_speed_dynamics()
    result5 = example_5_multiple_ramps()
    
    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETED")
    print("="*70)
    
    if HAS_MATPLOTLIB:
        print("\nVisualization plots have been saved to /tmp/")
    else:
        print("\nTip: Install matplotlib to generate visualization plots:")
        print("  pip install matplotlib")


if __name__ == "__main__":
    main()
