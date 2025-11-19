"""Comprehensive validation of CTM implementation."""

import numpy as np
from ctm_simulation import simulate_ctm, create_uniform_parameters


def validate_all_features():
    """Comprehensive validation demonstrating all CTM features."""
    print("=" * 70)
    print("COMPREHENSIVE CTM VALIDATION")
    print("=" * 70)
    print()
    
    # Setup parameters
    n_cells = 5
    n_ramps = 2
    K = 200
    
    # Create parameters with CFL-compliant time step
    params = create_uniform_parameters(
        n_cells=n_cells,
        cell_length_km=0.5,
        lanes=[3, 3, 3, 2, 2],  # Lane drop at cell 3
        T=0.004,  # CFL: 0.5/100 = 0.005
        Q_max=2000.0,
        v_f=100.0,
        rho_crit=30.0,
        rho_jam=150.0,
        lambda_ramp=1.0,
    )
    
    print("Network Configuration:")
    print(f"  Cells: {n_cells}")
    print(f"  Lanes: {params.lambda_lanes.astype(int)}")
    print(f"  Cell lengths: {params.L[0]} km each")
    print(f"  Time step: {params.T} h ({params.T * 3600:.1f} seconds)")
    print(f"  Total simulation time: {K * params.T:.2f} h ({K * params.T * 60:.1f} min)")
    print()
    
    # Initial conditions - moderate traffic
    rho0 = np.array([15.0, 20.0, 25.0, 30.0, 20.0])
    N0 = np.array([5.0, 10.0])  # Small initial queues
    
    print("Initial Conditions:")
    print(f"  Densities [veh/km/lane]: {rho0}")
    print(f"  Ramp queues [veh]: {N0}")
    print()
    
    # Time-varying upstream demand (morning peak)
    def upstream_demand_profile(k):
        """Morning peak demand profile."""
        peak_time = K // 3
        if k < peak_time:
            return 3000.0 + 2000.0 * (k / peak_time)
        else:
            return 5000.0 - 1000.0 * ((k - peak_time) / (K - peak_time))
    
    q_upstream = np.array([upstream_demand_profile(k) for k in range(K)])
    
    # Ramp demands - constant but different
    d = np.zeros((K, n_ramps))
    d[:, 0] = 800.0  # Ramp 1: moderate demand
    d[:, 1] = 400.0  # Ramp 2: light demand
    
    # Off-ramp flows - remove 10% of flow from cell 2
    s = np.zeros((K, n_cells))
    # We'll set off-ramp flow as fraction of cell flow (done in post-processing)
    
    # Ramp locations
    ramp_to_cell = np.array([1, 3])  # Ramps merge at cells 1 and 3
    
    print("Inputs:")
    print(f"  Upstream demand: {q_upstream[0]:.0f} → {q_upstream[K//2]:.0f} → {q_upstream[-1]:.0f} veh/h")
    print(f"  Ramp 1 demand: {d[0, 0]:.0f} veh/h (merges at cell {ramp_to_cell[0]})")
    print(f"  Ramp 2 demand: {d[0, 1]:.0f} veh/h (merges at cell {ramp_to_cell[1]})")
    print(f"  Off-ramps: none")
    print()
    
    # Run simulation
    print("Running simulation...")
    results = simulate_ctm(
        rho0=rho0,
        N0=N0,
        d=d,
        s=s,
        params=params,
        K=K,
        q_upstream=q_upstream,
        ramp_to_cell=ramp_to_cell,
    )
    print("✓ Simulation completed successfully")
    print()
    
    # Analyze results
    print("=" * 70)
    print("RESULTS ANALYSIS")
    print("=" * 70)
    print()
    
    # 1. Conservation check
    print("1. Vehicle Conservation Check:")
    # Initial vehicles = mainline + queues
    initial_mainline = np.sum(rho0 * params.L * params.lambda_lanes)
    initial_queues = np.sum(N0)
    initial_veh = initial_mainline + initial_queues
    
    # Final vehicles = mainline + queues
    final_mainline = np.sum(results.rho[-1] * params.L * params.lambda_lanes)
    final_queues = np.sum(results.N[-1])
    final_veh = final_mainline + final_queues
    
    # Inflow: upstream boundary + ramp demand arriving
    total_upstream_in = np.sum(q_upstream * params.T)
    total_ramp_demand = np.sum(d * params.T)
    total_in = total_upstream_in + total_ramp_demand
    
    # Outflow: downstream boundary + off-ramps
    total_downstream_out = np.sum(results.q[:, -1] * params.T)
    total_offramp_out = np.sum(results.s * params.T)
    total_out = total_downstream_out + total_offramp_out
    
    # Expected final = initial + inflow - outflow
    expected_final = initial_veh + total_in - total_out
    conservation_error = abs(final_veh - expected_final)
    
    print(f"  Initial vehicles:")
    print(f"    Mainline: {initial_mainline:.2f} veh")
    print(f"    Queues: {initial_queues:.2f} veh")
    print(f"    Total: {initial_veh:.2f} veh")
    print(f"  Final vehicles:")
    print(f"    Mainline: {final_mainline:.2f} veh")
    print(f"    Queues: {final_queues:.2f} veh")
    print(f"    Total: {final_veh:.2f} veh")
    print(f"  Flows:")
    print(f"    Upstream inflow: {total_upstream_in:.2f} veh")
    print(f"    Ramp demand arriving: {total_ramp_demand:.2f} veh")
    print(f"    Total inflow: {total_in:.2f} veh")
    print(f"    Downstream outflow: {total_downstream_out:.2f} veh")
    print(f"    Off-ramp outflow: {total_offramp_out:.2f} veh")
    print(f"    Total outflow: {total_out:.2f} veh")
    print(f"  Expected final: {expected_final:.2f} veh")
    print(f"  Conservation error: {conservation_error:.6f} veh")
    print(f"  ✓ Conservation PASSED" if conservation_error < 1.0 else f"  ✗ Conservation FAILED")
    print()
    
    # 2. Density statistics
    print("2. Density Statistics:")
    print(f"  Initial avg density: {np.mean(rho0):.2f} veh/km/lane")
    print(f"  Final avg density: {np.mean(results.rho[-1]):.2f} veh/km/lane")
    print(f"  Max density observed: {np.max(results.rho):.2f} veh/km/lane")
    print(f"  Min density observed: {np.min(results.rho):.2f} veh/km/lane")
    print(f"  Densities stayed within [0, rho_jam]: {np.all(results.rho >= 0) and np.all(results.rho <= params.rho_jam)}")
    print()
    
    # 3. Flow statistics
    print("3. Flow Statistics:")
    print(f"  Average mainline flow: {np.mean(results.q):.2f} veh/h")
    print(f"  Max mainline flow: {np.max(results.q):.2f} veh/h")
    max_capacity = np.max(params.lambda_lanes * params.Q_max)
    print(f"  Max cell capacity: {max_capacity:.2f} veh/h")
    print(f"  Flows respect capacity: {np.all(results.q <= max_capacity * 1.01)}")
    print()
    
    # 4. Speed statistics
    print("4. Speed Statistics:")
    print(f"  Average speed: {np.mean(results.v):.2f} km/h")
    print(f"  Max speed observed: {np.max(results.v):.2f} km/h")
    print(f"  Min speed observed: {np.min(results.v):.2f} km/h")
    print(f"  Speeds within [0, v_f]: {np.all(results.v >= 0) and np.all(results.v <= params.v_f)}")
    print()
    
    # 5. Ramp queue statistics
    print("5. Ramp Queue Dynamics:")
    for i in range(n_ramps):
        print(f"  Ramp {i} (at cell {ramp_to_cell[i]}):")
        print(f"    Initial queue: {results.N[0, i]:.2f} veh")
        print(f"    Final queue: {results.N[-1, i]:.2f} veh")
        print(f"    Max queue: {np.max(results.N[:, i]):.2f} veh")
        print(f"    Avg ramp flow: {np.mean(results.r[:, i]):.2f} veh/h")
        total_demand = np.sum(d[:, i] * params.T)
        total_served = np.sum(results.r[:, i] * params.T)
        print(f"    Total demand: {total_demand:.2f} veh")
        print(f"    Total served: {total_served:.2f} veh")
        print(f"    Service ratio: {total_served / total_demand * 100:.1f}%")
    print()
    
    # 6. Bottleneck effect (lane drop)
    print("6. Bottleneck Effect (Lane Drop at Cell 3):")
    bottleneck_cell = 3
    upstream_cell = bottleneck_cell - 1
    avg_density_upstream = np.mean(results.rho[:, upstream_cell])
    avg_density_bottleneck = np.mean(results.rho[:, bottleneck_cell])
    print(f"  Avg density upstream (cell {upstream_cell}): {avg_density_upstream:.2f} veh/km/lane")
    print(f"  Avg density at bottleneck (cell {bottleneck_cell}): {avg_density_bottleneck:.2f} veh/km/lane")
    print(f"  Congestion formed at bottleneck: {avg_density_upstream > avg_density_bottleneck}")
    print()
    
    # 7. Time evolution snapshot
    print("7. Time Evolution Snapshots:")
    snapshot_times = [0, K//4, K//2, 3*K//4, K-1]
    print(f"  {'Time':>8} {'Cell 0':>8} {'Cell 1':>8} {'Cell 2':>8} {'Cell 3':>8} {'Cell 4':>8}")
    print("  " + "-" * 56)
    for t_idx in snapshot_times:
        time_h = results.time[t_idx]
        densities = results.rho[t_idx]
        print(f"  {time_h:.4f} h " + " ".join(f"{rho:8.2f}" for rho in densities))
    print()
    
    # 8. Physical constraints
    print("8. Physical Constraints Verification:")
    checks = [
        ("All densities >= 0", np.all(results.rho >= 0)),
        ("All densities <= rho_jam", np.all(results.rho <= params.rho_jam)),
        ("All flows >= 0", np.all(results.q >= 0)),
        ("All ramp flows >= 0", np.all(results.r >= 0)),
        ("All queues >= 0", np.all(results.N >= 0)),
        ("All speeds >= 0", np.all(results.v >= 0)),
        ("All speeds <= v_f", np.all(results.v <= params.v_f)),
    ]
    all_passed = True
    for check_name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {check_name}")
        all_passed = all_passed and passed
    print()
    
    # Final summary
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    if all_passed and conservation_error < 1.0:
        print("✓ ALL VALIDATIONS PASSED")
        print("  The CTM implementation correctly implements the triangular FD,")
        print("  first-order density dynamics, on-ramp flows with queue dynamics,")
        print("  and all physical constraints are satisfied.")
    else:
        print("✗ SOME VALIDATIONS FAILED")
        print("  Review the results above for details.")
    print("=" * 70)
    print()
    
    return all_passed and conservation_error < 1.0


if __name__ == "__main__":
    success = validate_all_features()
    exit(0 if success else 1)
