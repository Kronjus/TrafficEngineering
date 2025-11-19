"""Validation script to demonstrate CTM behavior."""

import numpy as np
from ctm_simulation import simulate_ctm, create_uniform_parameters


def validate_basic_scenario():
    """Validate basic CTM scenario with detailed output."""
    print("=" * 70)
    print("CTM Validation: Basic Scenario")
    print("=" * 70)
    print()
    
    # Setup
    n_cells = 5
    n_ramps = 1
    K = 30
    
    params = create_uniform_parameters(
        n_cells=n_cells,
        cell_length_km=0.5,
        lanes=3,
        T=0.1,  # 6 minutes
        Q_max=2000.0,
        v_f=100.0,
        rho_crit=30.0,
        rho_jam=150.0,
        lambda_ramp=1.0,
    )
    
    print("Parameters:")
    print(f"  Cells: {n_cells}")
    print(f"  Cell length: {params.L[0]} km")
    print(f"  Lanes per cell: {params.lambda_lanes[0]}")
    print(f"  Time step: {params.T} h ({params.T * 60:.1f} min)")
    print(f"  Free-flow speed: {params.v_f} km/h")
    print(f"  Critical density: {params.rho_crit} veh/km/lane")
    print(f"  Jam density: {params.rho_jam} veh/km/lane")
    print(f"  Max capacity per lane: {params.Q_max} veh/h/lane")
    print(f"  Max capacity per cell: {params.lambda_lanes[0] * params.Q_max} veh/h")
    print()
    
    # Initial conditions
    rho0 = np.array([15.0, 20.0, 25.0, 20.0, 15.0])
    N0 = np.array([10.0])
    
    print("Initial Conditions:")
    print(f"  Densities: {rho0} veh/km/lane")
    print(f"  Ramp queue: {N0[0]} veh")
    print()
    
    # Inputs
    d = np.full((K, n_ramps), 800.0)  # Constant ramp demand
    s = np.zeros((K, n_cells))  # No off-ramps
    q_upstream = np.full(K, 4500.0)  # Constant upstream demand
    ramp_to_cell = np.array([2])  # Ramp merges at cell 2
    
    print("Inputs:")
    print(f"  Upstream demand: {q_upstream[0]} veh/h")
    print(f"  Ramp demand: {d[0, 0]} veh/h")
    print(f"  Ramp location: Cell {ramp_to_cell[0]}")
    print(f"  Off-ramp flows: 0 veh/h (all cells)")
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
    print("Simulation complete.")
    print()
    
    # Display results at key time points
    time_points = [0, K//3, 2*K//3, K]
    
    print("Results at Key Time Points:")
    print("-" * 70)
    for i, t_idx in enumerate(time_points):
        if t_idx >= len(results.time):
            t_idx = len(results.time) - 1
        time_h = results.time[t_idx]
        print(f"\nTime: {time_h:.2f} h ({time_h * 60:.1f} min)")
        print(f"  Densities [veh/km/lane]:")
        for cell_idx in range(n_cells):
            rho = results.rho[t_idx, cell_idx]
            print(f"    Cell {cell_idx}: {rho:6.2f}")
        
        if t_idx < len(results.q):
            print(f"  Flows [veh/h]:")
            for cell_idx in range(n_cells):
                q = results.q[t_idx, cell_idx] if t_idx < len(results.q) else 0
                print(f"    Cell {cell_idx}: {q:7.2f}")
            
            print(f"  Speeds [km/h]:")
            for cell_idx in range(n_cells):
                v = results.v[t_idx, cell_idx] if t_idx < len(results.v) else 0
                print(f"    Cell {cell_idx}: {v:6.2f}")
        
        print(f"  Ramp queue: {results.N[t_idx, 0]:6.2f} veh")
        if t_idx < len(results.r):
            print(f"  Ramp flow: {results.r[t_idx, 0]:6.2f} veh/h")
    
    print()
    print("-" * 70)
    
    # Summary statistics
    print("\nSummary Statistics:")
    print(f"  Average density across all cells: {np.mean(results.rho):.2f} veh/km/lane")
    print(f"  Max density observed: {np.max(results.rho):.2f} veh/km/lane")
    print(f"  Min density observed: {np.min(results.rho):.2f} veh/km/lane")
    print(f"  Average speed across all cells: {np.mean(results.v):.2f} km/h")
    print(f"  Average flow: {np.mean(results.q):.2f} veh/h")
    print(f"  Max ramp queue: {np.max(results.N):.2f} veh")
    print(f"  Total ramp vehicles served: {np.sum(results.r) * params.T:.2f} veh")
    print()
    
    # Check conservation
    initial_veh = np.sum(rho0 * params.L * params.lambda_lanes)
    final_veh = np.sum(results.rho[-1] * params.L * params.lambda_lanes)
    total_in = np.sum(q_upstream) * params.T + np.sum(results.r) * params.T
    total_out = np.sum(results.q[:, -1]) * params.T + np.sum(results.s) * params.T
    expected_final = initial_veh + total_in - total_out
    
    print("Conservation Check:")
    print(f"  Initial vehicles on mainline: {initial_veh:.2f} veh")
    print(f"  Final vehicles on mainline: {final_veh:.2f} veh")
    print(f"  Total inflow: {total_in:.2f} veh")
    print(f"  Total outflow: {total_out:.2f} veh")
    print(f"  Expected final: {expected_final:.2f} veh")
    print(f"  Conservation error: {abs(final_veh - expected_final):.4f} veh")
    
    if abs(final_veh - expected_final) < 1.0:
        print("  ✓ Conservation verified!")
    else:
        print("  ✗ Conservation violated!")
    
    print()
    print("=" * 70)
    print()


def validate_congestion_formation():
    """Validate congestion wave formation and propagation."""
    print("=" * 70)
    print("CTM Validation: Congestion Formation")
    print("=" * 70)
    print()
    
    n_cells = 4
    params = create_uniform_parameters(
        n_cells=n_cells,
        cell_length_km=0.5,
        lanes=[3, 3, 2, 2],  # Lane drop at cell 2
        T=0.05,
        Q_max=2000.0,
        v_f=100.0,
        rho_crit=30.0,
        rho_jam=150.0,
        lambda_ramp=1.0,
    )
    
    print("Scenario: Lane drop creates bottleneck")
    print(f"  Lanes: {params.lambda_lanes.astype(int)}")
    print(f"  Capacity cell 0-1: {params.lambda_lanes[0] * params.Q_max:.0f} veh/h")
    print(f"  Capacity cell 2-3: {params.lambda_lanes[2] * params.Q_max:.0f} veh/h")
    print()
    
    K = 40
    rho0 = np.array([10.0, 10.0, 10.0, 10.0])
    N0 = np.array([])
    d = np.zeros((K, 0))
    s = np.zeros((K, n_cells))
    # High upstream demand exceeding bottleneck capacity
    q_upstream = np.full(K, 5500.0)
    
    print(f"  Upstream demand: {q_upstream[0]} veh/h")
    print(f"  (Exceeds bottleneck capacity of {params.lambda_lanes[2] * params.Q_max:.0f} veh/h)")
    print()
    
    results = simulate_ctm(
        rho0=rho0,
        N0=N0,
        d=d,
        s=s,
        params=params,
        K=K,
        q_upstream=q_upstream,
        ramp_to_cell=np.array([], dtype=int),
    )
    
    print("Results:")
    print(f"  Final densities: {results.rho[-1]}")
    print(f"  Congestion formed upstream of bottleneck: {results.rho[-1, 1] > 50}")
    print()
    
    # Check if congestion builds up upstream
    if results.rho[-1, 1] > 50:
        print("  ✓ Congestion correctly forms upstream of lane drop")
    else:
        print("  ✗ Expected congestion upstream of lane drop")
    
    print()
    print("=" * 70)
    print()


if __name__ == "__main__":
    validate_basic_scenario()
    validate_congestion_formation()
