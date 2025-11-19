"""Debug CTM behavior step by step."""

import numpy as np
from ctm_simulation import simulate_ctm, create_uniform_parameters


def debug_simple_case():
    """Debug with simplest possible case - single cell, no ramps."""
    print("Debug: Single cell, constant inflow")
    print("=" * 60)
    
    params = create_uniform_parameters(
        n_cells=1,
        cell_length_km=1.0,
        lanes=3,
        T=0.01,  # Reduced from 0.1 to satisfy CFL
        Q_max=2000.0,
        v_f=100.0,
        rho_crit=30.0,
        rho_jam=150.0,
        lambda_ramp=1.0,
    )
    
    rho0 = np.array([20.0])
    N0 = np.array([])
    K = 20
    d = np.zeros((K, 0))
    s = np.zeros((K, 1))
    q_upstream = np.full(K, 4000.0)
    
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
    
    print(f"Initial density: {rho0[0]:.2f} veh/km/lane")
    print(f"Upstream flow: {q_upstream[0]:.2f} veh/h")
    print(f"Cell capacity: {params.lambda_lanes[0] * params.Q_max:.2f} veh/h")
    print()
    
    print("Time step evolution:")
    print(f"{'k':>3} {'rho':>8} {'q_out':>8} {'v':>8} {'inflow':>8} {'outflow':>8} {'delta_rho':>10}")
    print("-" * 60)
    
    for k in range(min(10, K)):
        rho = results.rho[k, 0]
        rho_next = results.rho[k+1, 0]
        q_out = results.q[k, 0]
        v = results.v[k, 0]
        q_in = q_upstream[k]
        
        # Manual calculation of delta_rho
        delta_rho = (params.T / (params.L[0] * params.lambda_lanes[0])) * (q_in - q_out)
        expected_next = rho + delta_rho
        
        print(f"{k:3d} {rho:8.2f} {q_out:8.2f} {v:8.2f} {q_in:8.2f} {q_out:8.2f} {delta_rho:10.4f}")
        print(f"     Expected next: {expected_next:.2f}, Actual next: {rho_next:.2f}")
    
    print()
    print(f"Final density: {results.rho[-1, 0]:.2f} veh/km/lane")
    print()


def debug_two_cells():
    """Debug two-cell scenario."""
    print("Debug: Two cells, constant inflow")
    print("=" * 60)
    
    params = create_uniform_parameters(
        n_cells=2,
        cell_length_km=0.5,
        lanes=3,
        T=0.005,  # Reduced to satisfy CFL
        Q_max=2000.0,
        v_f=100.0,
        rho_crit=30.0,
        rho_jam=150.0,
        lambda_ramp=1.0,
    )
    
    rho0 = np.array([20.0, 20.0])
    N0 = np.array([])
    K = 15
    d = np.zeros((K, 0))
    s = np.zeros((K, 2))
    q_upstream = np.full(K, 5000.0)
    
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
    
    print(f"Initial densities: {rho0}")
    print(f"Upstream flow: {q_upstream[0]:.2f} veh/h")
    print(f"Cell capacity: {params.lambda_lanes[0] * params.Q_max:.2f} veh/h")
    print()
    
    print("Evolution:")
    print(f"{'k':>3} {'rho0':>8} {'rho1':>8} {'q0':>8} {'q1':>8} {'v0':>8} {'v1':>8}")
    print("-" * 60)
    
    for k in range(min(15, K)):
        print(f"{k:3d} {results.rho[k, 0]:8.2f} {results.rho[k, 1]:8.2f} "
              f"{results.q[k, 0]:8.2f} {results.q[k, 1]:8.2f} "
              f"{results.v[k, 0]:8.2f} {results.v[k, 1]:8.2f}")
    
    print()


if __name__ == "__main__":
    debug_simple_case()
    print()
    debug_two_cells()
