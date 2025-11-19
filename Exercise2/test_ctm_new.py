"""Test suite for CTM simulation implementation."""

import numpy as np
from ctm_simulation import (
    CTMParameters,
    CTMResults,
    simulate_ctm,
    create_uniform_parameters,
)


def test_conservation():
    """Test that vehicle conservation holds."""
    # Simple 3-cell scenario
    n_cells = 3
    params = create_uniform_parameters(
        n_cells=n_cells,
        cell_length_km=1.0,
        lanes=2,
        T=0.01,  # Small time step for accuracy
        Q_max=2000.0,
        v_f=100.0,
        rho_crit=30.0,
        rho_jam=150.0,
        lambda_ramp=1.0,
    )
    
    # Initial conditions - moderate density
    rho0 = np.array([25.0, 25.0, 25.0])
    N0 = np.array([])  # No ramps
    
    # No ramps, no off-ramps, constant upstream
    K = 100
    d = np.zeros((K, 0))  # No ramps
    s = np.zeros((K, n_cells))  # No off-ramps
    q_upstream = np.full(K, 4000.0)  # Below capacity
    
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
    
    # Check conservation: total vehicles should be conserved
    # Initial vehicles
    initial_vehicles = np.sum(rho0 * params.L * params.lambda_lanes)
    
    # Final vehicles
    final_vehicles = np.sum(results.rho[-1] * params.L * params.lambda_lanes)
    
    # Net inflow (upstream inflow - outflow from last cell)
    total_inflow = np.sum(q_upstream) * params.T
    total_outflow = np.sum(results.q[:, -1]) * params.T
    
    expected_final = initial_vehicles + total_inflow - total_outflow
    
    print(f"Conservation test:")
    print(f"  Initial vehicles: {initial_vehicles:.2f}")
    print(f"  Final vehicles: {final_vehicles:.2f}")
    print(f"  Expected final: {expected_final:.2f}")
    print(f"  Total inflow: {total_inflow:.2f}")
    print(f"  Total outflow: {total_outflow:.2f}")
    print(f"  Difference: {abs(final_vehicles - expected_final):.6f}")
    print(f"  Test passed: {abs(final_vehicles - expected_final) < 1.0}")
    print()
    
    return abs(final_vehicles - expected_final) < 1.0


def test_steady_state():
    """Test steady-state behavior with constant inputs."""
    n_cells = 3
    params = create_uniform_parameters(
        n_cells=n_cells,
        cell_length_km=0.5,
        lanes=3,
        T=0.05,
        Q_max=2000.0,
        v_f=100.0,
        rho_crit=30.0,
        rho_jam=150.0,
        lambda_ramp=1.0,
    )
    
    # Start at equilibrium density
    rho_eq = 20.0  # Below critical
    rho0 = np.full(n_cells, rho_eq)
    N0 = np.array([])
    
    # Constant upstream flow matching equilibrium
    q_eq = params.v_f * rho_eq * params.lambda_lanes[0]
    
    K = 100
    d = np.zeros((K, 0))
    s = np.zeros((K, n_cells))
    q_upstream = np.full(K, q_eq)
    
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
    
    # Check that densities remain near initial values
    final_densities = results.rho[-1]
    max_deviation = np.max(np.abs(final_densities - rho0))
    
    print(f"Steady state test:")
    print(f"  Initial density: {rho_eq:.2f}")
    print(f"  Final densities: {final_densities}")
    print(f"  Max deviation: {max_deviation:.4f}")
    print(f"  Test passed: {max_deviation < 5.0}")
    print()
    
    return max_deviation < 5.0


def test_ramp_queue():
    """Test ramp queue dynamics."""
    n_cells = 2
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
    
    # Start with empty cells
    rho0 = np.array([10.0, 10.0])
    N0 = np.array([0.0])  # Empty queue initially
    
    K = 20
    # High ramp demand that cannot all be served
    ramp_demand = 3000.0  # Exceeds ramp capacity
    d = np.full((K, 1), ramp_demand)
    s = np.zeros((K, n_cells))
    q_upstream = np.full(K, 3000.0)
    
    # Ramp to cell 0
    ramp_to_cell = np.array([0])
    
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
    
    # Queue should build up when demand exceeds service rate
    initial_queue = results.N[0, 0]
    final_queue = results.N[-1, 0]
    
    print(f"Ramp queue test:")
    print(f"  Initial queue: {initial_queue:.2f} veh")
    print(f"  Final queue: {final_queue:.2f} veh")
    print(f"  Queue built up: {final_queue > initial_queue}")
    print(f"  Test passed: {final_queue > 10.0}")
    print()
    
    return final_queue > 10.0


def test_capacity_constraint():
    """Test that flows respect capacity constraints."""
    n_cells = 2
    params = create_uniform_parameters(
        n_cells=n_cells,
        cell_length_km=0.5,
        lanes=2,
        T=0.1,
        Q_max=2000.0,
        v_f=100.0,
        rho_crit=30.0,
        rho_jam=150.0,
        lambda_ramp=1.0,
    )
    
    max_capacity = params.lambda_lanes[0] * params.Q_max
    
    rho0 = np.array([40.0, 40.0])
    N0 = np.array([])
    
    K = 50
    d = np.zeros((K, 0))
    s = np.zeros((K, n_cells))
    # Very high upstream demand
    q_upstream = np.full(K, 10000.0)
    
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
    
    # All flows should be below capacity
    max_flow = np.max(results.q)
    
    print(f"Capacity constraint test:")
    print(f"  Max capacity: {max_capacity:.2f} veh/h")
    print(f"  Max observed flow: {max_flow:.2f} veh/h")
    print(f"  Capacity respected: {max_flow <= max_capacity * 1.01}")
    print(f"  Test passed: {max_flow <= max_capacity * 1.01}")
    print()
    
    return max_flow <= max_capacity * 1.01


def test_free_flow():
    """Test free-flow behavior at low densities."""
    n_cells = 3
    params = create_uniform_parameters(
        n_cells=n_cells,
        cell_length_km=0.5,
        lanes=3,
        T=0.1,
        Q_max=2000.0,
        v_f=100.0,
        rho_crit=30.0,
        rho_jam=150.0,
        lambda_ramp=1.0,
    )
    
    # Low density that supports free-flow
    # At free-flow: q = v_f * rho * lambda, so for q=3000, rho = 3000/(100*3) = 10
    rho0 = np.array([10.0, 10.0, 10.0])
    N0 = np.array([])
    
    K = 50
    d = np.zeros((K, 0))
    s = np.zeros((K, n_cells))
    # Upstream flow matching free-flow for initial density
    q_upstream = np.full(K, 3000.0)  # = 100 km/h * 10 veh/km/lane * 3 lanes
    
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
    
    # Speeds should be near free-flow
    avg_speed = np.mean(results.v)
    
    print(f"Free-flow test:")
    print(f"  Free-flow speed: {params.v_f:.2f} km/h")
    print(f"  Average observed speed: {avg_speed:.2f} km/h")
    print(f"  Test passed: {avg_speed > 0.85 * params.v_f}")
    print()
    
    return avg_speed > 0.85 * params.v_f


def run_all_tests():
    """Run all test cases."""
    print("=" * 60)
    print("CTM Simulation Test Suite")
    print("=" * 60)
    print()
    
    tests = [
        ("Conservation", test_conservation),
        ("Steady State", test_steady_state),
        ("Ramp Queue", test_ramp_queue),
        ("Capacity Constraint", test_capacity_constraint),
        ("Free Flow", test_free_flow),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
            print()
    
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")
    print()
    
    total = len(results)
    passed_count = sum(1 for _, p in results if p)
    print(f"Total: {passed_count}/{total} tests passed")
    print()
    
    return all(p for _, p in results)


if __name__ == "__main__":
    all_passed = run_all_tests()
    exit(0 if all_passed else 1)
