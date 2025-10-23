"""Grid-search tuning for P-ALINEA controller gains.

This module provides functions to automatically tune the proportional (Kp) and
integral (Ki) gains for the P-ALINEA ramp metering controller by minimizing
an objective function J = VHT + sum_k(r(k) - r(k-1))^2.

The tuning procedure:
1. Defines grid of (Kp, Ki) pairs to test
2. Runs simulation for each pair
3. Computes J = VHT + ramp_rate_penalty
4. Returns the (Kp, Ki) pair that minimizes J

Example usage:
    from alinea_tuning import grid_search_tune_alinea
    from ctm_simulation import build_uniform_mainline, CTMSimulation, OnRampConfig
    
    # Define scenario
    cells = build_uniform_mainline(...)
    base_ramp = OnRampConfig(...)
    
    # Tune gains
    best_params = grid_search_tune_alinea(
        cells=cells,
        ramp_config=base_ramp,
        simulation_class=CTMSimulation,
        upstream_demand_profile=5000.0,
        time_step_hours=0.05,
        steps=50,
        kp_range=(0.0, 1.0, 0.1),  # (min, max, step)
        ki_range=(10.0, 100.0, 10.0),  # (min, max, step)
    )
    
    print(f"Best Kp: {best_params['Kp']}, Ki: {best_params['Ki']}")
    print(f"Objective J: {best_params['J']}")
"""

from typing import Dict, List, Tuple, Callable, Optional, Any, Sequence
import sys


def _create_kp_ki_grid(
    kp_range: Tuple[float, float, float],
    ki_range: Tuple[float, float, float],
) -> List[Tuple[float, float]]:
    """Create a grid of (Kp, Ki) pairs to test.
    
    Parameters
    ----------
    kp_range : Tuple[float, float, float]
        (min, max, step) for Kp values.
    ki_range : Tuple[float, float, float]
        (min, max, step) for Ki values.
    
    Returns
    -------
    List[Tuple[float, float]]
        List of (Kp, Ki) pairs to test.
    """
    kp_min, kp_max, kp_step = kp_range
    ki_min, ki_max, ki_step = ki_range
    
    # Generate grid points
    kp_values = []
    kp = kp_min
    while kp <= kp_max + 1e-9:  # Small epsilon for floating point comparison
        kp_values.append(kp)
        kp += kp_step
    
    ki_values = []
    ki = ki_min
    while ki <= ki_max + 1e-9:
        ki_values.append(ki)
        ki += ki_step
    
    # Create all combinations
    pairs = [(kp, ki) for kp in kp_values for ki in ki_values]
    return pairs


def grid_search_tune_alinea(
    cells: Sequence[Any],
    ramp_config: Any,
    simulation_class: type,
    upstream_demand_profile: Any,
    time_step_hours: float,
    steps: int,
    kp_range: Tuple[float, float, float] = (0.0, 1.0, 0.1),
    ki_range: Tuple[float, float, float] = (10.0, 100.0, 10.0),
    cell_lengths_km: Optional[Dict[str, float]] = None,
    vht_weight: float = 1.0,
    ramp_penalty_weight: float = 1.0,
    downstream_supply_profile: Optional[Any] = None,
    verbose: bool = True,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Tune P-ALINEA gains (Kp, Ki) using grid search.
    
    Searches over a grid of (Kp, Ki) pairs and selects the one that minimizes:
        J = vht_weight * VHT + ramp_penalty_weight * sum_k(r(k) - r(k-1))^2
    
    Parameters
    ----------
    cells : Sequence
        Cell configuration for the simulation.
    ramp_config : OnRampConfig or METANETOnRampConfig
        Base ramp configuration. The gains will be overridden during tuning.
    simulation_class : type
        Simulation class (CTMSimulation or METANETSimulation).
    upstream_demand_profile : Profile
        Upstream demand profile.
    time_step_hours : float
        Simulation time step in hours.
    steps : int
        Number of simulation steps to run.
    kp_range : Tuple[float, float, float], optional
        (min, max, step) for Kp values. Default: (0.0, 1.0, 0.1).
    ki_range : Tuple[float, float, float], optional
        (min, max, step) for Ki values. Default: (10.0, 100.0, 10.0).
    cell_lengths_km : Dict[str, float], optional
        Mapping from cell name to length in km for VHT calculation.
        If None, VHT is computed per lane without length scaling.
    vht_weight : float, optional
        Weight for VHT term in objective. Default: 1.0.
    ramp_penalty_weight : float, optional
        Weight for ramp rate penalty term. Default: 1.0.
    downstream_supply_profile : Profile, optional
        Downstream supply profile. Default: None.
    verbose : bool, optional
        If True, print progress during tuning. Default: True.
    seed : int, optional
        Random seed for reproducibility. Default: None.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:
        - 'Kp': Best proportional gain
        - 'Ki': Best integral gain
        - 'J': Objective value (weighted sum)
        - 'VHT': Vehicle hours traveled
        - 'ramp_penalty': Ramp rate penalty
        - 'all_results': List of all tested configurations
    """
    # Set random seed if provided
    if seed is not None:
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
    
    # Generate grid of (Kp, Ki) pairs
    pairs = _create_kp_ki_grid(kp_range, ki_range)
    
    if verbose:
        print(f"Grid search tuning P-ALINEA controller")
        print(f"  Testing {len(pairs)} (Kp, Ki) combinations")
        print(f"  Kp range: {kp_range[0]:.3f} to {kp_range[1]:.3f} (step {kp_range[2]:.3f})")
        print(f"  Ki range: {ki_range[0]:.3f} to {ki_range[1]:.3f} (step {ki_range[2]:.3f})")
        print(f"  Steps per simulation: {steps}")
        print(f"  Objective: J = {vht_weight}*VHT + {ramp_penalty_weight}*sum(dr^2)")
        print()
    
    # If cell_lengths_km not provided, extract from cells
    if cell_lengths_km is None:
        cell_lengths_km = {}
        for cell in cells:
            # Compute length as a placeholder (assumes cell has length or uses 1.0)
            cell_length = getattr(cell, 'length_km', 1.0)
            cell_lengths_km[cell.name] = cell_length
    
    best_result = None
    all_results = []
    
    for idx, (kp, ki) in enumerate(pairs):
        if verbose:
            print(f"Testing [{idx+1}/{len(pairs)}]: Kp={kp:.3f}, Ki={ki:.3f}...", end=" ")
            sys.stdout.flush()
        
        try:
            # Create a copy of the ramp config with updated gains
            # We need to use object.__setattr__ since these are frozen dataclasses
            import copy
            test_ramp = copy.deepcopy(ramp_config)
            object.__setattr__(test_ramp, "alinea_gain", ki)
            object.__setattr__(test_ramp, "alinea_proportional_gain", kp)
            object.__setattr__(test_ramp, "alinea_enabled", True)
            # Reset runtime state
            object.__setattr__(test_ramp, "queue_veh", test_ramp.initial_queue_veh)
            object.__setattr__(test_ramp, "alinea_previous_error", 0.0)
            
            # Create simulation
            sim = simulation_class(
                cells=cells,
                time_step_hours=time_step_hours,
                upstream_demand_profile=upstream_demand_profile,
                downstream_supply_profile=downstream_supply_profile,
                on_ramps=[test_ramp],
            )
            
            # Run simulation
            result = sim.run(steps=steps)
            
            # Compute VHT
            vht = result.compute_vht(cell_lengths_km)
            
            # Compute ramp rate penalty
            ramp_penalty = result.compute_ramp_rate_penalty()
            
            # Compute objective
            J = vht_weight * vht + ramp_penalty_weight * ramp_penalty
            
            if verbose:
                print(f"J={J:.2f} (VHT={vht:.2f}, penalty={ramp_penalty:.2f})")
            
            # Record result
            result_entry = {
                'Kp': kp,
                'Ki': ki,
                'J': J,
                'VHT': vht,
                'ramp_penalty': ramp_penalty,
            }
            all_results.append(result_entry)
            
            # Update best if this is better
            if best_result is None or J < best_result['J']:
                best_result = result_entry.copy()
        
        except Exception as e:
            if verbose:
                print(f"FAILED: {e}")
            # Record failure
            all_results.append({
                'Kp': kp,
                'Ki': ki,
                'J': float('inf'),
                'VHT': float('nan'),
                'ramp_penalty': float('nan'),
                'error': str(e),
            })
    
    if best_result is None:
        raise RuntimeError("All simulation configurations failed.")
    
    # Add all results to best result dict
    best_result['all_results'] = all_results
    
    if verbose:
        print()
        print(f"Tuning complete!")
        print(f"  Best (Kp, Ki): ({best_result['Kp']:.3f}, {best_result['Ki']:.3f})")
        print(f"  Best J: {best_result['J']:.2f}")
        print(f"  VHT: {best_result['VHT']:.2f}")
        print(f"  Ramp penalty: {best_result['ramp_penalty']:.2f}")
        print()
    
    return best_result


def save_tuning_results_csv(results: Dict[str, Any], filename: str) -> None:
    """Save tuning results to a CSV file.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Results from grid_search_tune_alinea.
    filename : str
        Output CSV filename.
    """
    import csv
    
    all_results = results.get('all_results', [])
    if not all_results:
        raise ValueError("No results to save.")
    
    # Get field names from first result
    fieldnames = ['Kp', 'Ki', 'J', 'VHT', 'ramp_penalty']
    if 'error' in all_results[0]:
        fieldnames.append('error')
    
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in all_results:
            # Only write fields that exist
            row = {k: entry.get(k, '') for k in fieldnames}
            writer.writerow(row)
    
    print(f"Saved tuning results to {filename}")


if __name__ == "__main__":
    # Simple test/demo
    print("P-ALINEA Grid Search Tuning Module")
    print("Import this module and use grid_search_tune_alinea() to tune controller gains.")
    print()
    print("Example:")
    print("  from alinea_tuning import grid_search_tune_alinea")
    print("  results = grid_search_tune_alinea(...)")
    print("  print(f\"Best Kp: {results['Kp']}, Ki: {results['Ki']}\")")
