#!/usr/bin/env python3
"""Diagnostic script to verify unit consistency in METANET simulation results.

This script provides utilities to:
1. Reconstruct ramp queues from arrivals and admitted flows
2. Detect profile unit mismatches
3. Verify vehicle conservation across lane changes
4. Compare METANET results with expected values

Usage:
    python diagnose_metanet_units.py

Or import functions in a notebook:
    from diagnose_metanet_units import reconstruct_queue, diagnose_profile_units
"""

from typing import List, Optional, Union
import warnings


def reconstruct_queue(
    arrival_rate_profile: Union[float, List[float], callable],
    ramp_flows: List[float],
    initial_queue: float,
    dt: float,
    ramp_name: str = "ramp",
) -> List[float]:
    """Reconstruct ramp queue from arrivals and admitted flows.
    
    This function implements the queue equation:
        N(k+1) = N(k) + T * d(k) - T * r(k)
    
    where:
    - N(k): queue at step k (vehicles)
    - d(k): arrival rate at step k (veh/h)
    - r(k): ramp flow at step k (veh/h)
    - T: time step (hours)
    
    Compare the reconstructed queue to the reported queue to detect unit issues.
    
    Parameters
    ----------
    arrival_rate_profile : Union[float, List[float], callable]
        Arrival demand profile (should be in veh/h)
    ramp_flows : List[float]
        Admitted ramp flows from simulation (should be in veh/h)
    initial_queue : float
        Initial queue size (vehicles)
    dt : float
        Time step in hours
    ramp_name : str
        Name for diagnostic messages
    
    Returns
    -------
    List[float]
        Reconstructed queue at each step (including initial)
    
    Examples
    --------
    >>> # After running simulation:
    >>> reconstructed = reconstruct_queue(
    ...     arrival_rate_profile=600.0,  # veh/h
    ...     ramp_flows=result.ramp_flows["ramp_1"],
    ...     initial_queue=0.0,
    ...     dt=result.time_step_hours,
    ...     ramp_name="ramp_1"
    ... )
    >>> reported = result.ramp_queues["ramp_1"]
    >>> # Compare reconstructed to reported
    >>> max_diff = max(abs(r - p) for r, p in zip(reconstructed, reported))
    >>> print(f"Max difference: {max_diff:.2f} vehicles")
    """
    steps = len(ramp_flows)
    queue = [initial_queue]
    
    for step in range(steps):
        # Get arrival rate for this step
        if callable(arrival_rate_profile):
            arrival_rate = arrival_rate_profile(step)
        elif isinstance(arrival_rate_profile, (list, tuple)):
            arrival_rate = arrival_rate_profile[step] if step < len(arrival_rate_profile) else 0.0
        else:
            arrival_rate = float(arrival_rate_profile)
        
        # Compute arrivals and departures in vehicles (not rates)
        arrivals = arrival_rate * dt  # veh/h * hours = vehicles
        departures = ramp_flows[step] * dt  # veh/h * hours = vehicles
        
        # Update queue
        new_queue = max(0.0, queue[-1] + arrivals - departures)
        queue.append(new_queue)
    
    return queue


def compare_queues(
    reconstructed_queue: List[float],
    reported_queue: List[float],
    tolerance: float = 5.0,
    ramp_name: str = "ramp",
) -> None:
    """Compare reconstructed queue to reported queue and print diagnostics.
    
    Parameters
    ----------
    reconstructed_queue : List[float]
        Queue reconstructed from arrivals and flows
    reported_queue : List[float]
        Queue reported by simulation
    tolerance : float
        Maximum acceptable difference in vehicles (default: 5.0)
    ramp_name : str
        Name for diagnostic messages
    """
    print(f"\n=== Queue Comparison for {ramp_name} ===")
    print(f"Steps: {len(reported_queue)}")
    print(f"Reconstructed: max={max(reconstructed_queue):.1f}, "
          f"final={reconstructed_queue[-1]:.1f}")
    print(f"Reported:      max={max(reported_queue):.1f}, "
          f"final={reported_queue[-1]:.1f}")
    
    # Compute differences
    max_diff = 0.0
    max_diff_step = 0
    avg_diff = 0.0
    
    for step, (recon, report) in enumerate(zip(reconstructed_queue, reported_queue)):
        diff = abs(recon - report)
        avg_diff += diff
        if diff > max_diff:
            max_diff = diff
            max_diff_step = step
    
    avg_diff /= len(reported_queue)
    
    print(f"Max difference: {max_diff:.2f} vehicles at step {max_diff_step}")
    print(f"Avg difference: {avg_diff:.2f} vehicles")
    
    if max_diff > tolerance:
        print(f"\n⚠️  WARNING: Max difference ({max_diff:.2f}) exceeds tolerance ({tolerance:.2f})")
        print("Possible causes:")
        print("1. Profile returns veh per time-step instead of veh/h")
        print("   - Fix: multiply profile values by (1/dt)")
        print("2. ramp_flows stored with different units than expected")
        print("   - Check: Are ramp_flows in veh/h or vehicles per step?")
        print("3. Numerical precision differences")
        print("   - If difference is small (<1), likely benign")
        
        # Heuristic to detect unit mismatch
        ratio = max(reported_queue) / max(reconstructed_queue) if max(reconstructed_queue) > 0 else 0
        if 0.01 < ratio < 0.99 or 1.01 < ratio < 100:
            print(f"\n⚠️  Ratio of max queues: {ratio:.3f}")
            print("This suggests a systematic unit scaling error!")
    else:
        print(f"\n✅ Queue reconstruction matches (within {tolerance:.2f} vehicles)")


def check_vehicle_conservation(
    result,
    cell_names: Optional[List[str]] = None,
    tolerance: float = 10.0,
) -> None:
    """Check vehicle conservation across the network.
    
    This verifies that the total number of vehicles in the system changes
    only by the net inflow/outflow at boundaries.
    
    Parameters
    ----------
    result : SimulationResult
        Simulation result object
    cell_names : Optional[List[str]]
        List of cell names to check (default: all cells)
    tolerance : float
        Maximum acceptable vehicle loss/gain (default: 10.0)
    """
    print("\n=== Vehicle Conservation Check ===")
    
    if cell_names is None:
        cell_names = list(result.densities.keys())
    
    # Note: We can't fully check conservation without cell lengths and lanes
    # This would require access to the simulation object
    print("⚠️  Full conservation check requires cell geometry information.")
    print("Use check_vehicle_conservation_detailed() with simulation object.")


def check_profile_units(
    profile: Union[float, List[float], callable],
    dt: float,
    profile_name: str = "profile",
) -> None:
    """Check if a profile appears to be in correct units (veh/h vs veh per step).
    
    Parameters
    ----------
    profile : Union[float, List[float], callable]
        Profile to check
    dt : float
        Time step in hours
    profile_name : str
        Name for diagnostic messages
    """
    print(f"\n=== Profile Unit Check: {profile_name} ===")
    
    # Get sample value
    if callable(profile):
        try:
            sample = profile(0)
        except Exception as e:
            print(f"❌ Cannot call profile: {e}")
            return
    elif isinstance(profile, (list, tuple)):
        sample = profile[0] if profile else None
    else:
        sample = float(profile)
    
    if sample is None or sample <= 0:
        print("⚠️  Profile returns zero or None, cannot diagnose units")
        return
    
    print(f"Sample value: {sample:.2f}")
    print(f"Time step: {dt} hours")
    
    arrivals_per_step = sample * dt
    print(f"If veh/h: {arrivals_per_step:.2f} vehicles per time step")
    print(f"If veh/step: {sample:.2f} vehicles per time step")
    
    # Heuristics
    if sample < 10.0 and arrivals_per_step < 0.5:
        print("\n⚠️  WARNING: Profile may be in vehicles per time-step!")
        print(f"Sample ({sample:.2f}) is small, and sample*dt ({arrivals_per_step:.2f}) is tiny.")
        print("Expected: Profile should return veh/h (typically 100-10000)")
        print(f"Fix: Multiply profile values by {1.0/dt:.1f}")
    elif sample > 100000.0:
        print("\n⚠️  WARNING: Profile value is very large!")
        print(f"Sample: {sample:.0f}")
        print("Check: Is this in veh/h, or possibly veh/s or other units?")
    elif 10.0 <= sample <= 100000.0:
        print("\n✅ Profile appears to be in veh/h (reasonable magnitude)")
    else:
        print("\n⚠️  Profile value is unusual, verify units manually")


def print_unit_summary() -> None:
    """Print a summary of expected units in METANET simulation."""
    print("""
=== METANET Unit Summary ===

Time:
  - dt (time_step_hours): hours
  - tau_s: seconds (converted to hours internally)

Density:
  - rho: veh/km/lane (PER-LANE density, not total)

Speed:
  - v: km/h

Flow:
  - q, inflow, outflow: veh/h (TOTAL flow across all lanes)
  - Computed as: q = rho * v * lanes

Queue:
  - N (queue_veh): vehicles (count, not rate)
  - Updated as: N(k+1) = N(k) + T*d(k) - T*r(k)

Profiles:
  - upstream_demand, arrival_rate: veh/h (not veh per step!)

Lane Changes:
  - Density stays per-lane (veh/km/lane)
  - Flow uses lane factor (q = rho * v * lanes)
  - Density update: d_rho = T/(L*lanes) * (inflow - outflow)
    This ensures conservation when lanes change!

Common Error:
  Profile returning 100 when you mean 100 vehicles per 6-minute step (dt=0.1h)
  → Code interprets as 100 veh/h → Only 10 vehicles per step!
  → Fix: multiply by 1/dt or ensure profile already returns veh/h
""")


# Example usage
if __name__ == "__main__":
    print("METANET Unit Diagnostic Tool")
    print("=" * 50)
    
    print_unit_summary()
    
    print("\n\nTo use this tool with your simulation results:")
    print("=" * 50)
    print("""
# 1. Run your simulation
result = sim.run(steps=100)

# 2. Check profile units
from diagnose_metanet_units import check_profile_units
check_profile_units(on_ramp.arrival_rate_profile, dt=sim.dt, 
                   profile_name="arrival_rate")

# 3. Reconstruct and compare queues
from diagnose_metanet_units import reconstruct_queue, compare_queues
reconstructed = reconstruct_queue(
    arrival_rate_profile=on_ramp.arrival_rate_profile,
    ramp_flows=result.ramp_flows["ramp_1"],
    initial_queue=on_ramp.initial_queue_veh,
    dt=result.time_step_hours,
    ramp_name="ramp_1"
)
compare_queues(reconstructed, result.ramp_queues["ramp_1"], 
               ramp_name="ramp_1")

# 4. If there's a large discrepancy:
#    - Check that your arrival profile returns veh/h, not veh per step
#    - Verify dt is in hours, not seconds or minutes
#    - Ensure initial_queue_veh is in vehicles, not veh/h
""")
