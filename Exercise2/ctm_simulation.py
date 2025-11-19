"""Cell Transmission Model (CTM) simulation with triangular fundamental diagram.

This module implements the Cell Transmission Model (CTM) according to the
triangular fundamental diagram (FD) with first-order density dynamics, on-ramp
flows with queue dynamics, and off-ramp flows.

The implementation follows these core equations:

Triangular FD:
    q = rho * v

First-order density update:
    rho_i(k+1) = rho_i(k) + T/(L_i * lambda_i) * (q_(i-1)(k) + r_i(k) - q_i(k) - s_i(k))

Cell flow (sending flow from cell i):
    q_i(k) = min(
        Q_(i,max),
        Q_(i+1,max) * (rho_jam - rho_(i+1)(k)) / (rho_jam - rho_crit),
        v_f * rho_i(k) * lambda_i
    )

On-ramp flow:
    r_i(k) = min(
        Q_(ramp,max),
        Q_(i,max) * (rho_jam - rho_i(k)) / (rho_jam - rho_crit),
        d_i(k) + N_i(k) / T
    )

Ramp queue dynamics:
    N_i(k+1) = N_i(k) + T * (d_i(k) - r_i(k))

Speed in cell i:
    v_i(k) = q_i(k) / (lambda_i * rho_i(k))

Variables:
    q_i(k): Flow exiting cell i at time step k [veh/h]
    rho_i(k): Density within cell i at time step k [veh/km/lane]
    v_i(k): Speed in cell i at time step k [km/h]
    r_i(k): On-ramp flow entering cell i at time step k [veh/h]
    d_i(k): Demand for on-ramp of cell i at time step k [veh/h]
    N_i(k): Queue for on-ramp of cell i at time step k [veh]
    s_i(k): Off-ramp flow exiting cell i at time step k [veh/h]

Parameters:
    T: Time step length [h]
    L_i: Length of cell i [km]
    lambda_i: Number of lanes at cell i [-]
    Q_max: Maximum capacity of FD [veh/h/lane]
    v_f: Free-flow speed of FD [km/h]
    rho_crit: Critical density of FD [veh/km/lane]
    rho_jam: Jam density of FD [veh/km/lane]
    Q_(i,max) = lambda_i * Q_max: Maximum capacity at cell i [veh/h]
    Q_(ramp,max) = lambda_ramp * Q_max: Maximum capacity at on-ramp [veh/h]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union
import numpy as np


@dataclass
class CTMParameters:
    """Parameters for the CTM simulation.
    
    Attributes:
        T: Time step length [h]
        L: Cell lengths [km], array of length n_cells
        lambda_lanes: Number of lanes per cell [-], array of length n_cells
        lambda_ramp: Number of lanes at on-ramps [-]
        Q_max: Maximum capacity per lane [veh/h/lane]
        v_f: Free-flow speed [km/h]
        rho_crit: Critical density [veh/km/lane]
        rho_jam: Jam density [veh/km/lane]
    """
    T: float
    L: np.ndarray
    lambda_lanes: np.ndarray
    lambda_ramp: float
    Q_max: float
    v_f: float
    rho_crit: float
    rho_jam: float
    
    def __post_init__(self):
        """Validate parameters."""
        if self.T <= 0:
            raise ValueError("Time step T must be positive")
        if np.any(self.L <= 0):
            raise ValueError("Cell lengths L must be positive")
        if np.any(self.lambda_lanes <= 0):
            raise ValueError("Number of lanes must be positive")
        if self.lambda_ramp <= 0:
            raise ValueError("Ramp lanes must be positive")
        if self.Q_max <= 0:
            raise ValueError("Maximum capacity Q_max must be positive")
        if self.v_f <= 0:
            raise ValueError("Free-flow speed v_f must be positive")
        if self.rho_crit <= 0:
            raise ValueError("Critical density rho_crit must be positive")
        if self.rho_jam <= self.rho_crit:
            raise ValueError("Jam density rho_jam must be greater than critical density rho_crit")
        
        # Convert to numpy arrays if not already
        self.L = np.asarray(self.L, dtype=float)
        self.lambda_lanes = np.asarray(self.lambda_lanes, dtype=float)
        
        # Check CFL condition for numerical stability
        # CFL condition: T <= min(L_i) / v_f
        min_cell_length = np.min(self.L)
        max_stable_dt = min_cell_length / self.v_f
        if self.T > max_stable_dt:
            import warnings
            warnings.warn(
                f"Time step T={self.T:.4f} h exceeds CFL stability limit of "
                f"{max_stable_dt:.4f} h (= min(L)/{self.v_f}). "
                f"This may cause numerical oscillations. Consider reducing T.",
                UserWarning
            )


@dataclass
class CTMState:
    """State variables for the CTM simulation.
    
    Attributes:
        rho: Densities [veh/km/lane], shape (n_cells,)
        N: Ramp queues [veh], shape (n_ramps,)
        q: Cell flows [veh/h], shape (n_cells,)
        r: On-ramp flows [veh/h], shape (n_ramps,)
        v: Speeds [km/h], shape (n_cells,)
    """
    rho: np.ndarray
    N: np.ndarray
    q: np.ndarray
    r: np.ndarray
    v: np.ndarray


@dataclass
class CTMResults:
    """Results from CTM simulation.
    
    Attributes:
        rho: Densities over time [veh/km/lane], shape (K+1, n_cells)
        N: Ramp queues over time [veh], shape (K+1, n_ramps)
        q: Cell flows over time [veh/h], shape (K, n_cells)
        r: On-ramp flows over time [veh/h], shape (K, n_ramps)
        v: Speeds over time [km/h], shape (K, n_cells)
        s: Off-ramp flows over time [veh/h], shape (K, n_cells)
        time: Time vector [h], shape (K+1,)
    """
    rho: np.ndarray
    N: np.ndarray
    q: np.ndarray
    r: np.ndarray
    v: np.ndarray
    s: np.ndarray
    time: np.ndarray


def simulate_ctm(
    rho0: Union[np.ndarray, List[float], float],
    N0: Union[np.ndarray, List[float], float],
    d: Union[np.ndarray, List[List[float]], float],
    s: Union[np.ndarray, List[List[float]], float],
    params: CTMParameters,
    K: int,
    q_upstream: Optional[Union[np.ndarray, List[float], float]] = None,
    ramp_to_cell: Optional[Union[np.ndarray, List[int]]] = None,
) -> CTMResults:
    """Simulate the Cell Transmission Model with triangular fundamental diagram.
    
    This function implements the CTM according to the provided equations with
    triangular FD, first-order density updates, on-ramp flows with queue dynamics,
    and off-ramp flows.
    
    Parameters
    ----------
    rho0 : array_like
        Initial densities per cell [veh/km/lane], shape (n_cells,)
    N0 : array_like
        Initial ramp queues [veh], shape (n_ramps,)
    d : array_like
        On-ramp demand time series [veh/h], shape (K, n_ramps) or scalar
    s : array_like
        Off-ramp flows time series [veh/h], shape (K, n_cells) or scalar
    params : CTMParameters
        CTM parameters (T, L, lambda_lanes, lambda_ramp, Q_max, v_f, rho_crit, rho_jam)
    K : int
        Number of time steps
    q_upstream : array_like, optional
        Upstream inflow boundary condition [veh/h], shape (K,) or scalar.
        If None, uses free-flow capacity of first cell.
    ramp_to_cell : array_like, optional
        Mapping of ramps to cells, shape (n_ramps,). ramp_to_cell[i] gives the
        cell index where ramp i merges. If None, assumes one ramp per cell in order.
    
    Returns
    -------
    CTMResults
        Simulation results with time series of rho, N, q, r, v, s, and time
    
    Notes
    -----
    The simulation implements the following steps at each time k:
    1. Compute on-ramp flows r_i(k) using current rho_i(k), N_i(k), d_i(k)
    2. Compute cell flows q_i(k) considering downstream receiving capacity
    3. Compute speeds v_i(k) from flows and densities
    4. Update densities rho_i(k+1) using conservation equation
    5. Update ramp queues N_i(k+1)
    6. Enforce physical constraints (non-negative, bounded densities)
    
    Edge cases handled:
    - Division by zero in speed calculation (use v_f when rho near zero)
    - Division by zero in capacity fractions (requires rho_jam > rho_crit)
    - Negative flows/densities/queues (clamped to zero)
    - Densities exceeding jam density (clamped to rho_jam)
    """
    n_cells = len(params.L)
    
    # Convert inputs to numpy arrays and validate shapes
    rho0 = np.atleast_1d(rho0).astype(float)
    N0 = np.atleast_1d(N0).astype(float)
    
    if rho0.shape[0] != n_cells:
        if rho0.shape[0] == 1:
            rho0 = np.full(n_cells, rho0[0])
        else:
            raise ValueError(f"rho0 shape {rho0.shape} does not match n_cells={n_cells}")
    
    n_ramps = len(N0)
    
    # Handle demand d
    if np.isscalar(d) or (isinstance(d, (list, np.ndarray)) and np.asarray(d).ndim == 0):
        d_arr = np.full((K, n_ramps), float(d))
    else:
        d_arr = np.asarray(d, dtype=float)
        if d_arr.ndim == 1:
            d_arr = np.tile(d_arr[:, np.newaxis], (1, n_ramps))
        if d_arr.shape != (K, n_ramps):
            raise ValueError(f"Demand d shape {d_arr.shape} does not match (K={K}, n_ramps={n_ramps})")
    
    # Handle off-ramp flows s
    if np.isscalar(s) or (isinstance(s, (list, np.ndarray)) and np.asarray(s).ndim == 0):
        s_arr = np.full((K, n_cells), float(s))
    else:
        s_arr = np.asarray(s, dtype=float)
        if s_arr.ndim == 1:
            s_arr = np.tile(s_arr[:, np.newaxis], (1, n_cells))
        if s_arr.shape != (K, n_cells):
            raise ValueError(f"Off-ramp flows s shape {s_arr.shape} does not match (K={K}, n_cells={n_cells})")
    
    # Handle upstream inflow
    if q_upstream is None:
        # Default: free-flow capacity of first cell
        q_up = np.full(K, params.v_f * params.rho_crit * params.lambda_lanes[0])
    elif np.isscalar(q_upstream) or (isinstance(q_upstream, (list, np.ndarray)) and np.asarray(q_upstream).ndim == 0):
        q_up = np.full(K, float(q_upstream))
    else:
        q_up = np.asarray(q_upstream, dtype=float)
        if q_up.shape[0] != K:
            raise ValueError(f"q_upstream length {len(q_up)} does not match K={K}")
    
    # Handle ramp to cell mapping
    if ramp_to_cell is None:
        # Default: one ramp per cell in order
        if n_ramps > n_cells:
            raise ValueError(f"n_ramps={n_ramps} exceeds n_cells={n_cells} with default mapping")
        ramp_map = np.arange(n_ramps, dtype=int)
    else:
        ramp_map = np.asarray(ramp_to_cell, dtype=int)
        if ramp_map.shape[0] != n_ramps:
            raise ValueError(f"ramp_to_cell length {len(ramp_map)} does not match n_ramps={n_ramps}")
        if np.any(ramp_map < 0) or np.any(ramp_map >= n_cells):
            raise ValueError(f"ramp_to_cell indices must be in [0, {n_cells-1}]")
    
    # Initialize storage arrays
    rho_history = np.zeros((K + 1, n_cells))
    N_history = np.zeros((K + 1, n_ramps))
    q_history = np.zeros((K, n_cells))
    r_history = np.zeros((K, n_ramps))
    v_history = np.zeros((K, n_cells))
    
    # Set initial conditions
    rho_history[0] = np.clip(rho0, 0, params.rho_jam)
    N_history[0] = np.maximum(N0, 0)
    
    # Compute derived parameters
    Q_i_max = params.lambda_lanes * params.Q_max  # Max capacity per cell [veh/h]
    Q_ramp_max = params.lambda_ramp * params.Q_max  # Max ramp capacity [veh/h]
    
    # Small epsilon to avoid division by zero
    eps = 1e-9
    
    # Main simulation loop
    for k in range(K):
        rho_k = rho_history[k]
        N_k = N_history[k]
        d_k = d_arr[k]
        s_k = s_arr[k]
        
        # Step 1: Compute cell flows q_i(k) - needs to be computed before ramp flows
        # since they share receiving capacity
        q_k = np.zeros(n_cells)
        for i in range(n_cells):
            # Sending capacity: free-flow term
            sending = params.v_f * rho_k[i] * params.lambda_lanes[i]
            
            # Receiving capacity of downstream cell
            if i < n_cells - 1:
                # Internal cell: downstream is cell i+1
                receiving = Q_i_max[i + 1] * (params.rho_jam - rho_k[i + 1]) / (params.rho_jam - params.rho_crit)
            else:
                # Last cell: use its own capacity as downstream receiving
                # (free outflow boundary condition)
                receiving = Q_i_max[i]
            
            # q_i(k) = min of capacity, receiving, and sending
            q_k[i] = min(Q_i_max[i], receiving, sending)
            q_k[i] = max(0, q_k[i])  # Ensure non-negative
        
        # Step 2: Compute on-ramp flows r_i(k)
        # The ramp flow must respect the receiving capacity of the target cell
        # AFTER accounting for the mainline flow into that cell
        r_k = np.zeros(n_ramps)
        for i in range(n_ramps):
            cell_idx = ramp_map[i]
            # Total receiving capacity of target cell
            total_receiving = Q_i_max[cell_idx] * (params.rho_jam - rho_k[cell_idx]) / (params.rho_jam - params.rho_crit)
            # Mainline flow entering this cell
            if cell_idx == 0:
                mainline_in = q_up[k]
            else:
                mainline_in = q_k[cell_idx - 1]
            # Available receiving capacity for ramp after mainline
            available_receiving = max(0, total_receiving - mainline_in)
            # Available demand from queue and new arrivals
            available_demand = d_k[i] + N_k[i] / params.T
            # r_i(k) = min of three terms
            r_k[i] = min(Q_ramp_max, available_receiving, available_demand)
            r_k[i] = max(0, r_k[i])  # Ensure non-negative
        
        # Step 3: Compute speeds v_i(k) = q_i(k) / (lambda_i * rho_i(k))
        v_k = np.zeros(n_cells)
        for i in range(n_cells):
            if rho_k[i] > eps:
                v_k[i] = q_k[i] / (params.lambda_lanes[i] * rho_k[i])
            else:
                # Near-zero density: use free-flow speed
                v_k[i] = params.v_f
            # Clamp speed to reasonable range
            v_k[i] = np.clip(v_k[i], 0, params.v_f)
        
        # Step 4: Update densities rho_i(k+1)
        rho_next = rho_k.copy()
        for i in range(n_cells):
            # Inflow to cell i
            if i == 0:
                q_in = q_up[k]
            else:
                q_in = q_k[i - 1]
            
            # Add ramp inflow if applicable
            r_in = 0
            for j in range(n_ramps):
                if ramp_map[j] == i:
                    r_in += r_k[j]
            
            # Conservation equation
            delta_rho = (params.T / (params.L[i] * params.lambda_lanes[i])) * (
                q_in + r_in - q_k[i] - s_k[i]
            )
            rho_next[i] = rho_k[i] + delta_rho
            
            # Enforce bounds
            rho_next[i] = np.clip(rho_next[i], 0, params.rho_jam)
        
        # Step 5: Update ramp queues N_i(k+1)
        N_next = N_k + params.T * (d_k - r_k)
        N_next = np.maximum(N_next, 0)  # Ensure non-negative
        
        # Store results
        q_history[k] = q_k
        r_history[k] = r_k
        v_history[k] = v_k
        rho_history[k + 1] = rho_next
        N_history[k + 1] = N_next
    
    # Create time vector
    time = np.arange(K + 1) * params.T
    
    return CTMResults(
        rho=rho_history,
        N=N_history,
        q=q_history,
        r=r_history,
        v=v_history,
        s=s_arr,
        time=time,
    )


def create_uniform_parameters(
    n_cells: int,
    cell_length_km: float = 0.5,
    lanes: Union[int, List[int]] = 3,
    T: float = 0.1,
    Q_max: float = 2000.0,
    v_f: float = 100.0,
    rho_crit: float = 30.0,
    rho_jam: float = 150.0,
    lambda_ramp: float = 1.0,
) -> CTMParameters:
    """Create uniform CTM parameters for a simple freeway segment.
    
    Parameters
    ----------
    n_cells : int
        Number of cells
    cell_length_km : float
        Length of each cell [km]
    lanes : int or list of int
        Number of lanes per cell. If int, all cells have same lanes.
    T : float
        Time step [h]
    Q_max : float
        Maximum capacity per lane [veh/h/lane]
    v_f : float
        Free-flow speed [km/h]
    rho_crit : float
        Critical density [veh/km/lane]
    rho_jam : float
        Jam density [veh/km/lane]
    lambda_ramp : float
        Number of lanes at on-ramps
    
    Returns
    -------
    CTMParameters
        Configured CTM parameters
    """
    L = np.full(n_cells, cell_length_km)
    
    if isinstance(lanes, int):
        lambda_lanes = np.full(n_cells, lanes)
    else:
        lambda_lanes = np.array(lanes)
        if len(lambda_lanes) != n_cells:
            raise ValueError(f"lanes list length {len(lambda_lanes)} does not match n_cells={n_cells}")
    
    return CTMParameters(
        T=T,
        L=L,
        lambda_lanes=lambda_lanes,
        lambda_ramp=lambda_ramp,
        Q_max=Q_max,
        v_f=v_f,
        rho_crit=rho_crit,
        rho_jam=rho_jam,
    )


def run_simple_example():
    """Run a simple CTM example scenario.
    
    This example demonstrates:
    - 5-cell freeway with uniform parameters
    - Single on-ramp at cell 2 with constant demand
    - Constant upstream demand
    - No off-ramps
    
    Note: The time step is chosen to satisfy the CFL stability condition:
    T <= min(L_i) / v_f for numerical stability.
    """
    # Setup
    n_cells = 5
    n_ramps = 1
    K = 100  # More steps needed with smaller time step
    
    # Create parameters
    # CFL condition: T <= L/v_f = 0.5/100 = 0.005 h
    params = create_uniform_parameters(
        n_cells=n_cells,
        cell_length_km=0.5,
        lanes=3,
        T=0.004,  # Satisfies CFL condition
        Q_max=2000.0,
        v_f=100.0,
        rho_crit=30.0,
        rho_jam=150.0,
        lambda_ramp=1.0,
    )
    
    # Initial conditions
    rho0 = np.array([20.0, 25.0, 30.0, 25.0, 20.0])  # Initial densities
    N0 = np.array([5.0])  # Initial queue at ramp
    
    # Demand and flows
    d = np.full((K, n_ramps), 600.0)  # Constant ramp demand
    s = np.zeros((K, n_cells))  # No off-ramps
    q_upstream = np.full(K, 5000.0)  # Constant upstream demand
    
    # Ramp merges at cell 2
    ramp_to_cell = np.array([2])
    
    # Run simulation
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
    
    return results


if __name__ == "__main__":
    # Run example
    print("Running simple CTM example...")
    results = run_simple_example()
    
    print(f"\nSimulation completed:")
    print(f"  Time steps: {len(results.time) - 1}")
    print(f"  Final time: {results.time[-1]:.2f} hours")
    print(f"  Cells: {results.rho.shape[1]}")
    print(f"  Ramps: {results.N.shape[1]}")
    
    print(f"\nFinal densities [veh/km/lane]:")
    for i, rho in enumerate(results.rho[-1]):
        print(f"  Cell {i}: {rho:.2f}")
    
    print(f"\nFinal ramp queues [veh]:")
    for i, N in enumerate(results.N[-1]):
        print(f"  Ramp {i}: {N:.2f}")
    
    print(f"\nAverage speeds [km/h]:")
    for i in range(results.v.shape[1]):
        avg_v = np.mean(results.v[:, i])
        print(f"  Cell {i}: {avg_v:.2f}")
