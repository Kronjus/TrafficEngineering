"""METANET macroscopic traffic flow simulation.

This module provides a METANET implementation that mirrors the functionality
and API of the CTM simulation, supporting:

* Arbitrary numbers of cells with heterogeneous lane counts and parameters.
* A single upstream boundary condition and a downstream sink.
* Optional on-ramps that merge into any interior cell with configurable
  arrival demand, metering rate and merge priority.
* Speed dynamics with relaxation, anticipation, and convection terms.

The implementation focuses on clarity and pedagogical value while providing
a complete METANET simulation framework compatible with the existing CTM API.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

# Reuse the profile types and helper from CTM
from ctm_simulation import (
    Profile,
    _get_profile_value,
    SimulationResult,
)

NumberLike = Union[float, int]


@dataclass(frozen=True)
class METANETCellConfig:
    """Configuration parameters for a single METANET cell.

    Each field documents its physical meaning and expected units/constraints:

    - name: Human-readable cell identifier (string).
    - length_km: Cell length in kilometres (must be > 0).
    - lanes: Number of traffic lanes in the cell (integer, must be > 0).
    - free_flow_speed_kmh: Free-flow speed v_f in km/h (must be > 0).
    - jam_density_veh_per_km_per_lane: Jam density ρ_jam per lane in veh/km (must be > 0).
    - tau_s: Relaxation time τ in seconds (must be > 0), typically 10-30s.
    - nu: Anticipation/diffusion coefficient ν in km²/h (must be >= 0), typically small.
    - kappa: Regularization constant κ in veh/km/lane (must be > 0), small positive value.
    - delta: Dimensionless exponent (must be > 0) for equilibrium speed function, typically 1-4.
    - capacity_veh_per_hour_per_lane: Per-lane capacity in veh/h (must be > 0).
    - initial_density_veh_per_km_per_lane: Initial density per lane in veh/km (>= 0).
    - initial_speed_kmh: Initial speed in km/h (>= 0, <= free_flow_speed_kmh).

    Notes
    -----
    Validation is performed in ``__post_init__`` to catch common configuration
    mistakes early. The dataclass is frozen to make instances immutable.
    
    The parameter `nu` was previously named `eta` in older versions for backward
    compatibility. Both names refer to the same anticipation coefficient.
    """

    name: str
    length_km: float
    lanes: int
    free_flow_speed_kmh: float
    jam_density_veh_per_km_per_lane: float
    tau_s: float = 18.0  # Relaxation time in seconds
    nu: float = 60.0  # Anticipation coefficient (km²/h)
    kappa: float = 40.0  # Regularization constant (veh/km/lane)
    delta: float = 1.0  # Dimensionless exponent for equilibrium speed
    capacity_veh_per_hour_per_lane: float = 2200.0
    initial_density_veh_per_km_per_lane: float = 0.0
    initial_speed_kmh: float = 0.0  # Will be set to V(rho) if 0

    def __post_init__(self) -> None:
        # Ensure lane count is positive
        if self.lanes <= 0:
            raise ValueError("Number of lanes must be positive.")

        # Validate geometric and fundamental parameters
        for attr_name, attr_val in [
            ("length_km", self.length_km),
            ("free_flow_speed_kmh", self.free_flow_speed_kmh),
            ("jam_density_veh_per_km_per_lane", self.jam_density_veh_per_km_per_lane),
            ("tau_s", self.tau_s),
            ("kappa", self.kappa),
            ("delta", self.delta),
            ("capacity_veh_per_hour_per_lane", self.capacity_veh_per_hour_per_lane),
        ]:
            if attr_val <= 0:
                raise ValueError(f"{attr_name} must be positive.")

        # nu and initial values must be non-negative
        if self.nu < 0:
            raise ValueError("nu must be non-negative.")
        if self.initial_density_veh_per_km_per_lane < 0:
            raise ValueError("Initial density cannot be negative.")
        if self.initial_speed_kmh < 0:
            raise ValueError("Initial speed cannot be negative.")
        if self.initial_speed_kmh > self.free_flow_speed_kmh:
            raise ValueError("Initial speed cannot exceed free flow speed.")


@dataclass
class METANETOnRampConfig:
    """Configuration for an on-ramp merging into a mainline cell.

    This dataclass holds both static configuration and runtime state.

    Fields
    ------
    target_cell : Union[int, str]
        Index or name of the mainline cell this ramp connects to.
    arrival_rate_profile : Profile
        Arrival demand profile (veh/h).
    meter_rate_veh_per_hour : Optional[float]
        Optional ramp metering rate (veh/h). If None, the ramp is unmetered.
        If ALINEA is enabled, this serves as the initial metering rate.
    mainline_priority : float
        Fraction in [0, 1] giving priority share to mainline when merging.
    initial_queue_veh : float
        Initial queued vehicles on the ramp (veh).
    name : Optional[str]
        Optional human-readable identifier for the ramp.
    alinea_enabled : bool
        If True, enables ALINEA feedback ramp metering control. Requires
        meter_rate_veh_per_hour as initial rate. Default: False.
    alinea_gain : float
        ALINEA regulator gain K_R (veh/h per veh/km/lane). Higher values lead
        to more aggressive control. Typical range: 40-70. Default: 50.0.
    alinea_target_density : Optional[float]
        Target density (veh/km/lane) for ALINEA control. If None, uses 80% of
        jam density of the target cell. Default: None.
    alinea_measurement_cell : Optional[Union[int, str]]
        Cell index or name to use for density measurement in ALINEA control.
        If None, uses the target_cell (default behavior). Default: None.
    alinea_min_rate : float
        Minimum metering rate (veh/h) for ALINEA control. Default: 240.0.
    alinea_max_rate : float
        Maximum metering rate (veh/h) for ALINEA control. Default: 2400.0.
    alinea_slew_limit : Optional[float]
        Maximum rate of change of metering rate per hour (veh/h²). If None,
        no slew rate limiting is applied. This constrains how quickly the
        metering rate can change between time steps. Default: None.
    alinea_auto_tune : bool
        If True, automatically tunes the ALINEA gain K_R using a simulation-based
        optimizer. Requires a simulator function. Default: False.
    alinea_tuner_params : Optional[Dict]
        Parameters for auto-tuning: K_min, K_max, n_grid, objective, horizon.
        Default: None (uses sensible defaults).
    queue_veh : float
        Runtime queue size (veh). Initialized from initial_queue_veh.
    """

    target_cell: Union[int, str]
    arrival_rate_profile: Profile
    meter_rate_veh_per_hour: Optional[float] = None
    mainline_priority: float = 0.5
    initial_queue_veh: float = 0.0
    name: Optional[str] = None
    alinea_enabled: bool = False
    alinea_gain: float = 50.0
    alinea_target_density: Optional[float] = None
    alinea_measurement_cell: Optional[Union[int, str]] = None
    alinea_min_rate: float = 240.0
    alinea_max_rate: float = 2400.0
    alinea_slew_limit: Optional[float] = None
    alinea_auto_tune: bool = False
    alinea_tuner_params: Optional[Dict] = None

    # Runtime state
    queue_veh: float = field(init=False)

    def __post_init__(self) -> None:
        if not 0 <= self.mainline_priority <= 1:
            raise ValueError("mainline_priority must be within [0, 1].")
        if self.meter_rate_veh_per_hour is not None and self.meter_rate_veh_per_hour < 0:
            raise ValueError("meter_rate_veh_per_hour cannot be negative.")
        if self.initial_queue_veh < 0:
            raise ValueError("initial_queue_veh cannot be negative.")
        
        # ALINEA validation
        if self.alinea_enabled:
            if self.meter_rate_veh_per_hour is None:
                raise ValueError(
                    "ALINEA requires an initial meter_rate_veh_per_hour."
                )
            if self.alinea_gain <= 0:
                raise ValueError("alinea_gain must be positive.")
            if self.alinea_min_rate < 0:
                raise ValueError("alinea_min_rate cannot be negative.")
            if self.alinea_max_rate <= 0:
                raise ValueError("alinea_max_rate must be positive.")
            if self.alinea_min_rate > self.alinea_max_rate:
                raise ValueError(
                    "alinea_min_rate must not exceed alinea_max_rate."
                )
            if self.alinea_target_density is not None and self.alinea_target_density <= 0:
                raise ValueError("alinea_target_density must be positive.")
            if self.alinea_slew_limit is not None and self.alinea_slew_limit <= 0:
                raise ValueError("alinea_slew_limit must be positive.")
        
        self.queue_veh = float(self.initial_queue_veh)


def greenshields_speed(
    density: float,
    free_flow_speed_kmh: float,
    jam_density: float,
    delta: float = 1.0,
) -> float:
    """Compute equilibrium speed using generalized Greenshields model.

    V(ρ) = v_f * (1 - (ρ / ρ_jam)^delta)

    When delta=1, this reduces to the classic linear Greenshields model.
    Higher values of delta create more nonlinear speed-density relationships.

    Parameters
    ----------
    density : float
        Current density in veh/km/lane.
    free_flow_speed_kmh : float
        Free-flow speed in km/h.
    jam_density : float
        Jam density in veh/km/lane.
    delta : float, optional
        Dimensionless exponent controlling nonlinearity (default: 1.0).
        Must be positive. Typical values range from 1 to 4.

    Returns
    -------
    float
        Equilibrium speed in km/h, clamped to [0, v_f].
    """
    if jam_density <= 0:
        return free_flow_speed_kmh
    ratio = min(1.0, max(0.0, density / jam_density))
    return free_flow_speed_kmh * (1.0 - ratio ** delta)


class METANETSimulation:
    """METANET macroscopic traffic flow simulator.

    This class implements a discrete-time METANET model with:
    * heterogeneous cells (length, lanes, model parameters),
    * a single upstream demand boundary and optional downstream supply,
    * optional on-ramps with queueing and metering,
    * speed dynamics with relaxation, convection, and anticipation terms.

    Public methods:
    * __init__: configure the simulation instance.
    * run: execute the time-stepping loop and return a SimulationResult.

    The API mirrors the CTM simulation for easy interchangeability.
    """

    def __init__(
        self,
        cells: Sequence[METANETCellConfig],
        time_step_hours: float,
        upstream_demand_profile: Profile,
        downstream_supply_profile: Optional[Profile] = None,
        on_ramps: Optional[Sequence[METANETOnRampConfig]] = None,
        equilibrium_speed_func: Optional[Callable[[float, float, float], float]] = None,
    ) -> None:
        """Construct a METANETSimulation.

        Parameters
        ----------
        cells:
            Sequence of METANETCellConfig objects describing the mainline.
        time_step_hours:
            Simulation time step in hours (must be positive).
        upstream_demand_profile:
            Profile providing upstream arrival demand (veh/h).
        downstream_supply_profile:
            Optional profile constraining downstream discharge flow.
        on_ramps:
            Optional sequence of METANETOnRampConfig.
        equilibrium_speed_func:
            Optional function V(rho, v_f, rho_jam) -> speed_kmh.
            Defaults to Greenshields' model if not provided.

        Raises
        ------
        ValueError
            If time_step_hours is not positive or cells is empty.
        """
        if time_step_hours <= 0:
            raise ValueError("time_step_hours must be positive.")
        if not cells:
            raise ValueError("At least one cell configuration is required.")

        self.cells: List[METANETCellConfig] = list(cells)
        self.dt = float(time_step_hours)
        self.upstream_profile = upstream_demand_profile
        self.downstream_profile = downstream_supply_profile

        # Set equilibrium speed function
        self.equilibrium_speed = equilibrium_speed_func or greenshields_speed

        # Map cell names to indices
        self._cell_index: Dict[str, int] = {
            cell.name: idx for idx, cell in enumerate(self.cells)
        }

        # Normalize and store on-ramps
        self.on_ramps: List[METANETOnRampConfig] = []
        if on_ramps:
            for ramp in on_ramps:
                target_index = self._resolve_target_index(ramp.target_cell)
                object.__setattr__(ramp, "target_cell", target_index)
                ramp_name = ramp.name or f"ramp_{target_index}"
                object.__setattr__(ramp, "name", ramp_name)
                self.on_ramps.append(ramp)

        # Build lookup from cell index -> ramp config
        self._ramps_by_cell: Dict[int, METANETOnRampConfig] = {}
        for ramp in self.on_ramps:
            target_index = int(ramp.target_cell)
            if target_index in self._ramps_by_cell:
                raise ValueError(
                    "Only one on-ramp per target cell is supported."
                )
            self._ramps_by_cell[target_index] = ramp
            
            # Set default ALINEA target density if not specified
            if ramp.alinea_enabled and ramp.alinea_target_density is None:
                target_cell_config = self.cells[target_index]
                # Use 80% of jam density as default target
                default_target = 0.8 * target_cell_config.jam_density_veh_per_km_per_lane
                object.__setattr__(ramp, "alinea_target_density", default_target)
            
            # Resolve and set ALINEA measurement cell
            if ramp.alinea_enabled:
                if ramp.alinea_measurement_cell is None:
                    # Default to target cell
                    measurement_idx = target_index
                else:
                    # Resolve the measurement cell
                    measurement_idx = self._resolve_target_index(ramp.alinea_measurement_cell)
                object.__setattr__(ramp, "alinea_measurement_cell", measurement_idx)

    def _resolve_target_index(self, target: Union[int, str]) -> int:
        """Resolve an on-ramp target to an integer index."""
        if isinstance(target, int):
            if not 0 <= target < len(self.cells):
                raise IndexError("On-ramp target cell index out of range.")
            return target
        if target not in self._cell_index:
            raise KeyError(f"Unknown cell name '{target}'.")
        return self._cell_index[target]

    def auto_tune_alinea_gain(
        self,
        ramp_index: int,
        simulator_fn: Callable,
        tuner_params: Optional[Dict] = None,
    ) -> float:
        """Automatically tune the ALINEA gain for a specific ramp.
        
        Uses grid search to find the gain value that minimizes a specified
        objective function over a simulation horizon.
        
        Parameters
        ----------
        ramp_index : int
            Index of the ramp to tune (in self.on_ramps list).
        simulator_fn : Callable
            Function that simulates the system with a given gain.
            Signature: simulator_fn(gain: float) -> Dict[str, float]
            The function should return a dict with metrics like 'rmse', 'delay', etc.
        tuner_params : Optional[Dict]
            Tuning parameters:
            - K_min: Minimum gain to test (default: 0.1)
            - K_max: Maximum gain to test (default: 100.0)
            - n_grid: Number of grid points (default: 20)
            - objective: Metric to minimize (default: 'rmse')
            - verbose: Print tuning progress (default: False)
        
        Returns
        -------
        float
            The best gain value found.
        
        Raises
        ------
        ValueError
            If ramp_index is out of range or simulator_fn returns invalid metrics.
        """
        if not 0 <= ramp_index < len(self.on_ramps):
            raise ValueError(f"ramp_index {ramp_index} out of range.")
        
        ramp = self.on_ramps[ramp_index]
        if not ramp.alinea_enabled:
            raise ValueError(
                f"Ramp {ramp.name} does not have ALINEA enabled."
            )
        
        # Parse tuner parameters
        params = tuner_params or {}
        K_min = params.get('K_min', 0.1)
        K_max = params.get('K_max', 100.0)
        n_grid = params.get('n_grid', 20)
        objective = params.get('objective', 'rmse')
        verbose = params.get('verbose', False)
        
        if K_min >= K_max:
            raise ValueError("K_min must be less than K_max.")
        if n_grid < 2:
            raise ValueError("n_grid must be at least 2.")
        
        # Generate candidate gains using simple linspace
        K_candidates = [K_min + i * (K_max - K_min) / (n_grid - 1) for i in range(n_grid)]
        
        best_K = ramp.alinea_gain
        best_score = float('inf')
        
        if verbose:
            print(f"Auto-tuning ALINEA gain for ramp '{ramp.name}'")
            print(f"  Testing {n_grid} values from {K_min:.2f} to {K_max:.2f}")
            print(f"  Objective: {objective}")
        
        for K_cand in K_candidates:
            # Simulate with this candidate gain
            try:
                metrics = simulator_fn(K_cand)
                if not isinstance(metrics, dict):
                    raise ValueError(
                        "simulator_fn must return a dictionary of metrics."
                    )
                
                if objective not in metrics:
                    raise ValueError(
                        f"Objective '{objective}' not in metrics: {list(metrics.keys())}"
                    )
                
                score = metrics[objective]
                
                if verbose:
                    print(f"  K={K_cand:6.2f} -> {objective}={score:.4f}")
                
                if score < best_score:
                    best_score = score
                    best_K = K_cand
                    
            except Exception as e:
                if verbose:
                    print(f"  K={K_cand:6.2f} -> Error: {e}")
                continue
        
        if verbose:
            print(f"  Best K: {best_K:.2f} with {objective}={best_score:.4f}")
        
        # Update the ramp's gain
        object.__setattr__(ramp, "alinea_gain", best_K)
        
        return best_K

    def _apply_alinea_control(self, densities: Sequence[float]) -> None:
        """Apply ALINEA feedback control to update ramp metering rates.
        
        ALINEA algorithm: r(k+1) = r(k) + K_R * (ρ_target - ρ_measured)
        
        The measured density is taken from the configured measurement cell.
        Optional slew rate limiting constrains how quickly the rate can change.
        
        Parameters
        ----------
        densities : Sequence[float]
            Current densities (veh/km/lane) for all mainline cells.
        """
        for ramp in self.on_ramps:
            if not ramp.alinea_enabled:
                continue
                
            # Use configured measurement cell (defaults to target_cell)
            measurement_idx = int(ramp.alinea_measurement_cell)
            measured_density = densities[measurement_idx]
            target_density = ramp.alinea_target_density
            
            # ALINEA control law: adjust rate based on density error
            # K_R is in (veh/h) per (veh/km/lane), so units work out to veh/h
            density_error = target_density - measured_density
            rate_adjustment = ramp.alinea_gain * density_error
            
            # Update metering rate
            current_rate = ramp.meter_rate_veh_per_hour or 0.0
            new_rate = current_rate + rate_adjustment
            
            # Clamp to min/max bounds
            new_rate = max(ramp.alinea_min_rate, min(new_rate, ramp.alinea_max_rate))
            
            # Apply slew rate limiting if configured
            if ramp.alinea_slew_limit is not None:
                # Slew limit is in (veh/h²), dt is in hours
                # Maximum allowed change in this time step = slew_limit * dt (veh/h)
                max_delta = ramp.alinea_slew_limit * self.dt
                delta = new_rate - current_rate
                # Clamp delta to [-max_delta, +max_delta]
                delta = max(min(delta, max_delta), -max_delta)
                new_rate = current_rate + delta
            
            # Update the ramp's metering rate
            object.__setattr__(ramp, "meter_rate_veh_per_hour", new_rate)

    def run(self, steps: int) -> SimulationResult:
        """Run the METANET simulation for the requested number of time steps.

        Parameters
        ----------
        steps
            Number of discrete simulation steps to execute (must be positive).

        Returns
        -------
        SimulationResult
            Container with time series for densities, flows, speeds, and ramp data.
        """
        if steps <= 0:
            raise ValueError("steps must be a positive integer.")

        # Initialize densities and speeds
        densities = [
            cell.initial_density_veh_per_km_per_lane for cell in self.cells
        ]
        speeds = []
        for idx, cell in enumerate(self.cells):
            if cell.initial_speed_kmh > 0:
                speeds.append(cell.initial_speed_kmh)
            else:
                # Initialize speed to equilibrium speed at initial density
                v_eq = self.equilibrium_speed(
                    densities[idx],
                    cell.free_flow_speed_kmh,
                    cell.jam_density_veh_per_km_per_lane,
                    cell.delta,
                )
                speeds.append(max(1.0, v_eq))  # Ensure non-zero initial speed

        # Initialize history containers
        density_history: Dict[str, List[float]] = {
            cell.name: [density] for cell, density in zip(self.cells, densities)
        }
        flow_history: Dict[str, List[float]] = {cell.name: [] for cell in self.cells}
        ramp_queue_history: Dict[str, List[float]] = {
            ramp.name: [ramp.queue_veh] for ramp in self.on_ramps
        }
        ramp_flow_history: Dict[str, List[float]] = {
            ramp.name: [] for ramp in self.on_ramps
        }

        # Main time-stepping loop
        for step in range(steps):
            # Apply ALINEA feedback control if enabled
            self._apply_alinea_control(densities)
            
            # Update on-ramp queues and get potential ramp flows
            ramp_potentials = self._update_ramp_queues(step)
            ramp_flows_step: Dict[str, float] = {
                ramp.name: 0.0 for ramp in self.on_ramps
            }

            # Evaluate upstream demand and downstream supply
            upstream_demand = _get_profile_value(
                self.upstream_profile, step, self.dt
            )
            last_cell = self.cells[-1]
            max_downstream_flow = (
                last_cell.capacity_veh_per_hour_per_lane * last_cell.lanes
            )
            downstream_supply_raw = (
                _get_profile_value(self.downstream_profile, step, self.dt)
                if self.downstream_profile is not None
                else max_downstream_flow
            )
            downstream_supply = max(0.0, min(downstream_supply_raw, max_downstream_flow))

            # Prepare inflow/outflow accumulators
            inflows = [0.0] * len(self.cells)
            outflows = [0.0] * len(self.cells)

            # Compute flows into each cell
            for idx in range(len(self.cells)):
                # Receiving capacity based on available space
                receiving = self._compute_receiving(densities[idx], self.cells[idx])
                
                # Demand from upstream (either boundary or previous cell)
                if idx == 0:
                    mainline_demand = upstream_demand
                else:
                    # Flow from previous cell = min(sending, previous receiving)
                    mainline_demand = densities[idx-1] * speeds[idx-1] * self.cells[idx-1].lanes
                    mainline_demand = min(mainline_demand, self.cells[idx-1].capacity_veh_per_hour_per_lane * self.cells[idx-1].lanes)
                
                ramp_flow = 0.0
                ramp = self._ramps_by_cell.get(idx)
                if ramp is not None:
                    # Merge mainline and ramp
                    potential = ramp_potentials[ramp.name]
                    main_flow, ramp_flow = self._merge_flows(
                        mainline_demand,
                        potential,
                        receiving,
                        ramp.mainline_priority,
                    )
                    # Dequeue ramp vehicles
                    ramp.queue_veh = max(0.0, ramp.queue_veh - ramp_flow * self.dt)
                    ramp_flows_step[ramp.name] = ramp_flow
                else:
                    main_flow = min(mainline_demand, receiving)

                inflow = main_flow + ramp_flow
                inflows[idx] = inflow

                if idx > 0:
                    outflows[idx - 1] = main_flow

            # Handle flow out of the last cell
            last_idx = len(self.cells) - 1
            last_flow = densities[last_idx] * speeds[last_idx] * self.cells[last_idx].lanes
            last_flow = min(last_flow, self.cells[last_idx].capacity_veh_per_hour_per_lane * self.cells[last_idx].lanes)
            outflow_last = min(last_flow, downstream_supply)
            outflows[last_idx] = outflow_last

            # Record flows for all cells
            for idx in range(len(self.cells)):
                flow_history[self.cells[idx].name].append(outflows[idx])

            # Save ramp flow and queue histories
            for ramp in self.on_ramps:
                ramp_flow_history[ramp.name].append(ramp_flows_step[ramp.name])
                ramp_queue_history[ramp.name].append(ramp.queue_veh)

            # Update densities and speeds using METANET dynamics
            new_densities, new_speeds = self._update_state(
                densities, speeds, inflows, outflows
            )
            densities = new_densities
            speeds = new_speeds

            # Record new state
            for idx, cell in enumerate(self.cells):
                density_history[cell.name].append(densities[idx])

        # Return results using the shared SimulationResult container
        return SimulationResult(
            densities=density_history,
            flows=flow_history,
            ramp_queues=ramp_queue_history,
            ramp_flows=ramp_flow_history,
            time_step_hours=self.dt,
        )

    def _compute_flows(
        self, densities: Sequence[float], speeds: Sequence[float]
    ) -> List[float]:
        """Compute flows q_i = rho_i * v_i * lanes for each cell."""
        flows = []
        for idx, (rho, v, cell) in enumerate(zip(densities, speeds, self.cells)):
            flow = rho * v * cell.lanes  # veh/km/lane * km/h * lanes = veh/h
            flows.append(max(0.0, flow))
        return flows

    def _compute_receiving(self, density: float, cell: METANETCellConfig) -> float:
        """Compute receiving capacity based on available space.

        Similar to CTM receiving: capacity based on available headroom.
        """
        total_capacity = cell.capacity_veh_per_hour_per_lane * cell.lanes
        # Use a backward wave speed approximation (conservative)
        congestion_wave_speed_kmh = 20.0  # Typical value
        remaining_density = max(0.0, cell.jam_density_veh_per_km_per_lane - density)
        supply = congestion_wave_speed_kmh * remaining_density * cell.lanes
        return min(supply, total_capacity)

    def _update_ramp_queues(self, step: int) -> Dict[str, float]:
        """Advance on-ramp queues by arrivals and compute ramp potential."""
        potentials: Dict[str, float] = {}
        for ramp in self.on_ramps:
            arrivals = _get_profile_value(ramp.arrival_rate_profile, step, self.dt)
            ramp.queue_veh += arrivals * self.dt
            meter_rate = ramp.meter_rate_veh_per_hour
            potential = ramp.queue_veh / self.dt
            if meter_rate is not None:
                potential = min(potential, meter_rate)
            potentials[ramp.name] = max(0.0, potential)
        return potentials

    def _merge_flows(
        self,
        mainline_demand: float,
        ramp_demand: float,
        supply: float,
        mainline_priority: float,
    ) -> Tuple[float, float]:
        """Merge mainline and ramp demands into available supply.

        Uses the same priority-based merging logic as CTM.
        """
        if supply <= 0:
            return 0.0, 0.0

        mainline_demand = max(0.0, mainline_demand)
        ramp_demand = max(0.0, ramp_demand)

        if mainline_demand + ramp_demand <= supply:
            return mainline_demand, ramp_demand

        mainline_share = mainline_priority
        ramp_share = 1.0 - mainline_share

        main_flow = min(mainline_demand, mainline_share * supply)
        ramp_flow = min(ramp_demand, ramp_share * supply)

        remaining_supply = supply - main_flow - ramp_flow
        remaining_main_demand = max(0.0, mainline_demand - main_flow)
        remaining_ramp_demand = max(0.0, ramp_demand - ramp_flow)

        if remaining_supply > 0:
            total_remaining_demand = remaining_main_demand + remaining_ramp_demand
            if total_remaining_demand > 0:
                additional_main = remaining_supply * (
                    remaining_main_demand / total_remaining_demand
                )
                additional_ramp = remaining_supply - additional_main
                main_flow += min(additional_main, remaining_main_demand)
                ramp_flow += min(additional_ramp, remaining_ramp_demand)

        return main_flow, ramp_flow

    def _update_state(
        self,
        densities: Sequence[float],
        speeds: Sequence[float],
        inflows: Sequence[float],
        outflows: Sequence[float],
    ) -> Tuple[List[float], List[float]]:
        """Update densities and speeds using METANET dynamics.

        METANET equations:
        - Density: rho_i[k+1] = rho_i[k] + (T/L_i) * (q_{i-1}[k] - q_i[k] + r_i[k])
        - Speed: v_i[k+1] = v_i[k] + T * (
            (V(rho_i) - v_i) / tau
            - (v_i / L_i) * (v_i - v_{i-1})
            - (eta / tau) * (rho_{i+1} - rho_i) / (rho_i + kappa)
          )
        """
        new_densities = []
        new_speeds = []

        for idx, cell in enumerate(self.cells):
            # Update density (conservation of vehicles)
            inflow = inflows[idx]
            outflow = outflows[idx]
            change = (self.dt / cell.length_km) * (
                inflow / cell.lanes - outflow / cell.lanes
            )
            new_rho = densities[idx] + change
            new_rho = max(0.0, min(new_rho, cell.jam_density_veh_per_km_per_lane))

            # Update speed (METANET speed dynamics)
            # Use current (not new) density for speed update
            rho = max(densities[idx], 0.1)  # Avoid division issues at zero density
            v = speeds[idx]

            # Equilibrium speed term
            v_eq = self.equilibrium_speed(
                densities[idx], cell.free_flow_speed_kmh, cell.jam_density_veh_per_km_per_lane, cell.delta
            )
            tau_hours = cell.tau_s / 3600.0  # Convert tau from seconds to hours
            relaxation_term = (v_eq - v) / tau_hours

            # Convection term (only if there's upstream flow)
            if idx > 0 and v > 0:
                v_upstream = speeds[idx - 1]
                convection_term = -(v / cell.length_km) * (v - v_upstream)
            else:
                convection_term = 0.0

            # Anticipation term
            if idx < len(self.cells) - 1:
                rho_downstream = densities[idx + 1]
                # nu is in km²/h, convert to proper units
                nu_hours = cell.nu  # Already in km²/h
                anticipation_term = -(nu_hours / tau_hours) * (
                    (rho_downstream - densities[idx]) / (rho + cell.kappa)
                )
            else:
                anticipation_term = 0.0

            # Speed update with bounds checking
            dv = self.dt * (relaxation_term + convection_term + anticipation_term)
            new_v = v + dv
            new_v = max(0.0, min(new_v, cell.free_flow_speed_kmh))
            
            # If density is very low, reset speed to equilibrium
            if new_rho < 1.0:
                new_v = self.equilibrium_speed(
                    new_rho, cell.free_flow_speed_kmh, cell.jam_density_veh_per_km_per_lane, cell.delta
                )

            new_densities.append(new_rho)
            new_speeds.append(new_v)

        return new_densities, new_speeds


def build_uniform_metanet_mainline(
    *,
    num_cells: int,
    cell_length_km: float,
    lanes: Union[int, Sequence[int]],
    free_flow_speed_kmh: float,
    jam_density_veh_per_km_per_lane: float,
    tau_s: float = 18.0,
    nu: float = 60.0,
    kappa: float = 40.0,
    delta: float = 1.0,
    capacity_veh_per_hour_per_lane: float = 2200.0,
    initial_density_veh_per_km_per_lane: Union[float, Sequence[float]] = 0.0,
    initial_speed_kmh: Union[float, Sequence[float]] = 0.0,
) -> List[METANETCellConfig]:
    """Create a list of METANETCellConfig objects representing a uniform mainline.

    This helper constructs num_cells mainline cells using the provided METANET
    parameters. It accepts either scalar or sequence values for lanes, initial
    density, and initial speed.

    Parameters
    ----------
    num_cells:
        Number of mainline cells to create. Must be positive.
    cell_length_km:
        Length of each cell in kilometres.
    lanes:
        Either a single integer or sequence with one integer per cell.
    free_flow_speed_kmh:
        Free flow speed (km/h) applied to all cells.
    jam_density_veh_per_km_per_lane:
        Per-lane jam density (veh/km/lane) applied to all cells.
    tau_s:
        Relaxation time in seconds (default: 18.0).
    nu:
        Anticipation/diffusion coefficient in km²/h (default: 60.0).
    kappa:
        Regularization constant in veh/km/lane (default: 40.0).
    delta:
        Dimensionless exponent for equilibrium speed (default: 1.0).
    capacity_veh_per_hour_per_lane:
        Per-lane capacity (veh/h/lane, default: 2200.0).
    initial_density_veh_per_km_per_lane:
        Initial density per lane (scalar or sequence).
    initial_speed_kmh:
        Initial speed in km/h (scalar or sequence). If 0, uses V(rho).

    Returns
    -------
    List[METANETCellConfig]
        Configured METANET cells.

    Raises
    ------
    ValueError
        If num_cells is not positive or sequence lengths don't match.
    """
    if num_cells <= 0:
        raise ValueError("num_cells must be positive.")

    # Normalize lanes
    if isinstance(lanes, int):
        lane_profile = [lanes] * num_cells
    else:
        lane_profile = list(lanes)
        if len(lane_profile) != num_cells:
            raise ValueError("lanes sequence must match num_cells.")

    # Normalize initial densities
    if isinstance(initial_density_veh_per_km_per_lane, (int, float)):
        density_profile = [float(initial_density_veh_per_km_per_lane)] * num_cells
    else:
        density_profile = [float(v) for v in initial_density_veh_per_km_per_lane]
        if len(density_profile) != num_cells:
            raise ValueError("initial_density sequence must match num_cells.")

    # Normalize initial speeds
    if isinstance(initial_speed_kmh, (int, float)):
        speed_profile = [float(initial_speed_kmh)] * num_cells
    else:
        speed_profile = [float(v) for v in initial_speed_kmh]
        if len(speed_profile) != num_cells:
            raise ValueError("initial_speed sequence must match num_cells.")

    # Construct cells
    cells: List[METANETCellConfig] = []
    for idx in range(num_cells):
        cells.append(
            METANETCellConfig(
                name=f"cell_{idx}",
                length_km=cell_length_km,
                lanes=lane_profile[idx],
                free_flow_speed_kmh=free_flow_speed_kmh,
                jam_density_veh_per_km_per_lane=jam_density_veh_per_km_per_lane,
                tau_s=tau_s,
                nu=nu,
                kappa=kappa,
                delta=delta,
                capacity_veh_per_hour_per_lane=capacity_veh_per_hour_per_lane,
                initial_density_veh_per_km_per_lane=density_profile[idx],
                initial_speed_kmh=speed_profile[idx],
            )
        )
    return cells


def run_basic_metanet_scenario(steps: int = 20) -> SimulationResult:
    """Run a small METANET demonstration scenario.

    This scenario mirrors the CTM basic scenario to allow comparison.
    """

    def triangular_profile(step: int) -> float:
        peak = 1800.0
        if step < 5:
            return peak * (step / 5)
        if step < 10:
            return peak
        return max(0.0, peak - 200.0 * (step - 10))

    cells = build_uniform_metanet_mainline(
        num_cells=3,
        cell_length_km=0.5,
        lanes=[3, 3, 2],
        free_flow_speed_kmh=100.0,
        jam_density_veh_per_km_per_lane=160.0,
        tau_s=18.0,
        nu=60.0,
        kappa=40.0,
        delta=1.0,
        capacity_veh_per_hour_per_lane=2200.0,
        initial_density_veh_per_km_per_lane=[20.0, 25.0, 30.0],
    )

    on_ramp = METANETOnRampConfig(
        target_cell=1,
        arrival_rate_profile=[300.0] * steps,
        meter_rate_veh_per_hour=600.0,
        mainline_priority=0.6,
        initial_queue_veh=10.0,
        name="demo_ramp",
    )

    sim = METANETSimulation(
        cells=cells,
        time_step_hours=0.1,
        upstream_demand_profile=triangular_profile,
        downstream_supply_profile=2000.0,
        on_ramps=[on_ramp],
    )
    return sim.run(steps)


__all__ = [
    "METANETSimulation",
    "METANETCellConfig",
    "METANETOnRampConfig",
    "SimulationResult",
    "build_uniform_metanet_mainline",
    "run_basic_metanet_scenario",
    "greenshields_speed",
]


if __name__ == "__main__":  # pragma: no cover - convenience usage
    result = run_basic_metanet_scenario(steps=20)
    try:
        df = result.to_dataframe()
    except RuntimeError:
        from pprint import pprint

        print("pandas not installed - printing raw dictionaries instead\n")
        pprint(
            {
                "time_hours": result.time_vector(),
                "densities": result.densities,
                "flows": result.flows,
                "ramp_queues": result.ramp_queues,
                "ramp_flows": result.ramp_flows,
            }
        )
    else:
        print(df.head())
