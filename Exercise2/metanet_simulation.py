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

UNIT CONVENTIONS
----------------
This module uses consistent physical units throughout:

Time:
  - time_step_hours (dt, T): hours
  - tau_s: seconds (converted to hours internally as tau_hours = tau_s / 3600.0)

Space and Lanes:
  - length_km (L_i): kilometres
  - lanes (lambda_i): dimensionless count

Density:
  - density (rho): veh/km/lane (per-lane density)
  - This is stored as per-lane density to simplify fundamental diagram calculations.
  - When lane counts change, density remains in veh/km/lane (not total veh/km).

Speed:
  - speed (v): km/h

Flow:
  - flows (q, inflow, outflow, ramp_flow): veh/h (TOTAL flow across all lanes)
  - Flow is computed as: q = rho * v * lanes (veh/km/lane * km/h * lanes = veh/h)
  - Profiles (upstream_demand, downstream_supply, arrival_rate): veh/h

Queue:
  - ramp queues (N_i, queue_veh): vehicles (count, not rate)
  - Queue update: N(k+1) = N(k) + T * d(k) - T * r(k)
    where d(k) and r(k) are in veh/h, T is in hours, so T*d is vehicles.

Parameters:
  - nu: km²/h (anticipation coefficient)
  - kappa: veh/km/lane (regularization constant)
  - delta: dimensionless (ramp coupling parameter)
  - capacity_veh_per_hour_per_lane: veh/h/lane
  - max_ramp_flow_veh_per_hour: veh/h

CRITICAL NOTES FOR LANE DROPS/INCREASES
----------------------------------------
When the number of lanes changes between cells:

1. Density remains per-lane (veh/km/lane), NOT total density (veh/km).
   This allows fundamental diagrams to remain consistent across lane changes.

2. Flow is computed as TOTAL flow: q = rho * v * lanes (veh/h).
   This accounts for the lane factor when computing actual vehicle movement.

3. Density update uses: d_rho = (T / (L * lanes)) * (inflow - outflow)
   The division by lanes ensures the per-lane density is correctly updated
   from total flows. This is the KEY to vehicle conservation:
   - inflow, outflow are in veh/h (total)
   - T * (inflow - outflow) gives vehicles (count)
   - Dividing by (L * lanes) gives veh/(km * lanes) = veh/km/lane ✓

4. Example: 3 lanes -> 2 lanes (lane drop)
   - Cell 0 has 3 lanes with density rho_0 = 40 veh/km/lane
   - Flow out of cell 0: q_0 = 40 * 80 * 3 = 9600 veh/h
   - This flow enters cell 1 (2 lanes)
   - Density change in cell 1: d_rho_1 = (T/(L*2)) * 9600
   - The factor of 2 (not 3) correctly accounts for cell 1's lane count.

PROFILE EXPECTATIONS
--------------------
All profile functions (upstream_demand, arrival_rate, downstream_supply)
must return flow rates in veh/h, NOT vehicles per time step.

Common error: If a profile returns 100 when you mean 100 vehicles per 
time step (dt=0.1 hours), the code will interpret this as 100 veh/h,
resulting in only 10 vehicles per step—off by a factor of 1/dt.

To diagnose: Use the built-in _diagnose_profile_units() method, which
warns if profile values appear to be in wrong units.

To fix: Either:
  - Multiply profile values by (1/dt) to convert per-step to per-hour, OR
  - Ensure profiles already return veh/h (recommended)
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
    - critical_density_veh_per_km_per_lane: Critical density ρ_cr per lane in veh/km (must be > 0).
    - tau_s: Relaxation time τ in seconds (must be > 0), typically 10-30s.
    - nu: Anticipation/diffusion coefficient ν in km²/h (must be >= 0), typically small.
    - kappa: Regularization constant κ in veh/km/lane (must be > 0), small positive value.
    - delta: Dimensionless parameter for ramp metering coupling (must be > 0), typically 0.0122.
    - capacity_veh_per_hour_per_lane: Per-lane capacity in veh/h (must be > 0).
    - max_ramp_flow_veh_per_hour: Maximum ramp flow Q_max_ramp in veh/h (must be > 0), default 2000.0.
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
    critical_density_veh_per_km_per_lane: float = 33.5  # Critical density (veh/km/lane)
    tau_s: float = 18.0  # Relaxation time in seconds
    nu: float = 60.0  # Anticipation coefficient (km²/h)
    kappa: float = 40.0  # Regularization constant (veh/km/lane)
    delta: float = 0.0122  # Ramp metering coupling parameter
    capacity_veh_per_hour_per_lane: float = 2200.0
    max_ramp_flow_veh_per_hour: float = 2000.0  # Maximum ramp flow (veh/h)
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
            ("critical_density_veh_per_km_per_lane", self.critical_density_veh_per_km_per_lane),
            ("tau_s", self.tau_s),
            ("kappa", self.kappa),
            ("delta", self.delta),
            ("capacity_veh_per_hour_per_lane", self.capacity_veh_per_hour_per_lane),
            ("max_ramp_flow_veh_per_hour", self.max_ramp_flow_veh_per_hour),
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
    mainline_priority : float
        Fraction in [0, 1] giving priority share to mainline when merging.
    initial_queue_veh : float
        Initial queued vehicles on the ramp (veh).
    name : Optional[str]
        Optional human-readable identifier for the ramp.
    queue_veh : float
        Runtime queue size (veh). Initialized from initial_queue_veh.
    """

    target_cell: Union[int, str]
    arrival_rate_profile: Profile
    meter_rate_veh_per_hour: Optional[float] = None
    mainline_priority: float = 0.5
    initial_queue_veh: float = 0.0
    name: Optional[str] = None

    # Runtime state
    queue_veh: float = field(init=False)

    def __post_init__(self) -> None:
        if not 0 <= self.mainline_priority <= 1:
            raise ValueError("mainline_priority must be within [0, 1].")
        if self.meter_rate_veh_per_hour is not None and self.meter_rate_veh_per_hour < 0:
            raise ValueError("meter_rate_veh_per_hour cannot be negative.")
        if self.initial_queue_veh < 0:
            raise ValueError("initial_queue_veh cannot be negative.")
        
        self.queue_veh = float(self.initial_queue_veh)


def exponential_equilibrium_speed(
    density: float,
    free_flow_speed_kmh: float,
    critical_density: float,
) -> float:
    """Compute equilibrium speed using exponential model.

    V_s(ρ) = v_f * exp(-1/2 * (ρ / ρ_cr)^2)

    This is the standard METANET equilibrium speed function.

    Parameters
    ----------
    density : float
        Current density in veh/km/lane.
    free_flow_speed_kmh : float
        Free-flow speed in km/h.
    critical_density : float
        Critical density in veh/km/lane.

    Returns
    -------
    float
        Equilibrium speed in km/h, clamped to [0, v_f].
    """
    if critical_density <= 0:
        return free_flow_speed_kmh
    ratio = density / critical_density
    speed = free_flow_speed_kmh * math.exp(-0.5 * ratio * ratio)
    return max(0.0, min(speed, free_flow_speed_kmh))


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

    This function is kept for backward compatibility.

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


# Alias for backward compatibility and test compatibility
mfd_speed = greenshields_speed


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
            Optional function V(rho, v_f, rho_cr) -> speed_kmh.
            Defaults to exponential equilibrium speed if not provided.

        Raises
        ------
        ValueError
            If time_step_hours is not positive or cells is empty.
        """
        if time_step_hours <= 0:
            raise ValueError("time_step_hours must be positive.")
        if not cells:
            raise ValueError("At least one cell configuration is required.")
        
        # Unit validation: time step should be in hours (typically 0.001 to 0.1)
        if time_step_hours > 1.0:
            import warnings
            warnings.warn(
                f"time_step_hours={time_step_hours} is unusually large. "
                f"Expected range: 0.001 to 0.1 hours. "
                f"Ensure this is in hours, not seconds.",
                UserWarning
            )

        self.cells: List[METANETCellConfig] = list(cells)
        self.dt = float(time_step_hours)
        self.upstream_profile = upstream_demand_profile
        self.downstream_profile = downstream_supply_profile

        # Set equilibrium speed function
        self.equilibrium_speed = equilibrium_speed_func or exponential_equilibrium_speed

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
        
        # Diagnose profile units to detect common errors
        self._diagnose_profile_units(self.upstream_profile, "upstream_demand_profile")
        if self.downstream_profile is not None:
            self._diagnose_profile_units(self.downstream_profile, "downstream_supply_profile")
        for ramp in self.on_ramps:
            self._diagnose_profile_units(
                ramp.arrival_rate_profile, 
                f"on_ramp '{ramp.name}' arrival_rate_profile"
            )

    def _diagnose_profile_units(self, profile: Profile, name: str) -> None:
        """Diagnose whether a profile returns veh/h or veh per time step.
        
        This heuristic check helps detect common unit errors where profiles
        return vehicles per time step instead of the expected veh/h.
        
        Parameters
        ----------
        profile : Profile
            The profile to check (constant, list, or callable).
        name : str
            Human-readable name for error messages.
        
        Warnings
        --------
        UserWarning
            If the profile appears to return veh per time step instead of veh/h.
        """
        import warnings
        
        try:
            sample = _get_profile_value(profile, 0, self.dt)
        except Exception:
            # Can't diagnose, skip
            return
        
        if sample is None or sample <= 0:
            return
        
        # Heuristic: if sample is small (< 10) but sample*dt would be tiny (< 0.1),
        # the profile is likely returning per-step values when it should return veh/h
        arrivals_per_step = sample * self.dt
        
        # Case 1: sample is small and arrivals_per_step is also small -> likely per-step units
        if sample < 10.0 and arrivals_per_step < 0.5:
            warnings.warn(
                f"Profile '{name}' may be in vehicles per time-step "
                f"(sample value={sample:.2f}, dt={self.dt}).\n"
                f"Code expects veh/h. If {sample:.2f} is vehicles per step, "
                f"multiply your profile values by (1/dt) = {1.0/self.dt:.1f}.",
                UserWarning
            )
        
        # Case 2: very large sample (> 100000) might indicate wrong units
        if sample > 100000.0:
            warnings.warn(
                f"Profile '{name}' returns very large value ({sample:.0f}).\n"
                f"Ensure this is in veh/h, not veh/s or other units.",
                UserWarning
            )


    def _resolve_target_index(self, target: Union[int, str]) -> int:
        """Resolve an on-ramp target to an integer index."""
        if isinstance(target, int):
            if not 0 <= target < len(self.cells):
                raise IndexError("On-ramp target cell index out of range.")
            return target
        if target not in self._cell_index:
            raise KeyError(f"Unknown cell name '{target}'.")
        return self._cell_index[target]

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
                    cell.critical_density_veh_per_km_per_lane,
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
        
        # Initialize upstream queue (vehicles waiting to enter the first cell)
        upstream_queue_veh = 0.0
        upstream_queue_history: List[float] = [upstream_queue_veh]

        # Main time-stepping loop
        for step in range(steps):
            # Update on-ramp queues (add arrivals)
            ramp_flows_placeholder = self._update_ramp_queues(step)
            ramp_flows_step: Dict[str, float] = {
                ramp.name: 0.0 for ramp in self.on_ramps
            }

            # Evaluate upstream demand and downstream supply
            upstream_demand_raw = _get_profile_value(
                self.upstream_profile, step, self.dt
            )
            upstream_demand_raw = max(0.0, upstream_demand_raw)  # Ensure non-negative
            
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

            # Add new arrivals to upstream queue (veh/h * dt_hours -> veh)
            upstream_queue_veh += upstream_demand_raw * self.dt

            # Prepare inflow/outflow accumulators and store ramp flows
            inflows = [0.0] * len(self.cells)
            outflows = [0.0] * len(self.cells)
            ramp_flows = [0.0] * len(self.cells)  # Track ramp flow per cell

            # Compute flows into each cell
            for idx in range(len(self.cells)):
                # Demand from upstream (either boundary or previous cell)
                if idx == 0:
                    # For the first cell, mainline demand comes from the upstream queue
                    # Convert queue to an hourly rate
                    queue_potential = upstream_queue_veh / self.dt if self.dt > 0 else 0.0
                    receiving = self._compute_receiving(densities[idx], self.cells[idx])
                    mainline_flow_in = min(queue_potential, receiving)
                else:
                    # Flow from previous cell: q_i = rho_i * v_i * lambda_i
                    # Units: (veh/km/lane) * (km/h) * lanes = veh/h (total flow)
                    # CRITICAL: Must respect both what cell idx-1 can send AND what cell idx can receive
                    
                    # What can cell idx-1 send? Limited by vehicles at START of time step
                    # IMPORTANT: Do NOT include inflows[idx-1] here as that would allow
                    # vehicles to enter and exit cell idx-1 in the same time step,
                    # violating METANET/CTM temporal discretization where flows are
                    # computed from state at time t, then used to update to time t+1.
                    prev_current_vehicles = densities[idx-1] * self.cells[idx-1].length_km * self.cells[idx-1].lanes
                    max_sending = prev_current_vehicles / self.dt
                    
                    # Theoretical flow from previous cell
                    theoretical_flow = densities[idx-1] * speeds[idx-1] * self.cells[idx-1].lanes
                    # Constrain by capacity and availability
                    prev_capacity = self.cells[idx-1].capacity_veh_per_hour_per_lane * self.cells[idx-1].lanes
                    sending = min(theoretical_flow, prev_capacity, max_sending)
                    
                    # What can cell idx receive?
                    receiving = self._compute_receiving(densities[idx], self.cells[idx])
                    
                    # Actual flow is minimum of sending and receiving
                    mainline_flow_in = min(sending, receiving)
                
                ramp_flow = 0.0
                ramp = self._ramps_by_cell.get(idx)
                if ramp is not None:
                    # Compute ramp flow using METANET formula
                    ramp_flow = self._compute_ramp_flow(
                        ramp, 
                        self.cells[idx], 
                        densities[idx],
                        step
                    )
                    
                    # Update ramp queue: N_i(k+1) = N_i(k) + T*d_i(k) - T*r_i(k)
                    # Note: arrivals were already added in _update_ramp_queues
                    # Here we subtract the admitted flow
                    # Units: queue_veh in vehicles, ramp_flow in veh/h, dt in hours
                    #        -> vehicles - (veh/h * hours) = vehicles ✓
                    d_i = _get_profile_value(ramp.arrival_rate_profile, step, self.dt)
                    ramp.queue_veh = max(0.0, ramp.queue_veh - ramp_flow * self.dt)
                    ramp_flows_step[ramp.name] = ramp_flow
                    ramp_flows[idx] = ramp_flow
                    
                    # Receiving capacity was already checked above for mainline
                    # Now check if mainline + ramp exceeds it
                    total_demand = mainline_flow_in + ramp_flow
                    if total_demand > receiving:
                        # Split the available receiving capacity by priority
                        main_flow, actual_ramp_flow = self._merge_flows(
                            mainline_flow_in,
                            ramp_flow,
                            receiving,
                            ramp.mainline_priority,
                        )
                        # If ramp flow was reduced, adjust queue accordingly
                        # Add back the vehicles that couldn't be admitted
                        # Units: (ramp_flow - actual_ramp_flow) in veh/h, dt in hours
                        #        -> veh/h * hours = vehicles ✓
                        if actual_ramp_flow < ramp_flow:
                            ramp.queue_veh += (ramp_flow - actual_ramp_flow) * self.dt
                            ramp_flow = actual_ramp_flow
                            ramp_flows_step[ramp.name] = ramp_flow
                            ramp_flows[idx] = ramp_flow
                    else:
                        main_flow = mainline_flow_in

                else:
                    main_flow = mainline_flow_in

                inflow = main_flow + ramp_flow
                inflows[idx] = inflow

                if idx > 0:
                    outflows[idx - 1] = main_flow
                else:
                    # For the first cell, remove accepted flow from upstream queue
                    upstream_queue_veh = max(0.0, upstream_queue_veh - main_flow * self.dt)

            # Handle flow out of the last cell
            last_idx = len(self.cells) - 1
            # Total flow: density * speed * lanes (veh/km/lane * km/h * lanes = veh/h)
            last_flow = densities[last_idx] * speeds[last_idx] * self.cells[last_idx].lanes
            last_flow = min(last_flow, self.cells[last_idx].capacity_veh_per_hour_per_lane * self.cells[last_idx].lanes)
            
            # CRITICAL FIX: Constrain outflow to vehicles available at START of time step
            # IMPORTANT: Do NOT include inflow_vehicles here as that would allow
            # vehicles to enter and exit the last cell in the same time step,
            # violating METANET/CTM temporal discretization where flows are
            # computed from state at time t, then used to update to time t+1.
            current_vehicles = densities[last_idx] * self.cells[last_idx].length_km * self.cells[last_idx].lanes
            max_available_flow = current_vehicles / self.dt
            last_flow = min(last_flow, max_available_flow)
            
            outflow_last = min(last_flow, downstream_supply)
            outflows[last_idx] = outflow_last

            # Record flows for all cells
            for idx in range(len(self.cells)):
                flow_history[self.cells[idx].name].append(outflows[idx])

            # Save ramp flow and queue histories
            for ramp in self.on_ramps:
                ramp_flow_history[ramp.name].append(ramp_flows_step[ramp.name])
                ramp_queue_history[ramp.name].append(ramp.queue_veh)

            # Save upstream queue history for this step (after the update)
            upstream_queue_history.append(upstream_queue_veh)

            # Update densities and speeds using METANET dynamics
            new_densities, new_speeds = self._update_state(
                densities, speeds, inflows, outflows, ramp_flows
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
            upstream_queue=upstream_queue_history,
            time_step_hours=self.dt,
        )

    def _compute_flows(
        self, densities: Sequence[float], speeds: Sequence[float]
    ) -> List[float]:
        """Compute flows q_i = rho_i * v_i * lanes for each cell.
        
        Units:
        - rho_i: veh/km/lane
        - v_i: km/h
        - lanes: dimensionless
        - flow: (veh/km/lane) * (km/h) * lanes = veh/h (total flow) ✓
        """
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

    def _compute_ramp_flow(
        self, 
        ramp: METANETOnRampConfig, 
        target_cell: METANETCellConfig,
        target_density: float,
        step: int
    ) -> float:
        """Compute ramp flow according to METANET specification.
        
        r_i(k) = min(Q_max_ramp, Q_max_i*(rho_jam-rho_i)/(rho_jam-rho_cr), d_i + N_i/T)
        
        Parameters
        ----------
        ramp : METANETOnRampConfig
            The on-ramp configuration
        target_cell : METANETCellConfig
            The mainline cell receiving the ramp flow
        target_density : float
            Current density at target cell (veh/km/lane)
        step : int
            Current time step
            
        Returns
        -------
        float
            Ramp flow in veh/h
            
        Units
        -----
        - d_i: veh/h (arrival demand from profile)
        - N_i: vehicles (queue)
        - T: hours (time step)
        - N_i/T: veh/h (queue discharge rate)
        - Q_max_ramp, Q_max_i: veh/h (capacities)
        - All terms have units veh/h, return value is veh/h ✓
        """
        # Get arrival demand d_i(k) in veh/h
        d_i = _get_profile_value(ramp.arrival_rate_profile, step, self.dt)
        
        # Maximum ramp flow capacity Q_max_ramp (veh/h)
        Q_max_ramp = target_cell.max_ramp_flow_veh_per_hour
        
        # Supply-limited mainline capacity Q_max_i * (rho_jam - rho_i) / (rho_jam - rho_cr)
        # All density terms cancel out, result is in veh/h
        rho_jam = target_cell.jam_density_veh_per_km_per_lane
        rho_cr = target_cell.critical_density_veh_per_km_per_lane
        Q_max_i = target_cell.capacity_veh_per_hour_per_lane * target_cell.lanes
        
        if rho_jam > rho_cr:
            supply_limited_flow = Q_max_i * (rho_jam - target_density) / (rho_jam - rho_cr)
        else:
            supply_limited_flow = Q_max_i
        supply_limited_flow = max(0.0, supply_limited_flow)
        
        # Queue-limited demand: d_i(k) + N_i(k) / T
        # Units: veh/h + vehicles/hours = veh/h ✓
        queue_limited_demand = d_i + ramp.queue_veh / self.dt
        
        # Take minimum of all three constraints (all in veh/h)
        r_i = min(Q_max_ramp, supply_limited_flow, queue_limited_demand)
        r_i = max(0.0, r_i)
        
        # Apply metering if configured (veh/h)
        if ramp.meter_rate_veh_per_hour is not None:
            r_i = min(r_i, ramp.meter_rate_veh_per_hour)
        
        return r_i

    def _update_ramp_queues(self, step: int) -> Dict[str, float]:
        """Update on-ramp queues by adding arrivals.
        
        Implements: N_i(k+1) = N_i(k) + T * d_i(k)
        (The subtraction of r_i(k) happens later in the main loop)
        
        Units:
        - N_i (queue_veh): vehicles
        - d_i: veh/h (from profile)
        - T (self.dt): hours
        - T * d_i: vehicles ✓
        """
        ramp_flows: Dict[str, float] = {}
        for ramp in self.on_ramps:
            # Get arrival demand d_i(k) in veh/h
            d_i = _get_profile_value(ramp.arrival_rate_profile, step, self.dt)
            
            # Add arrivals to queue: arrivals = (veh/h) * hours = vehicles
            ramp.queue_veh += d_i * self.dt
            
            # Ramp flow will be computed during merge, just store placeholder
            ramp_flows[ramp.name] = 0.0
        return ramp_flows

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
        ramp_flows: Sequence[float],
    ) -> Tuple[List[float], List[float]]:
        """Update densities and speeds using METANET dynamics.

        METANET equations (from specification):
        - Density: rho_i(k+1) = rho_i(k) + T/(L_i * lambda_i) * (q_{i-1}(k) + r_i(k) - q_i(k) - s_i(k))
        - Speed: v_i(k+1) = v_i(k) + T/tau * (V_s(rho_i(k)) - v_i(k))
                          + T/L_i * v_i(k) * (v_{i-1}(k) - v_i(k))
                          - T*nu/(tau*L_i) * (rho_{i+1}(k) - rho_i(k)) / (rho_i(k) + kappa)
                          - T*delta/(L_i*lambda_i) * r_i(k) * v_i(k) / (rho_i(k) + kappa)
        - Flow: q_i(k) = rho_i(k) * v_i(k) * lambda_i
        - Equilibrium speed: V_s(rho) = v_f * exp(-1/2 * (rho/rho_cr)^2)
        
        UNIT SUMMARY:
        -------------
        All inputs and outputs use consistent units:
        - densities: veh/km/lane (per-lane density)
        - speeds: km/h
        - inflows, outflows, ramp_flows: veh/h (total flow across all lanes)
        - T (self.dt): hours
        - L_i: km
        - lambda_i: lanes (dimensionless count)
        
        CRITICAL: The density update divides by (L_i * lambda_i) to convert
        total flow (veh/h) to per-lane density change (veh/km/lane). This
        ensures correct vehicle conservation when lane counts change between cells.
        """
        new_densities = []
        new_speeds = []

        for idx, cell in enumerate(self.cells):
            # Current state
            rho = densities[idx]
            v = speeds[idx]
            L_i = cell.length_km
            lambda_i = cell.lanes
            
            # Get actual flows (already computed in main loop accounting for constraints)
            # Units clarification:
            # - inflow_total, outflow_total: veh/h (total flow across all lanes)
            # - r_i: veh/h (ramp flow)
            # - rho: veh/km/lane (per-lane density)
            # - L_i: km (cell length)
            # - lambda_i: dimensionless (number of lanes)
            inflow_total = inflows[idx]  # veh/h
            outflow_total = outflows[idx]  # veh/h
            r_i = ramp_flows[idx]  # veh/h
            s_i = 0.0  # Off-ramp flow (not specified, assume 0)
            
            # Update density: rho_i(k+1) = rho_i(k) + T/(L_i * lambda_i) * (inflow_total - outflow_total - s_i)
            # Unit derivation:
            #   - Net flow = (inflow_total - outflow_total - s_i) in veh/h
            #   - T/(L_i * lambda_i) in hours/(km * lanes)
            #   - hours/(km * lanes) * veh/h = veh/(km * lanes) = veh/km/lane ✓
            # This converts total flow (veh/h) to per-lane density change (veh/km/lane)
            d_rho = (self.dt / (L_i * lambda_i)) * (inflow_total - outflow_total - s_i)
            new_rho = rho + d_rho
            new_rho = max(0.0, min(new_rho, cell.jam_density_veh_per_km_per_lane))
            
            # Prepare for speed update
            rho_safe = max(rho, 1e-6)  # Avoid division by zero
            tau_hours = cell.tau_s / 3600.0  # Convert tau from seconds to hours
            
            # Equilibrium speed: V_s(rho) = v_f * exp(-1/2 * (rho/rho_cr)^2)
            V_s = self.equilibrium_speed(
                rho, 
                cell.free_flow_speed_kmh, 
                cell.critical_density_veh_per_km_per_lane
            )
            
            # Relaxation term: T/tau * (V_s(rho_i) - v_i)
            # Units: (1/tau_hours) in 1/hours, (V_s - v) in km/h
            #        -> (1/hours) * (km/h) = km/h² (acceleration)
            # When multiplied by T (hours), gives speed change in km/h ✓
            relaxation_term = (1.0 / tau_hours) * (V_s - v)
            
            # Convection term: T/L_i * v_i * (v_{i-1} - v_i)
            # Units: (1/L_i) in 1/km, v_i in km/h, (v_{i-1} - v_i) in km/h
            #        -> (1/km) * (km/h) * (km/h) = km/h² (acceleration)
            # When multiplied by T (hours), gives speed change in km/h ✓
            if idx > 0:
                v_i_minus_1 = speeds[idx - 1]
                convection_term = (v / L_i) * (v_i_minus_1 - v)
            else:
                convection_term = 0.0
            
            # Anticipation/gradient term: -T*nu/(tau*L_i) * (rho_{i+1} - rho_i) / (rho_i + kappa)
            # Units: nu in km²/h, tau_hours in hours, L_i in km
            #        rho in veh/km/lane, kappa in veh/km/lane
            #        -> (km²/h) / (hours * km) * (veh/km/lane) / (veh/km/lane)
            #        -> (km/h) / hours * dimensionless = km/h² (acceleration)
            # When multiplied by T (hours), gives speed change in km/h ✓
            if idx < len(self.cells) - 1:
                rho_i_plus_1 = densities[idx + 1]
                anticipation_term = -(cell.nu / (tau_hours * L_i)) * (
                    (rho_i_plus_1 - rho) / (rho_safe + cell.kappa)
                )
            else:
                anticipation_term = 0.0
            
            # Ramp coupling term: -T*delta/(L_i*lambda_i) * r_i * v_i / (rho_i + kappa)
            # Units: delta dimensionless, L_i in km, lambda_i dimensionless
            #        r_i in veh/h, v_i in km/h, rho in veh/km/lane, kappa in veh/km/lane
            #        -> (1/(km * lanes)) * (veh/h) * (km/h) / (veh/km/lane)
            #        -> (veh*km)/(h * km * lanes) / (veh/km/lane)
            #        -> (veh/h/lanes) * (lane*km/veh) = km/h (per lane)
            #        -> km/h / lanes, but delta accounts for merging dynamics
            # The factor 1/lambda_i ensures proper scaling when lanes change.
            # When multiplied by T (hours), gives speed change in km/h ✓
            if r_i > 0:
                ramp_coupling_term = -(cell.delta / (L_i * lambda_i)) * (
                    r_i * v / (rho_safe + cell.kappa)
                )
            else:
                ramp_coupling_term = 0.0
            
            # Speed update: v_i(k+1) = v_i(k) + T * (all terms)
            dv = self.dt * (relaxation_term + convection_term + anticipation_term + ramp_coupling_term)
            new_v = v + dv
            
            # Bound speed to physical range
            new_v = max(0.0, min(new_v, cell.free_flow_speed_kmh))
            
            # If density is very low, reset speed to equilibrium
            if new_rho < 1.0:
                new_v = self.equilibrium_speed(
                    new_rho, 
                    cell.free_flow_speed_kmh, 
                    cell.critical_density_veh_per_km_per_lane
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
    critical_density_veh_per_km_per_lane: float = 33.5,
    tau_s: float = 18.0,
    nu: float = 60.0,
    kappa: float = 40.0,
    delta: float = 0.0122,
    capacity_veh_per_hour_per_lane: float = 2200.0,
    max_ramp_flow_veh_per_hour: float = 2000.0,
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
    critical_density_veh_per_km_per_lane:
        Critical density (veh/km/lane) for equilibrium speed (default: 33.5).
    tau_s:
        Relaxation time in seconds (default: 18.0).
    nu:
        Anticipation/diffusion coefficient in km²/h (default: 60.0).
    kappa:
        Regularization constant in veh/km/lane (default: 40.0).
    delta:
        Ramp metering coupling parameter (default: 0.0122).
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
                critical_density_veh_per_km_per_lane=critical_density_veh_per_km_per_lane,
                tau_s=tau_s,
                nu=nu,
                kappa=kappa,
                delta=delta,
                capacity_veh_per_hour_per_lane=capacity_veh_per_hour_per_lane,
                max_ramp_flow_veh_per_hour=max_ramp_flow_veh_per_hour,
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
    "exponential_equilibrium_speed",
    "greenshields_speed",
    "mfd_speed",  # Alias for greenshields_speed
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
