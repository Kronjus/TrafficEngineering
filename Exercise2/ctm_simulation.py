"""Simple Cell Transmission Model (CTM) simulation.

This module provides a lightweight CTM implementation that supports:

* Arbitrary numbers of cells with heterogeneous lane counts and parameters.
* A single upstream boundary condition and a downstream sink.
* Optional on-ramps that merge into any interior cell (including the first
  mainline cell) with configurable arrival demand, metering rate and merge
  priority.

The implementation focuses on clarity over micro-optimisation and aims to
offer a pedagogical reference for CTM simulations.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

NumberLike = Union[float, int]
Profile = Union[
    NumberLike,
    Sequence[NumberLike],
    Callable[..., NumberLike],
]

_SECONDS_PARAM_NAMES = {"t_s", "time_s", "time_seconds", "seconds", "t"}
_HOURS_PARAM_NAMES = {"t_h", "time_h", "time_hours", "hours"}
_STEP_PARAM_NAMES = {"step", "k", "index", "iteration", "idx"}


def _get_profile_value(profile: Profile, step: int, dt_hours: float) -> float:
    """Return the value of a demand/supply profile at ``step``.

    Parameters
    ----------
    profile:
        Either a constant (``int`` or ``float``), a sequence indexed by the
        simulation step, or a callable.  Callables may accept the simulation
        time (in seconds or hours) or the discrete step index; see below.
    step:
        Zero-based simulation step.
    dt_hours:
        Simulation time-step expressed in hours.

    Behavior and supported callable signatures
    ------------------------------------------
    - If *profile* is callable, the function attempts to infer how to call it:
      1. Inspect the function signature. If the first positional parameter's
         name matches known time/step names, call the function with the
         corresponding single argument (time in seconds, time in hours, or step).
      2. Otherwise try calling with keyword arguments ``step``, ``time_hours``,
         and ``time_seconds`` (some callables accept keyword-only args).
      3. If signature introspection fails or the above attempts raise
         TypeError, fall back to trying to call the callable with a single
         argument in the order (time_seconds, time_hours, step), and finally
         step as a last resort. This preserves backwards compatibility.
    - If *profile* is a Sequence, return the element at index *step* if within
      range, otherwise return the last element (hold-last behaviour).
    - Otherwise coerce *profile* to float and return it.

    Returns
    -------
    float
        The profile value for the given step (converted to float).
    """
    if callable(profile):  # type: ignore[arg-type]
        # Precompute time values to pass to callables when needed.
        time_hours = step * dt_hours
        time_seconds = time_hours * 3600.0

        # Attempt to introspect the callable's signature to make an informed call.
        try:
            signature = inspect.signature(profile)
        except (TypeError, ValueError):
            # If signature can't be obtained, fall back to non-introspection paths.
            signature = None

        if signature is not None:
            # Collect positional (non-var) parameters to inspect the first one.
            positional = [
                parameter
                for parameter in signature.parameters.values()
                if parameter.kind
                   in (
                       inspect.Parameter.POSITIONAL_ONLY,
                       inspect.Parameter.POSITIONAL_OR_KEYWORD,
                   )
            ]

            if positional:
                first = positional[0]
                param_name = first.name

                # If the first positional parameter name indicates seconds, hours or step,
                # call with the corresponding single positional argument.
                if param_name in _SECONDS_PARAM_NAMES:
                    return float(profile(time_seconds))
                if param_name in _HOURS_PARAM_NAMES:
                    return float(profile(time_hours))
                if param_name in _STEP_PARAM_NAMES:
                    return float(profile(step))

                # If the first positional parameter is required but unrecognized,
                # default to the historic behaviour of passing the step index.
                if first.default is inspect._empty:
                    return float(profile(step))

            # If positional inspection didn't decide, try calling with keyword args.
            try:
                return float(
                    profile(
                        step=step,
                        time_hours=time_hours,
                        time_seconds=time_seconds,
                    )
                )
            except TypeError:
                # If the callable doesn't accept these keywords, fall through to
                # non-introspective fallbacks below.
                pass

        # Fallback strategy when introspection is unavailable or inconclusive:
        # try calling with time_seconds, then time_hours, then step as single arg.
        for argument in (time_seconds, time_hours, step):
            try:
                return float(profile(argument))
            except TypeError:
                continue
        # Final fallback: call with step (legacy behaviour).
        return float(profile(step))
    if isinstance(profile, Sequence):  # type: ignore[arg-type]
        # Sequence: return element at step when available, otherwise hold-last.
        if step < len(profile):
            return float(profile[step])
        return float(profile[-1])
    # Scalar numeric profile: coerce and return.
    return float(profile)


@dataclass(frozen=True)
class CellConfig:
    """Configuration parameters for a single CTM cell.

    Each field documents its physical meaning and expected units / constraints:

    - name: Human-readable cell identifier (string).
    - length_km: Cell length in kilometres (must be > 0).
    - lanes: Number of traffic lanes in the cell (integer, must be > 0).
    - free_flow_speed_kmh: Free-flow speed in km/h (must be > 0).
    - congestion_wave_speed_kmh: Backward wave (congestion) speed in km/h (must be > 0).
    - capacity_veh_per_hour_per_lane: Per-lane capacity in vehicles per hour (must be > 0).
    - jam_density_veh_per_km_per_lane: Jam density per lane in vehicles per km (must be > 0).
    - initial_density_veh_per_km_per_lane: Initial density per lane in vehicles per km (>= 0).

    Notes
    -----
    Validation is performed in ``__post_init__`` to catch common configuration
    mistakes early (non-positive geometry/FD parameters, negative densities or
    lane counts). The dataclass is frozen to make instances immutable after
    construction, ensuring cell configuration remains stable during simulation.
    """

    # Human-readable identifier for the cell (e.g., "cell_0")
    name: str
    # Cell length in kilometres (float > 0)
    length_km: float
    # Number of lanes (int > 0)
    lanes: int
    # Free-flow speed (km/h) used in sending calculation (float > 0)
    free_flow_speed_kmh: float
    # Backward congestion wave speed (km/h) used in receiving calculation (float > 0)
    congestion_wave_speed_kmh: float
    # Per-lane capacity (veh/h/lane) used to cap sending/receiving (float > 0)
    capacity_veh_per_hour_per_lane: float
    # Jam density per lane (veh/km/lane) used to compute available headroom (float > 0)
    jam_density_veh_per_km_per_lane: float
    # Initial density per lane (veh/km/lane). Defaults to 0.0 and must be non-negative.
    initial_density_veh_per_km_per_lane: float = 0.0

    def __post_init__(self) -> None:
        # Ensure the lane count is a positive integer.
        if self.lanes <= 0:
            raise ValueError("Number of lanes must be positive.")

        # Validate that geometric and fundamental-diagram parameters are positive.
        # These parameters are required to be strictly greater than zero to avoid
        # degenerate or non-physical behaviour in model calculations.
        for attr in (
                self.length_km,
                self.free_flow_speed_kmh,
                self.congestion_wave_speed_kmh,
                self.capacity_veh_per_hour_per_lane,
                self.jam_density_veh_per_km_per_lane,
        ):
            if attr <= 0:
                raise ValueError("Cell parameters must be positive.")

        # Initial density must be non-negative. (No automatic clamping to jam
        # density here; callers should ensure realistic initial conditions.)
        if self.initial_density_veh_per_km_per_lane < 0:
            raise ValueError("Initial density cannot be negative.")


@dataclass
class OnRampConfig:
    """Configuration for an on-ramp merging into a mainline cell.

    This dataclass holds both static configuration and a small amount of
    runtime state used by the simulator.

    Fields
    ------
    target_cell : Union[int, str]
        Index or name of the mainline cell this ramp connects to. Resolved to
        an integer index by the simulator during construction.
    arrival_rate_profile : Profile
        Arrival demand profile (veh/h). Can be a number, sequence or callable;
        semantics follow `_get_profile_value`.
    meter_rate_veh_per_hour : Optional[float]
        Optional ramp metering rate (veh/h). If ``None`` the ramp is unmetered.
        When ALINEA is enabled, this serves as the initial metering rate.
    mainline_priority : float
        Fraction in [0, 1] giving the initial priority share to the mainline
        when merging with ramp flow. A value of 0.5 gives equal priority.
    initial_queue_veh : float
        Initial queued vehicles on the ramp (veh). Used to initialise
        runtime ``queue_veh``.
    name : Optional[str]
        Optional human-readable identifier for the ramp. The simulator may
        assign a default name if omitted.
    alinea_enabled : bool
        Enable ALINEA ramp metering control (default: False).
    alinea_gain : float
        ALINEA controller gain Kr in (veh/h)/(veh/km/lane) (default: 70.0).
    alinea_target_density : float
        Target density for ALINEA controller in veh/km/lane (default: 30.0).
    alinea_min_rate : float
        Minimum metering rate for ALINEA in veh/h (default: 0.0).
    alinea_max_rate : float
        Maximum metering rate for ALINEA in veh/h (default: 2000.0).
    queue_veh : float
        Runtime queue size (veh). Marked ``init=False`` and initialised in
        ``__post_init__`` from ``initial_queue_veh``.
    """

    target_cell: Union[int, str]
    arrival_rate_profile: Profile
    meter_rate_veh_per_hour: Optional[float] = None
    mainline_priority: float = 0.5
    initial_queue_veh: float = 0.0
    name: Optional[str] = None
    
    # ALINEA ramp metering control
    alinea_enabled: bool = False
    alinea_gain: float = 70.0
    alinea_target_density: float = 30.0
    alinea_min_rate: float = 0.0
    alinea_max_rate: float = 2000.0

    # Runtime state: current queue size in vehicles. Not provided by caller.
    queue_veh: float = field(init=False)

    def __post_init__(self) -> None:
        # Validate that priority is a proper fraction.
        if not 0 <= self.mainline_priority <= 1:
            raise ValueError("mainline_priority must be within [0, 1].")

        # Meter rate, if provided, must be non-negative.
        if self.meter_rate_veh_per_hour is not None and self.meter_rate_veh_per_hour < 0:
            raise ValueError("meter_rate_veh_per_hour cannot be negative.")

        # Initial queue cannot be negative.
        if self.initial_queue_veh < 0:
            raise ValueError("initial_queue_veh cannot be negative.")
        
        # ALINEA validation
        if self.alinea_enabled:
            if self.alinea_gain <= 0:
                raise ValueError("alinea_gain must be positive.")
            if self.alinea_target_density <= 0:
                raise ValueError("alinea_target_density must be positive.")
            if self.alinea_min_rate < 0:
                raise ValueError("alinea_min_rate cannot be negative.")
            if self.alinea_max_rate <= 0:
                raise ValueError("alinea_max_rate must be positive.")
            if self.alinea_min_rate > self.alinea_max_rate:
                raise ValueError("alinea_min_rate cannot exceed alinea_max_rate.")
            # If ALINEA is enabled and no initial meter rate, use max rate
            if self.meter_rate_veh_per_hour is None:
                self.meter_rate_veh_per_hour = self.alinea_max_rate

        # Initialise the runtime queue as a float copy of the configured initial queue.
        self.queue_veh = float(self.initial_queue_veh)


@dataclass
class SimulationResult:
    r"""Container for the simulation time series.

    The simulator stores \*densities\* (veh/km/lane) with an initial value at
    ``t = 0`` followed by the state after each time step.  Flow-like series do
    not contain the initial condition, and instead align with the \*start\* of
    each simulation interval.  Helper methods are provided to make it easier to
    analyse or visualise the results without having to remember the exact
    layout.
    """

    # Densities: mapping from cell name -> list of densities (veh/km/lane).
    densities: Dict[str, List[float]]
    # Flows: mapping from link/cell name -> list of flows (veh/h) aligned with
    # simulation intervals (no initial condition entry).
    flows: Dict[str, List[float]]
    # Ramp queues: mapping ramp name -> list of queue sizes (veh) sampled at the
    # same times as densities (including initial value).
    ramp_queues: Dict[str, List[float]]
    # Ramp flows: mapping ramp name -> list of accepted ramp flows (veh/h)
    # aligned with simulation intervals (padded when converting to a DataFrame).
    ramp_flows: Dict[str, List[float]]
    # Upstream queue: list of queued vehicles (veh) at the upstream boundary,
    # sampled at the same times as densities (including initial value).
    upstream_queue: List[float]
    # Time step used by the simulation (hours).
    time_step_hours: float

    def __post_init__(self) -> None:
        # Basic validation to catch misconfigured results early.
        if self.time_step_hours <= 0:
            raise ValueError("time_step_hours must be positive.")
        if not self.densities:
            raise ValueError("densities cannot be empty.")
        
        # Validate time-series consistency:
        # - densities: length == steps (include initial sample at t=0)
        # - flows: length == steps - 1 (flows align with intervals)
        # - ramp_queues: length == steps
        # - ramp_flows: length == steps - 1
        # - upstream_queue: length == steps
        steps = self.steps
        expected_intervals = steps - 1
        
        # Check all density series have the same length
        for name, series in self.densities.items():
            if len(series) != steps:
                raise ValueError(
                    f"Density series '{name}' has length {len(series)}, "
                    f"expected {steps} (must match first series)."
                )
        
        # Check all flow series align with intervals
        for name, series in self.flows.items():
            if len(series) != expected_intervals:
                raise ValueError(
                    f"Flow series '{name}' has length {len(series)}, "
                    f"expected {expected_intervals} (steps - 1 for interval alignment)."
                )
        
        # Check all ramp queue series match density length
        for name, series in self.ramp_queues.items():
            if len(series) != steps:
                raise ValueError(
                    f"Ramp queue series '{name}' has length {len(series)}, "
                    f"expected {steps} (must match density time points)."
                )
        
        # Check all ramp flow series align with intervals
        for name, series in self.ramp_flows.items():
            if len(series) != expected_intervals:
                raise ValueError(
                    f"Ramp flow series '{name}' has length {len(series)}, "
                    f"expected {expected_intervals} (steps - 1 for interval alignment)."
                )
        
        # Check upstream queue series matches density length
        if len(self.upstream_queue) != steps:
            raise ValueError(
                f"Upstream queue series has length {len(self.upstream_queue)}, "
                f"expected {steps} (must match density time points)."
            )

    @property
    def duration_hours(self) -> float:
        """Return the simulated duration in hours.

        The duration is computed from the number of recorded density samples.
        Because densities include the initial state at t=0, the total simulated
        duration equals (steps - 1) * dt.
        """
        return (self.steps - 1) * self.time_step_hours

    @property
    def steps(self) -> int:
        """Return the number of recorded density samples.

        All density series are expected to have the same length.  The first
        density series encountered is used to determine the number of samples.
        """
        first_series = next(iter(self.densities.values()))
        return len(first_series)

    def time_vector(self) -> List[float]:
        """Return the time vector (hours) corresponding to density samples.

        Produces a list [0, dt, 2*dt, ...] with length equal to ``self.steps``.
        """
        return [idx * self.time_step_hours for idx in range(self.steps)]

    def interval_vector(self) -> List[Tuple[float, float]]:
        """Return (start, end) pairs in hours for flow-aligned data.

        Flow and ramp_flow series align with simulation intervals; this method
        returns the corresponding list of (start, end) times for each interval.
        """
        times = self.time_vector()
        # Pair consecutive times to form intervals; length = steps - 1.
        return list(zip(times[:-1], times[1:]))

    def to_dataframe(self) -> "pd.DataFrame":
        """Return a tidy pandas ``DataFrame`` with the simulation outputs.

        The method requires :mod:`pandas` to be available.  If the dependency is
        missing, a ``RuntimeError`` with installation instructions is raised.
        The resulting frame uses a ``MultiIndex`` over the columns with levels
        ``(kind, location)`` so that densities and flows can be selected either
        jointly or independently.

        Notes:
        - Density series are inserted as-is (they include the initial state).
        - Flow-like series (flows, ramp_flows) are padded by repeating the last
          value so that they have the same length as density series.  If a
          flow series is empty it is padded with 0.0.
        - Ramp queues are recorded at the same times as densities and inserted
          without padding.
        - Upstream queue is recorded at the same times as densities and inserted
          without padding.
        """
        try:
            import pandas as pd
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "pandas is required for `SimulationResult.to_dataframe()`."
                " Install it with `pip install pandas`."
            ) from exc

        time = self.time_vector()
        # Build a mapping (kind, id) -> series for DataFrame construction.
        data: Dict[Tuple[str, str], List[float]] = {}

        # Add densities directly; they include the initial sample at t=0.
        for name, series in self.densities.items():
            data[("density", name)] = series

        # Flows align with intervals; pad by repeating the last value to match
        # the density time vector length. If a flow series is empty, use 0.0.
        for name, series in self.flows.items():
            padded = series + [series[-1] if series else 0.0]
            data[("flow", name)] = padded

        # Ramp queues are recorded with densities (including initial queue).
        for name, series in self.ramp_queues.items():
            data[("ramp_queue", name)] = series

        # Ramp flows align with intervals; pad similarly to mainline flows.
        for name, series in self.ramp_flows.items():
            padded = series + [series[-1] if series else 0.0]
            data[("ramp_flow", name)] = padded

        # Upstream queue is recorded with densities (including initial queue).
        data[("upstream_queue", "boundary")] = self.upstream_queue

        # Sort column keys to produce stable column ordering in the DataFrame.
        columns = pd.MultiIndex.from_tuples(sorted(data), names=["kind", "id"])
        return pd.DataFrame(data, index=time, columns=columns)

    def compute_vkt_vht(self, cells: Sequence[CellConfig]) -> Tuple[float, float]:
        """Compute Vehicle Kilometers Traveled (VKT) and Vehicle Hours Traveled (VHT).
        
        VKT measures total distance traveled by all vehicles:
            VKT = sum over all cells and intervals of (flow * dt * cell_length)
        
        VHT measures total time spent by all vehicles:
            VHT = sum over all cells and intervals of (avg_vehicles_in_cell * dt)
            where avg_vehicles_in_cell = 0.5 * (rho_t + rho_{t+1}) * lanes * length
        
        Parameters
        ----------
        cells : Sequence[CellConfig]
            Cell configurations containing geometry (length, lanes) and names.
        
        Returns
        -------
        vkt : float
            Total vehicle kilometers traveled.
        vht : float
            Total vehicle hours traveled.
        
        Notes
        -----
        - Flows are in veh/h, time_step_hours is in hours, cell_length_km is in km.
        - Densities are per-lane (veh/km/lane), so multiply by lanes for total density.
        - The trapezoidal rule is used for VHT: average of density at t and t+1.
        """
        dt = self.time_step_hours
        intervals = self.steps - 1
        
        # Create a mapping from cell name to cell config for quick lookup
        cell_map = {cell.name: cell for cell in cells}
        
        total_vkt = 0.0
        total_vht = 0.0
        
        # Compute VKT from flows
        for name, flow_series in self.flows.items():
            cell = cell_map.get(name)
            if cell is None:
                continue  # Skip flows not associated with a mainline cell
            
            for i in range(intervals):
                # Get flow for this interval (already validated to have correct length)
                f = float(flow_series[i])
                # VKT for this cell-interval: flow (veh/h) * dt (h) * length (km)
                total_vkt += f * dt * cell.length_km
        
        # Compute VHT from densities
        for name, rho_series in self.densities.items():
            cell = cell_map.get(name)
            if cell is None:
                continue  # Skip densities not associated with a mainline cell
            
            for i in range(intervals):
                # Get densities at start and end of interval
                rho_t = float(rho_series[i])
                rho_tp1 = float(rho_series[i + 1])
                
                # Average per-lane density over the interval
                avg_rho_per_lane = 0.5 * (rho_t + rho_tp1)
                
                # Total vehicles in cell (average over interval)
                # = avg_density (veh/km/lane) * lanes * length (km)
                vehicles_in_cell = avg_rho_per_lane * cell.lanes * cell.length_km
                
                # VHT for this cell-interval: vehicles * dt (h)
                total_vht += vehicles_in_cell * dt
        
        return total_vkt, total_vht
    
    def compute_velocity_series(
        self, cells: Sequence[CellConfig]
    ) -> Dict[str, List[float]]:
        """Compute velocity time series for each mainline cell.
        
        For each cell and each interval, computes velocity as:
            v = flow / avg_density_total
        where:
            - flow is in veh/h (from flows series)
            - avg_density_total = 0.5 * (rho_t + rho_{t+1}) * lanes (veh/km total)
            - resulting velocity is in km/h
        
        Parameters
        ----------
        cells : Sequence[CellConfig]
            Cell configurations containing geometry (lanes) and names.
        
        Returns
        -------
        velocities : Dict[str, List[float]]
            Mapping from cell name to velocity series (km/h).
            Each series has length == steps - 1 (aligned with intervals/flows).
            When avg_density_total == 0, velocity is set to 0.0.
        
        Notes
        -----
        - Velocity is computed for each interval using the average density.
        - When density is zero (empty cell), velocity is set to 0.0 to avoid
          division by zero. This is physically reasonable: no vehicles means
          no meaningful velocity measurement.
        """
        intervals = self.steps - 1
        cell_map = {cell.name: cell for cell in cells}
        
        velocities: Dict[str, List[float]] = {}
        
        for name in self.flows.keys():
            cell = cell_map.get(name)
            if cell is None:
                continue  # Skip flows not associated with a mainline cell
            
            flow_series = self.flows[name]
            rho_series = self.densities.get(name)
            
            if rho_series is None:
                # No density data for this cell; can't compute velocity
                continue
            
            velocity_series: List[float] = []
            
            for i in range(intervals):
                # Get flow for this interval
                flow = float(flow_series[i])
                
                # Get densities at start and end of interval
                rho_t = float(rho_series[i])
                rho_tp1 = float(rho_series[i + 1])
                
                # Average per-lane density
                avg_rho_per_lane = 0.5 * (rho_t + rho_tp1)
                
                # Total average density (veh/km total across all lanes)
                avg_rho_total = avg_rho_per_lane * cell.lanes
                
                # Compute velocity: flow (veh/h) / density (veh/km) = km/h
                if avg_rho_total > 0:
                    velocity = flow / avg_rho_total
                else:
                    # No vehicles in cell; set velocity to 0
                    velocity = 0.0
                
                velocity_series.append(velocity)
            
            velocities[name] = velocity_series
        
        return velocities



class CTMSimulation:
    """Cell Transmission Model simulator supporting on-ramps.

    This class implements a simple, pedagogical CTM with:
    * heterogeneous cells (length, lanes, fundamental diagram params),
    * a single upstream demand boundary and optional downstream supply,
    * optional on-ramps that merge into mainline cells with queueing and
      metering.

    Public methods:
    * __init__: configure the simulation instance.
    * run: execute the time-stepping loop and return a SimulationResult.

    Internal helpers implement the CTM mechanics: sending/receiving calculations,
    ramp queue updates, merging logic and density integration.
    """

    def __init__(
            self,
            cells: Sequence[CellConfig],
            time_step_hours: float,
            upstream_demand_profile: Profile,
            downstream_supply_profile: Optional[Profile] = None,
            on_ramps: Optional[Sequence[OnRampConfig]] = None,
    ) -> None:
        """Construct a CTMSimulation.

        Parameters
        ----------
        cells:
            Sequence of configured :class:`CellConfig` objects describing the
            mainline cells in upstream-to-downstream order.
        time_step_hours:
            Simulation time step in hours (must be positive).
        upstream_demand_profile:
            Profile providing upstream arrival demand.  May be a scalar, a
            sequence or a callable (see module `_get_profile_value` for
            accepted callable signatures).
        downstream_supply_profile:
            Optional profile constraining downstream discharge flow.  If
            omitted the last cell's capacity is used.
        on_ramps:
            Optional sequence of :class:`OnRampConfig`.  At construction ramp
            targets are resolved to integer indices and each ramp is assigned a
            stable name.  Only one on-ramp per target cell is supported.

        Raises
        ------
        ValueError
            If `time_step_hours` is not positive or `cells` is empty.
        IndexError, KeyError
            If an on-ramp target refers to an invalid index or unknown name.
        """
        if time_step_hours <= 0:
            raise ValueError("time_step_hours must be positive.")
        if not cells:
            raise ValueError("At least one cell configuration is required.")

        self.cells: List[CellConfig] = list(cells)
        self.dt = float(time_step_hours)
        self.upstream_profile = upstream_demand_profile
        self.downstream_profile = downstream_supply_profile

        # Map cell names to indices for name-based ramp targeting.
        self._cell_index: Dict[str, int] = {cell.name: idx for idx, cell in enumerate(self.cells)}

        # Normalise and store on-ramps: resolve targets and ensure unique names.
        self.on_ramps: List[OnRampConfig] = []
        if on_ramps:
            for ramp in on_ramps:
                target_index = self._resolve_target_index(ramp.target_cell)
                object.__setattr__(ramp, "target_cell", target_index)
                ramp_name = ramp.name or f"ramp_{target_index}"
                object.__setattr__(ramp, "name", ramp_name)
                self.on_ramps.append(ramp)

        # Build quick lookup from cell index -> ramp config, disallowing >1 ramp/cell.
        self._ramps_by_cell: Dict[int, OnRampConfig] = {}
        for ramp in self.on_ramps:
            target_index = int(ramp.target_cell)
            if target_index in self._ramps_by_cell:
                raise ValueError(
                    "Only one on-ramp per target cell is supported by this simulator."
                )
            self._ramps_by_cell[target_index] = ramp

    def _resolve_target_index(self, target: Union[int, str]) -> int:
        """Resolve an on-ramp target (index or cell name) to an integer index.

        Parameters
        ----------
        target
            Either an integer index or a string cell name.

        Returns
        -------
        int
            The integer index of the target cell.

        Raises
        ------
        IndexError
            If the integer index is out of range.
        KeyError
            If a string name is not found among configured cells.
        """
        if isinstance(target, int):
            if not 0 <= target < len(self.cells):
                raise IndexError("On-ramp target cell index out of range.")
            return target
        if target not in self._cell_index:
            raise KeyError(f"Unknown cell name '{target}'.")
        return self._cell_index[target]

    def run(self, steps: int) -> SimulationResult:
        """Run the CTM simulation for the requested number of time steps.

        The simulator records densities with an initial value at t=0 followed by
        the state after each time step; flows and ramp flows align with
        simulation intervals.

        Parameters
        ----------
        steps
            Number of discrete simulation steps to execute (must be positive).

        Returns
        -------
        SimulationResult
            Container with time series for densities, flows, ramp queues/flows,
            and upstream queue.
        """
        if steps <= 0:
            raise ValueError("steps must be a positive integer.")

        # Initialise per-cell densities and history containers.
        densities = [cell.initial_density_veh_per_km_per_lane for cell in self.cells]
        density_history: Dict[str, List[float]] = {
            cell.name: [density] for cell, density in zip(self.cells, densities)
        }
        flow_history: Dict[str, List[float]] = {cell.name: [] for cell in self.cells}
        ramp_queue_history: Dict[str, List[float]] = {
            ramp.name: [ramp.queue_veh] for ramp in self.on_ramps
        }
        ramp_flow_history: Dict[str, List[float]] = {ramp.name: [] for ramp in self.on_ramps}
        
        # Initialize upstream queue (vehicles waiting to enter the first cell)
        upstream_queue_veh = 0.0
        upstream_queue_history: List[float] = [upstream_queue_veh]

        # Main time-stepping loop.
        for step in range(steps):
            # Apply ALINEA ramp metering control
            self._apply_alinea_control(densities, step)
            
            # Compute sending/receiving capacities from current densities.
            sending = self._compute_sending(densities)
            receiving = self._compute_receiving(densities)

            # Update on-ramp queues and get potential ramp contributions for this step.
            ramp_potentials = self._update_ramp_queues(step)
            ramp_flows_step: Dict[str, float] = {ramp.name: 0.0 for ramp in self.on_ramps}

            # Prepare inflow/outflow accumulators.
            inflows = [0.0] * len(self.cells)
            outflows = [0.0] * len(self.cells)

            # Evaluate upstream demand and downstream supply for this step.
            upstream_demand_raw = _get_profile_value(self.upstream_profile, step, self.dt)
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
            # Constrain downstream supply to [0, last_cell_capacity].
            downstream_supply = max(0.0, min(downstream_supply_raw, max_downstream_flow))

            # Add new arrivals to upstream queue (veh/h * dt_hours -> veh)
            upstream_queue_veh += upstream_demand_raw * self.dt
            
            # Compute flows into each cell (including upstream boundary to first cell).
            for idx in range(len(self.cells)):
                supply = receiving[idx]
                
                # For the first cell, mainline demand comes from the upstream queue
                if idx == 0:
                    # Convert queue to an hourly rate
                    queue_potential = upstream_queue_veh / self.dt if self.dt > 0 else 0.0
                    mainline_demand = queue_potential
                else:
                    mainline_demand = sending[idx - 1]
                
                ramp_flow = 0.0

                ramp = self._ramps_by_cell.get(idx)
                if ramp is not None:
                    # Merge mainline and ramp according to priority and available supply.
                    potential = ramp_potentials[ramp.name]
                    main_flow, ramp_flow = self._merge_flows(
                        mainline_demand,
                        potential,
                        supply,
                        ramp.mainline_priority,
                    )
                    # Dequeue ramp vehicles according to the meter/ramp_flow (veh/h -> veh over dt).
                    ramp.queue_veh = max(0.0, ramp.queue_veh - ramp_flow * self.dt)
                    ramp_flows_step[ramp.name] = ramp_flow
                else:
                    # No ramp: mainline accepts as much as supply allows.
                    main_flow = min(mainline_demand, supply)

                inflow = main_flow + ramp_flow
                inflows[idx] += inflow

                if idx > 0:
                    # The outflow from the upstream cell equals the accepted mainline flow.
                    outflows[idx - 1] = main_flow
                else:
                    # For the first cell, remove accepted flow from upstream queue
                    upstream_queue_veh = max(0.0, upstream_queue_veh - main_flow * self.dt)

            # Handle flow out of the last cell to downstream boundary.
            last_idx = len(self.cells) - 1
            outflow_last = min(sending[last_idx], downstream_supply)
            outflows[last_idx] = outflow_last
            flow_history[self.cells[last_idx].name].append(outflow_last)

            # Record flows for internal links (all except the last cell's downstream).
            for idx in range(len(self.cells) - 1):
                flow_history[self.cells[idx].name].append(outflows[idx])

            # Save ramp flow and queue histories for this step.
            for ramp in self.on_ramps:
                ramp_flow_history[ramp.name].append(ramp_flows_step[ramp.name])
                ramp_queue_history[ramp.name].append(ramp.queue_veh)

            # Save upstream queue history for this step (after the update)
            upstream_queue_history.append(upstream_queue_veh)

            # Integrate densities using conservation: rho_new = rho + (dt/length) * (inflow - outflow)/lanes
            for idx, cell in enumerate(self.cells):
                inflow = inflows[idx]
                outflow = outflows[idx]
                densities[idx] = self._update_density(
                    density=densities[idx],
                    inflow=inflow,
                    outflow=outflow,
                    cell=cell,
                )
                density_history[cell.name].append(densities[idx])

        # Package and return results.
        return SimulationResult(
            densities=density_history,
            flows=flow_history,
            ramp_queues=ramp_queue_history,
            ramp_flows=ramp_flow_history,
            upstream_queue=upstream_queue_history,
            time_step_hours=self.dt,
        )

    def _compute_sending(self, densities: Sequence[float]) -> List[float]:
        """Compute sending flow (veh/h) from each cell given current densities.

        For a triangular-like FD the sending is min(free_flow_speed * rho * lanes, total_capacity).
        """
        sending = []
        for density, cell in zip(densities, self.cells):
            total_capacity = cell.capacity_veh_per_hour_per_lane * cell.lanes
            demand = cell.free_flow_speed_kmh * density * cell.lanes
            sending.append(min(demand, total_capacity))
        return sending

    def _compute_receiving(self, densities: Sequence[float]) -> List[float]:
        """Compute receiving flow (veh/h) for each cell.

        Receiving is proportional to the available headroom up to cell capacity:
        min(congestion_wave_speed * (rho_j - rho) * lanes, total_capacity).
        """
        receiving = []
        for density, cell in zip(densities, self.cells):
            total_capacity = cell.capacity_veh_per_hour_per_lane * cell.lanes
            remaining_density = max(0.0, cell.jam_density_veh_per_km_per_lane - density)
            supply = cell.congestion_wave_speed_kmh * remaining_density * cell.lanes
            receiving.append(min(supply, total_capacity))
        return receiving

    def _update_ramp_queues(self, step: int) -> Dict[str, float]:
        """Advance on-ramp queues by arrivals and compute the ramp potential (veh/h).

        Returns a mapping ramp.name -> potential ramp flow (veh/h) available to be
        merged this step.  Metering limits are respected.
        """
        potentials: Dict[str, float] = {}
        for ramp in self.on_ramps:
            arrivals = _get_profile_value(ramp.arrival_rate_profile, step, self.dt)
            # Add arrivals (veh/h * dt_hours -> veh).
            ramp.queue_veh += arrivals * self.dt
            meter_rate = ramp.meter_rate_veh_per_hour
            # Potential is the queued vehicles expressed as an equivalent hourly rate.
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

        A simple priority-based split is used: allocate the priority shares,
        then redistribute any leftover supply proportionally to remaining demands.

        Parameters
        ----------
        mainline_demand, ramp_demand:
            Demands in veh/h.
        supply:
            Available receiving capacity in veh/h.
        mainline_priority:
            Fraction in [0, 1] giving initial share to mainline.

        Returns
        -------
        (main_flow, ramp_flow)
            Accepted flows (veh/h) from mainline and ramp respectively.
        """
        if supply <= 0:
            return 0.0, 0.0

        mainline_demand = max(0.0, mainline_demand)
        ramp_demand = max(0.0, ramp_demand)

        # If total demand fits, accept everything.
        if mainline_demand + ramp_demand <= supply:
            return mainline_demand, ramp_demand

        mainline_share = mainline_priority
        ramp_share = 1.0 - mainline_share

        # Initial allocation by priority share (bounded by each demand).
        main_flow = min(mainline_demand, mainline_share * supply)
        ramp_flow = min(ramp_demand, ramp_share * supply)

        # Distribute any remaining supply proportionally to leftover demand.
        remaining_supply = supply - main_flow - ramp_flow
        remaining_main_demand = max(0.0, mainline_demand - main_flow)
        remaining_ramp_demand = max(0.0, ramp_demand - ramp_flow)

        if remaining_supply > 0:
            total_remaining_demand = remaining_main_demand + remaining_ramp_demand
            if total_remaining_demand > 0:
                additional_main = remaining_supply * (remaining_main_demand / total_remaining_demand)
                additional_ramp = remaining_supply - additional_main
                main_flow += min(additional_main, remaining_main_demand)
                ramp_flow += min(additional_ramp, remaining_ramp_demand)

        return main_flow, ramp_flow

    def _apply_alinea_control(self, densities: Sequence[float], step: int) -> None:
        """Apply ALINEA ramp metering control to update meter rates.
        
        ALINEA is a feedback controller that adjusts ramp metering rates based on
        downstream occupancy (density). The control law is:
        
            r(k) = r(k-1) + Kr * (ρ_target - ρ_downstream(k))
        
        where:
            - r(k) is the metering rate at time k (veh/h)
            - Kr is the controller gain ((veh/h)/(veh/km/lane))
            - ρ_target is the target density (veh/km/lane)
            - ρ_downstream(k) is the measured downstream density (veh/km/lane)
        
        Parameters
        ----------
        densities : Sequence[float]
            Current per-lane densities for all cells (veh/km/lane).
        step : int
            Current simulation step (used for logging/debugging).
        
        Notes
        -----
        The metering rate is clamped to [alinea_min_rate, alinea_max_rate].
        If ALINEA is not enabled for a ramp, this method does nothing.
        """
        for ramp in self.on_ramps:
            if not ramp.alinea_enabled:
                continue
            
            # Get downstream density (target cell where ramp merges)
            target_idx = int(ramp.target_cell)
            downstream_density = densities[target_idx]
            
            # Compute density error (target - actual)
            density_error = ramp.alinea_target_density - downstream_density
            
            # Apply ALINEA control law
            # Units: (veh/h) + ((veh/h)/(veh/km/lane)) * (veh/km/lane) = veh/h ✓
            rate_adjustment = ramp.alinea_gain * density_error
            new_rate = ramp.meter_rate_veh_per_hour + rate_adjustment
            
            # Clamp to physical bounds
            new_rate = max(ramp.alinea_min_rate, min(new_rate, ramp.alinea_max_rate))
            
            # Update meter rate
            ramp.meter_rate_veh_per_hour = new_rate

    def _update_density(
            self,
            density: float,
            inflow: float,
            outflow: float,
            cell: CellConfig,
    ) -> float:
        """Integrate density for a single cell using conservation of vehicles.

        The change is (dt / length_km) * ((inflow - outflow) / lanes), where inflow
        and outflow are in veh/h and dt is in hours.  Result is clamped to [0, rho_j].
        """
        change = (self.dt / cell.length_km) * (
                inflow / cell.lanes - outflow / cell.lanes
        )
        new_density = density + change
        return max(0.0, min(new_density, cell.jam_density_veh_per_km_per_lane))


def build_uniform_mainline(
        *,
        num_cells: int,
        cell_length_km: float,
        lanes: Sequence[int] | int,
        free_flow_speed_kmh: float,
        congestion_wave_speed_kmh: float,
        capacity_veh_per_hour_per_lane: float,
        jam_density_veh_per_km_per_lane: float,
        initial_density_veh_per_km_per_lane: float | Sequence[float] = 0.0,
) -> List[CellConfig]:
    """
    Create a list of :class:`CellConfig` objects representing a uniform mainline.

    This helper constructs `num_cells` mainline cells using the provided scalar
    CTM parameters.  It accepts either a single integer for `lanes` (applied to
    every cell) or a sequence of lane counts (one per cell).  Similarly,
    `initial_density_veh_per_km_per_lane` may be a scalar applied to every
    cell or a sequence with one initial density per cell.

    Parameters
    ----------
    num_cells:
        Number of mainline cells to create.  Must be positive.
    cell_length_km:
        Length of each cell in kilometres.  Used directly for each cell's
        `length_km` attribute.
    lanes:
        Either a single integer applied to every cell, or a sequence with one
        integer per cell to allow lane drops/additions.
    free_flow_speed_kmh:
        Free flow speed (km/h) applied to all cells.
    congestion_wave_speed_kmh:
        Backward wave speed (km/h) applied to all cells.
    capacity_veh_per_hour_per_lane:
        Per-lane capacity (veh/h/lane) applied to all cells.
    jam_density_veh_per_km_per_lane:
        Per-lane jam density (veh/km/lane) applied to all cells.
    initial_density_veh_per_km_per_lane:
        Optional scalar or sequence describing the starting density per lane for
        each cell.  If a sequence is provided its length must equal `num_cells`.

    Returns
    -------
    List[CellConfig]
        A list of configured :class:`CellConfig` instances named `cell_0`,
        `cell_1`, ..., `cell_{num_cells-1}`.

    Raises
    ------
    ValueError
        If `num_cells` is not positive, or if the provided `lanes` or
        `initial_density_veh_per_km_per_lane` sequences do not match
        `num_cells`.
    """
    # Validate basic argument values.
    if num_cells <= 0:
        # `num_cells` must be positive to construct cells.
        raise ValueError("num_cells must be positive.")

    # Normalize lanes into a list of length `num_cells`.
    if isinstance(lanes, int):
        # Single integer: replicate for every cell.
        lane_profile = [lanes] * num_cells
    else:
        # Sequence: convert to list and validate length.
        lane_profile = list(lanes)
        if len(lane_profile) != num_cells:
            raise ValueError("lanes sequence must match num_cells.")

    # Normalize initial densities into a list of length `num_cells`.
    if isinstance(initial_density_veh_per_km_per_lane, (int, float)):
        # Scalar initial density: replicate for every cell.
        density_profile = [float(initial_density_veh_per_km_per_lane)] * num_cells
    else:
        # Sequence: coerce to float list and validate length.
        density_profile = [float(value) for value in initial_density_veh_per_km_per_lane]
        if len(density_profile) != num_cells:
            raise ValueError("initial_density sequence must match num_cells.")

    # Construct CellConfig objects for each cell index.
    cells: List[CellConfig] = []
    for idx in range(num_cells):
        # Each cell receives the same FD and geometric parameters, with per-cell
        # lane and initial density from the normalized profiles.
        cells.append(
            CellConfig(
                name=f"cell_{idx}",
                length_km=cell_length_km,
                lanes=lane_profile[idx],
                free_flow_speed_kmh=free_flow_speed_kmh,
                congestion_wave_speed_kmh=congestion_wave_speed_kmh,
                capacity_veh_per_hour_per_lane=capacity_veh_per_hour_per_lane,
                jam_density_veh_per_km_per_lane=jam_density_veh_per_km_per_lane,
                initial_density_veh_per_km_per_lane=density_profile[idx],
            )
        )
    # Return the list of configured cells.
    return cells


def run_basic_scenario(steps: int = 20) -> SimulationResult:
    """Run a small demonstration scenario and return its results.

    The scenario uses three mainline cells, a single on-ramp feeding the second
    cell, and a triangular upstream demand profile to demonstrate different
    congestion regimes.  The helper is primarily intended for users exploring
    the module for the first time.
    """

    def triangular_profile(step: int) -> float:
        peak = 1800.0
        if step < 5:
            return peak * (step / 5)
        if step < 10:
            return peak
        return max(0.0, peak - 200.0 * (step - 10))

    cells = build_uniform_mainline(
        num_cells=3,
        cell_length_km=0.5,
        lanes=[3, 3, 2],
        free_flow_speed_kmh=100.0,
        congestion_wave_speed_kmh=20.0,
        capacity_veh_per_hour_per_lane=2200.0,
        jam_density_veh_per_km_per_lane=160.0,
        initial_density_veh_per_km_per_lane=[20.0, 25.0, 30.0],
    )

    on_ramp = OnRampConfig(
        target_cell=1,
        arrival_rate_profile=[300.0] * steps,
        meter_rate_veh_per_hour=600.0,
        mainline_priority=0.6,
        initial_queue_veh=10.0,
        name="demo_ramp",
    )

    sim = CTMSimulation(
        cells=cells,
        time_step_hours=0.1,
        upstream_demand_profile=triangular_profile,
        downstream_supply_profile=2000.0,
        on_ramps=[on_ramp],
    )
    return sim.run(steps)


__all__ = [
    "CTMSimulation",
    "CellConfig",
    "OnRampConfig",
    "SimulationResult",
    "build_uniform_mainline",
    "run_basic_scenario",
]

if __name__ == "__main__":  # pragma: no cover - convenience usage
    result = run_basic_scenario(steps=20)
    try:
        df = result.to_dataframe()
    except RuntimeError:
        from pprint import pprint

        print("pandas not installed - printing raw dictionaries instead\n")
        pprint({
            "time_hours": result.time_vector(),
            "densities": result.densities,
            "flows": result.flows,
            "ramp_queues": result.ramp_queues,
            "ramp_flows": result.ramp_flows,
        })
    else:
        print(df.head())
