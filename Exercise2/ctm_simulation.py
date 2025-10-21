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

from dataclasses import dataclass, field
import inspect
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
    """

    if callable(profile):  # type: ignore[arg-type]
        time_hours = step * dt_hours
        time_seconds = time_hours * 3600.0

        try:
            signature = inspect.signature(profile)
        except (TypeError, ValueError):
            signature = None

        if signature is not None:
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

                if param_name in _SECONDS_PARAM_NAMES:
                    return float(profile(time_seconds))
                if param_name in _HOURS_PARAM_NAMES:
                    return float(profile(time_hours))
                if param_name in _STEP_PARAM_NAMES:
                    return float(profile(step))

                if first.default is inspect._empty:
                    # Unknown required positional parameter – default to the
                    # legacy behaviour (step index) for backwards compatibility.
                    return float(profile(step))

            try:
                return float(
                    profile(
                        step=step,
                        time_hours=time_hours,
                        time_seconds=time_seconds,
                    )
                )
            except TypeError:
                # Fall back to legacy behaviour below.
                pass

        # Default to historic semantics (step index) and offer additional
        # fallbacks so that callables expecting time-based arguments can still
        # be used even when signature introspection fails.
        for argument in (time_seconds, time_hours, step):
            try:
                return float(profile(argument))
            except TypeError:
                continue
        return float(profile(step))
    if isinstance(profile, Sequence):  # type: ignore[arg-type]
        if step < len(profile):
            return float(profile[step])
        return float(profile[-1])
    return float(profile)


@dataclass(frozen=True)
class CellConfig:
    """Configuration parameters for a single CTM cell."""

    name: str
    length_km: float
    lanes: int
    free_flow_speed_kmh: float
    congestion_wave_speed_kmh: float
    capacity_veh_per_hour_per_lane: float
    jam_density_veh_per_km_per_lane: float
    initial_density_veh_per_km_per_lane: float = 0.0

    def __post_init__(self) -> None:
        if self.lanes <= 0:
            raise ValueError("Number of lanes must be positive.")
        for attr in (
            self.length_km,
            self.free_flow_speed_kmh,
            self.congestion_wave_speed_kmh,
            self.capacity_veh_per_hour_per_lane,
            self.jam_density_veh_per_km_per_lane,
        ):
            if attr <= 0:
                raise ValueError("Cell parameters must be positive.")
        if self.initial_density_veh_per_km_per_lane < 0:
            raise ValueError("Initial density cannot be negative.")


@dataclass
class OnRampConfig:
    """Configuration for an on-ramp merging into a mainline cell."""

    target_cell: Union[int, str]
    arrival_rate_profile: Profile
    meter_rate_veh_per_hour: Optional[float] = None
    mainline_priority: float = 0.5
    initial_queue_veh: float = 0.0
    name: Optional[str] = None

    queue_veh: float = field(init=False)

    def __post_init__(self) -> None:
        if not 0 <= self.mainline_priority <= 1:
            raise ValueError("mainline_priority must be within [0, 1].")
        if self.meter_rate_veh_per_hour is not None and self.meter_rate_veh_per_hour < 0:
            raise ValueError("meter_rate_veh_per_hour cannot be negative.")
        if self.initial_queue_veh < 0:
            raise ValueError("initial_queue_veh cannot be negative.")
        self.queue_veh = float(self.initial_queue_veh)


@dataclass
class SimulationResult:
    """Container for the simulation time series.

    The simulator stores *densities* (veh/km/lane) with an initial value at
    ``t = 0`` followed by the state after each time step.  Flow-like series do
    not contain the initial condition, and instead align with the *start* of
    each simulation interval.  Helper methods are provided to make it easier to
    analyse or visualise the results without having to remember the exact
    layout.
    """

    densities: Dict[str, List[float]]
    flows: Dict[str, List[float]]
    ramp_queues: Dict[str, List[float]]
    ramp_flows: Dict[str, List[float]]
    time_step_hours: float

    def __post_init__(self) -> None:
        if self.time_step_hours <= 0:
            raise ValueError("time_step_hours must be positive.")
        if not self.densities:
            raise ValueError("densities cannot be empty.")

    @property
    def duration_hours(self) -> float:
        """Return the simulated duration in hours."""

        return (self.steps - 1) * self.time_step_hours

    @property
    def steps(self) -> int:
        """Return the number of recorded density samples."""

        first_series = next(iter(self.densities.values()))
        return len(first_series)

    def time_vector(self) -> List[float]:
        """Return the time vector (hours) corresponding to density samples."""

        return [idx * self.time_step_hours for idx in range(self.steps)]

    def interval_vector(self) -> List[Tuple[float, float]]:
        """Return (start, end) pairs in hours for flow-aligned data."""

        times = self.time_vector()
        return list(zip(times[:-1], times[1:]))

    def to_dataframe(self) -> "pd.DataFrame":
        """Return a tidy pandas ``DataFrame`` with the simulation outputs.

        The method requires :mod:`pandas` to be available.  If the dependency is
        missing, a ``RuntimeError`` with installation instructions is raised.
        The resulting frame uses a ``MultiIndex`` over the columns with levels
        ``(kind, location)`` so that densities and flows can be selected either
        jointly or independently.
        """

        try:
            import pandas as pd
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "pandas is required for `SimulationResult.to_dataframe()`."
                " Install it with `pip install pandas`."
            ) from exc

        time = self.time_vector()
        data: Dict[Tuple[str, str], List[float]] = {}

        for name, series in self.densities.items():
            data[("density", name)] = series
        for name, series in self.flows.items():
            padded = series + [series[-1] if series else 0.0]
            data[("flow", name)] = padded
        for name, series in self.ramp_queues.items():
            data[("ramp_queue", name)] = series
        for name, series in self.ramp_flows.items():
            padded = series + [series[-1] if series else 0.0]
            data[("ramp_flow", name)] = padded

        columns = pd.MultiIndex.from_tuples(sorted(data), names=["kind", "id"])
        return pd.DataFrame(data, index=time, columns=columns)


class CTMSimulation:
    """Cell Transmission Model simulator supporting on-ramps."""

    def __init__(
        self,
        cells: Sequence[CellConfig],
        time_step_hours: float,
        upstream_demand_profile: Profile,
        downstream_supply_profile: Optional[Profile] = None,
        on_ramps: Optional[Sequence[OnRampConfig]] = None,
    ) -> None:
        if time_step_hours <= 0:
            raise ValueError("time_step_hours must be positive.")
        if not cells:
            raise ValueError("At least one cell configuration is required.")

        self.cells: List[CellConfig] = list(cells)
        self.dt = float(time_step_hours)
        self.upstream_profile = upstream_demand_profile
        self.downstream_profile = downstream_supply_profile

        self._cell_index: Dict[str, int] = {cell.name: idx for idx, cell in enumerate(self.cells)}

        self.on_ramps: List[OnRampConfig] = []
        if on_ramps:
            for ramp in on_ramps:
                target_index = self._resolve_target_index(ramp.target_cell)
                object.__setattr__(ramp, "target_cell", target_index)
                ramp_name = ramp.name or f"ramp_{target_index}"
                object.__setattr__(ramp, "name", ramp_name)
                self.on_ramps.append(ramp)

        self._ramps_by_cell: Dict[int, OnRampConfig] = {}
        for ramp in self.on_ramps:
            target_index = int(ramp.target_cell)
            if target_index in self._ramps_by_cell:
                raise ValueError(
                    "Only one on-ramp per target cell is supported by this simulator."
                )
            self._ramps_by_cell[target_index] = ramp

    def _resolve_target_index(self, target: Union[int, str]) -> int:
        if isinstance(target, int):
            if not 0 <= target < len(self.cells):
                raise IndexError("On-ramp target cell index out of range.")
            return target
        if target not in self._cell_index:
            raise KeyError(f"Unknown cell name '{target}'.")
        return self._cell_index[target]

    def run(self, steps: int) -> SimulationResult:
        """Run the CTM simulation for the requested number of time steps."""

        if steps <= 0:
            raise ValueError("steps must be a positive integer.")

        densities = [cell.initial_density_veh_per_km_per_lane for cell in self.cells]
        density_history: Dict[str, List[float]] = {
            cell.name: [density] for cell, density in zip(self.cells, densities)
        }
        flow_history: Dict[str, List[float]] = {cell.name: [] for cell in self.cells}
        ramp_queue_history: Dict[str, List[float]] = {
            ramp.name: [ramp.queue_veh] for ramp in self.on_ramps
        }
        ramp_flow_history: Dict[str, List[float]] = {ramp.name: [] for ramp in self.on_ramps}

        for step in range(steps):
            sending = self._compute_sending(densities)
            receiving = self._compute_receiving(densities)

            ramp_potentials = self._update_ramp_queues(step)
            ramp_flows_step: Dict[str, float] = {ramp.name: 0.0 for ramp in self.on_ramps}

            inflows = [0.0] * len(self.cells)
            outflows = [0.0] * len(self.cells)

            upstream_demand = _get_profile_value(self.upstream_profile, step, self.dt)
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

            # Flows into each cell from upstream (including boundary to first cell).
            for idx in range(len(self.cells)):
                supply = receiving[idx]
                mainline_demand = upstream_demand if idx == 0 else sending[idx - 1]
                ramp_flow = 0.0

                ramp = self._ramps_by_cell.get(idx)
                if ramp is not None:
                    potential = ramp_potentials[ramp.name]
                    main_flow, ramp_flow = self._merge_flows(
                        mainline_demand,
                        potential,
                        supply,
                        ramp.mainline_priority,
                    )
                    ramp.queue_veh = max(0.0, ramp.queue_veh - ramp_flow * self.dt)
                    ramp_flows_step[ramp.name] = ramp_flow
                else:
                    main_flow = min(mainline_demand, supply)

                inflow = main_flow + ramp_flow
                inflows[idx] += inflow

                if idx > 0:
                    outflows[idx - 1] = main_flow

            # Flow out of the last cell towards downstream boundary.
            last_idx = len(self.cells) - 1
            outflow_last = min(sending[last_idx], downstream_supply)
            outflows[last_idx] = outflow_last
            flow_history[self.cells[last_idx].name].append(outflow_last)

            for idx in range(len(self.cells) - 1):
                flow_history[self.cells[idx].name].append(outflows[idx])

            for ramp in self.on_ramps:
                ramp_flow_history[ramp.name].append(ramp_flows_step[ramp.name])
                ramp_queue_history[ramp.name].append(ramp.queue_veh)

            # Update densities.
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

        return SimulationResult(
            densities=density_history,
            flows=flow_history,
            ramp_queues=ramp_queue_history,
            ramp_flows=ramp_flow_history,
            time_step_hours=self.dt,
        )

    def _compute_sending(self, densities: Sequence[float]) -> List[float]:
        sending = []
        for density, cell in zip(densities, self.cells):
            total_capacity = cell.capacity_veh_per_hour_per_lane * cell.lanes
            demand = cell.free_flow_speed_kmh * density * cell.lanes
            sending.append(min(demand, total_capacity))
        return sending

    def _compute_receiving(self, densities: Sequence[float]) -> List[float]:
        receiving = []
        for density, cell in zip(densities, self.cells):
            total_capacity = cell.capacity_veh_per_hour_per_lane * cell.lanes
            remaining_density = max(0.0, cell.jam_density_veh_per_km_per_lane - density)
            supply = cell.congestion_wave_speed_kmh * remaining_density * cell.lanes
            receiving.append(min(supply, total_capacity))
        return receiving

    def _update_ramp_queues(self, step: int) -> Dict[str, float]:
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
                additional_main = remaining_supply * (remaining_main_demand / total_remaining_demand)
                additional_ramp = remaining_supply - additional_main
                main_flow += min(additional_main, remaining_main_demand)
                ramp_flow += min(additional_ramp, remaining_ramp_demand)

        return main_flow, ramp_flow

    def _update_density(
        self,
        density: float,
        inflow: float,
        outflow: float,
        cell: CellConfig,
    ) -> float:
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
    """Utility to create a list of :class:`CellConfig` objects quickly.

    Parameters
    ----------
    num_cells:
        Number of mainline cells to create.
    cell_length_km:
        Length of each cell in kilometres.
    lanes:
        Either a single integer applied to every cell, or a sequence with one
        entry per cell to allow lane drops/additions.
    free_flow_speed_kmh, congestion_wave_speed_kmh,
    capacity_veh_per_hour_per_lane, jam_density_veh_per_km_per_lane:
        Standard CTM parameters.  Scalars apply to all cells.
    initial_density_veh_per_km_per_lane:
        Optional scalar or sequence describing the starting density per lane.
    """

    if num_cells <= 0:
        raise ValueError("num_cells must be positive.")

    if isinstance(lanes, int):
        lane_profile = [lanes] * num_cells
    else:
        lane_profile = list(lanes)
        if len(lane_profile) != num_cells:
            raise ValueError("lanes sequence must match num_cells.")

    if isinstance(initial_density_veh_per_km_per_lane, (int, float)):
        density_profile = [float(initial_density_veh_per_km_per_lane)] * num_cells
    else:
        density_profile = [float(value) for value in initial_density_veh_per_km_per_lane]
        if len(density_profile) != num_cells:
            raise ValueError("initial_density sequence must match num_cells.")

    cells: List[CellConfig] = []
    for idx in range(num_cells):
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
