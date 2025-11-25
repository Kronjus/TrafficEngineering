from dataclasses import dataclass, field

from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

NumberLike = Union[float, int]
Profile = Union[
    NumberLike,
    Sequence[NumberLike],
    Callable[..., NumberLike],
]


@dataclass(frozen=True)
class CellConfig:
    name: str
    cell_nr: int
    length: float
    lanes: int
    dt: float
    q_max: float
    v_f: float
    rho_jam: float

    @property
    def rho_crit(self) -> float:
        return self.q_max / self.v_f


@dataclass(frozen=True)
class OnRampConfig:
    target_cell: Union[int, str]
    arrival_rate_profile: Profile
    name: Optional[str] = None


class CTMSimulation:
    def __init__(self, cells: Sequence[CellConfig], time_step_hours: float, upstream_demand_profile: Profile,
                 on_ramps: Optional[Sequence[OnRampConfig]] = None) -> None:
        self.cells: List[CellConfig] = list(cells)
        self.dt = float(time_step_hours)
        self.upstream_profile = upstream_demand_profile

        self._cell_index: Dict[str, int] = {cell.name: idx for idx, cell in enumerate(self.cells)}

        self.onr
