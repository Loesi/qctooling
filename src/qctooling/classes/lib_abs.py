from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from typing import Literal, List


@dataclass(frozen=True)
class State:
    idx: int
    multiplicity: int
    irrep: str
    energy: float
    state_info: List[tuple[float, str]] # List for different Configuration / donor-acceptor pairs; float is contribution str is info

    def __str__(self) -> str:
        return f"{self.idx}-{self.multiplicity}{self.irrep}"


@dataclass(frozen=True)
class Abs:
    donor: npt.NDArray[np.str_]
    acceptor: npt.NDArray[np.str_]
    energy: npt.NDArray[np.float64] 
    fosz: npt.NDArray[np.float64]