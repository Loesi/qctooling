from dataclasses import dataclass
import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class Xyz:
    elements: npt.NDArray[np.str_]
    coordinates: npt.NDArray[np.float64]