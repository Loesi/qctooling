from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from typing import Literal, List

l2orb = ["s", "p", "d", "f", "g"]
cs = {
    "p": ["y", "z", "x"],
    "d": ["xy", "yz", "z2", "xz", "x2-y2"],
    "f": ["y(3x2-y2)", "xyz", "yz2", "z3", "xz2", "z(x2-y2), x(x2-3y2)"]
}

@dataclass(frozen=True)
class Basis_grp:
    """
    Stores a basis function, m is used as Azimuthal Order (along Condon-Shortly) and not as magnetic qn
    """
    atom_idx: int
    n: int
    l: int
    alpha: npt.NDArray[np.float64]
    coeff: npt.NDArray[np.float64]
    otype: Literal['Spherical', 'Carthesian'] = 'Spherical'

    def __post_init__(self) -> None:
        assert self.l < self.n, f"Azimuthal quantum number has to be lower than principal quantum number, got {self.l}, {self.n}"
        assert abs(self.l) <= self.n, f"Absolute of magnetic order number has to be lower or equal to the azimuthal quantum number"

    def __str__(self) -> str:
        shell = l2orb[self.l]
        return f"{self.n}{shell}"

    def asCat(self):
        return Basis_grp(self.atom_idx, self.n, self.l, self.alpha, self.coeff, 'Carthesian')

    @property
    def n_orb(self) -> int:
        if self.otype == 'Spherical':
            return 2 * self.l + 1
        elif self.otype == 'Carthesian':
            return int((self.l + 1) * (self.l + 2) / 2)
        else:
            raise ValueError(f"Only 'Spherical' and 'Carthesian' are valid otypes not: {self.otype}")

@dataclass(frozen=True)
class Wfn:
    basis: List[Basis_grp]
    C: npt.NDArray[np.float64]
    O: npt.NDArray[np.float64]
    E: npt.NDArray[np.float64]
    S: npt.NDArray[np.bool]
    I: npt.NDArray[np.str_]

