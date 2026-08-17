from dataclasses import dataclass
from functools import cached_property
import numpy as np
import numpy.typing as npt
from typing import Literal, List, Union, Tuple

from .lib_xyz import Xyz
from ..intor import overlap, kinetic, nuclear

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

    @property
    def asTuple(self) -> Tuple[int,int,npt.ArrayLike,npt.ArrayLike]:
        return (self.atom_idx, self.l, self.alpha, self.coeff[:,None])

@dataclass(frozen=True)
class Wfn:
    basis: List[Basis_grp] # as unnormalized coeffs (N_cont * d_k)
    xyz: Xyz
    C: npt.NDArray[np.float64]
    O: npt.NDArray[np.float64]
    E: npt.NDArray[np.float64]
    S: npt.NDArray[np.bool]
    I: npt.NDArray[np.str_]

    @cached_property
    def basis_str(self):
        return [f"{self.xyz.elements[b.atom_idx]}-{b}" for b in self.basis]

    @cached_property
    def S_matrix(self) -> npt.NDArray[np.float64]:
        return overlap(self.xyz.coordinates, [b.asTuple for b in self.basis])

    @cached_property
    def S_matrix_root(self) -> npt.NDArray[np.float64]:
        s, U = np.linalg.eigh(self.S_matrix)
        return U @ np.diag(1.0/np.sqrt(s)) @ U.T

    @cached_property
    def density(self) -> Union[npt.NDArray[np.float64], Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
        if self.C.shape[0] != 1:
            return (self.C[0].T @ self.C[0], self.C[1].T @ self.C[1])
        else:
            return self.C[0].T @ self.C[0]
