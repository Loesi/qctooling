from dataclasses import dataclass
from functools import cached_property
import numpy as np
import numpy.typing as npt
from typing import Literal, List, Union, Tuple

from .lib_xyz import Xyz
from ..util import elements
from ..intor import overlap, kinetic, nuclear

l2orb = ["s", "p", "d", "f", "g"]
cs = {
    "p": ["y", "z", "x"],
    "d": ["xy", "yz", "z2", "xz", "x2-y2"],
    "f": ["y(3x2-y2)", "xyz", "yz2", "z3", "xz2", "z(x2-y2)", "x(x2-3y2)"]
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
    def _basis_str(self) -> List[Tuple[str, str, int, str, str]]:
        vals = []
        for b in self.basis:
            match b.l:
                case 0:
                    vals += [
                        (b.atom_idx, self.xyz.elements[b.atom_idx], b.n, "s", "")
                        ]
                case 1:
                    vals += [
                        (b.atom_idx, self.xyz.elements[b.atom_idx], b.n, "p", m)
                        for m in ["x", "y", "z"]
                        ]
                case 2:
                    vals += [
                        (b.atom_idx, self.xyz.elements[b.atom_idx], b.n, "d", m)
                        for m in ["z2", "xz", "yz", "x2y2", "xy"]
                        ]
                case 3:
                    vals += [
                        (b.atom_idx, self.xyz.elements[b.atom_idx], b.n, "f", m)
                        for m in ["z3", "xz2", "yz2", "z(x2-y2)", "xyz", "x(x2-3y2)", "y(3x2-y2)"]
                        ]
                case 4:
                    raise NotImplementedError("Not yet added order for g orbitals")
        return vals

    def basis_str(self, fmt: str = "{idx:03d}{element}-{n}{l}{m}") -> npt.NDArray[np.str_]:
        vals = self._basis_str
        try:
            strs = [
                fmt.format(idx=idx, element=element, l=l, m=m)
                for idx, element, n, l, m in vals
            ]
        except (KeyError, IndexError) as e:
            raise ValueError(
                f"Invalid format string {format!r}: unknown field {e}. "
                f"Valid fields: idx, element, l, m"
            ) from e
        return np.array(strs, dtype=np.str_)

    @cached_property
    def S_matrix(self) -> npt.NDArray[np.float64]:
        return overlap(self.xyz.coordinates, [b.asTuple for b in self.basis])

    @cached_property
    def S_matrix_root(self) -> npt.NDArray[np.float64]:
        s, U = np.linalg.eigh(self.S_matrix)
        return U @ np.diag(1.0/np.sqrt(s)) @ U.T

    @cached_property
    def density(self) -> Union[npt.NDArray[np.float64], Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
        C_occ = self.O[:,:,None] * self.C
        if self.C.shape[0] != 1:
            return (C_occ[0].T @ C_occ[0], C_occ[1].T @ C_occ[1])
        else:
            return (C_occ[0]/2).T @ (C_occ[0]/2)
        
    @cached_property
    def charge(self) -> int:
        n_e = int(np.sum(self.O))
        n_p = sum(elements.index(e) for e in self.xyz.elements)
        return n_e - n_p

    @cached_property
    def spin(self) -> int:
        if self.O.shape[0] == 1:
            return 0
        return int(np.abs(np.sum(self.O[0]) - np.sum(self.O[1]))
)