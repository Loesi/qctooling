import warnings
import numpy as np
import numpy.typing as npt
from typing import Tuple, Literal
from ..classes import Wfn, elements

def loewDensity(wfn: Wfn) -> npt.NDArray:
    return np.stack([np.square(np.abs(wfn.c[i] @ wfn.sqrtS)) for i in range(wfn.c.shape[0])], axis=0)

def loewAO_Occ_by_spin(wfn: Wfn) -> np.ndarray[Tuple[Literal[2],int]]:
    assert wfn.c.shape[0] == 2
    return np.sum(wfn.o[:,:,None] * loewDensity(wfn), axis=(1))

def loewAO_Occ(wfn: Wfn) -> npt.NDArray[np.float64]:
    return np.sum(wfn.o[:,:,None] * loewDensity(wfn), axis=(0,1))

def _loewAtomAssing(wfn) -> npt.NDArray:
    return np.repeat(*np.array([(b.atom_idx, b.n_orb) for b in wfn.basis]).T)

def loewCharges(wfn:Wfn) -> npt.NDArray:
    atom_assign = _loewAtomAssing(wfn)
    ne = np.zeros(wfn.xyz.elements.shape, dtype=np.float64)
    np.add.at(ne, atom_assign, loewAO_Occ(wfn))
    return np.array([elements.index(e) for e in wfn.xyz.elements]) - ne

def loewSpinPop(wfn:Wfn) -> npt.NDArray:
    if wfn.c.shape[0] == 1:
        warnings.warn(f"{wfn.name} is closed shell and has no spinPopulation")
        return np.zeros(wfn.xyz.elements.shape, dtype=np.float64)
    atom_assign = _loewAtomAssing(wfn)
    alpha, beta = loewAO_Occ_by_spin(wfn)
    ne_a, ne_b = np.zeros(wfn.xyz.elements.shape, dtype=np.float64), np.zeros(wfn.xyz.elements.shape, dtype=np.float64)
    np.add.at(ne_a, atom_assign, alpha)
    np.add.at(ne_b, atom_assign, beta)
    return ne_b - ne_a
    
    