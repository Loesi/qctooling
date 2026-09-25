import numpy as np
import numpy.typing as npt
from typing import Tuple
from ..classes import Wfn
from ._basis_match import checkBasisMatch

def _spin_NDO(C1_occ:npt.NDArray, C2_occ:npt.NDArray, S:npt.NDArray, sqrtS:npt.NDArray, sqrtS_inv:npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
    P1 = C1_occ.T @ C1_occ
    P2 = C2_occ.T @ C2_occ
    dP = P1 - P2

    D = sqrtS @ dP @ sqrtS
    eigen_vals, eigen_vecs = np.linalg.eigh(D)
    order = np.argsort(np.abs(eigen_vals))[::-1]
    C = (sqrtS_inv @ eigen_vecs).T
    assert np.all(np.isclose(x := C @ S @ C.T, np.eye(P1.shape[0]))), f"something wrong, this should be 1: {x}"
    return C[order][None,:,:], eigen_vals[order][None,:]


def generate_NDOs(wfn1: Wfn, wfn2: Wfn) -> Wfn:
    checkBasisMatch(wfn1, wfn2)

    match wfn1.c.shape[0], wfn2.c.shape[0]:
        case 1,1:
            n_occ = ((np.sum(wfn1.o, axis=1)/2).astype(np.int32), (np.sum(wfn2.o, axis=1)/2).astype(np.int32))
            C, E = _spin_NDO(wfn1.o[0,:n_occ[0][0]], wfn2.c[0,:n_occ[1][0]], wfn1.S, wfn1.sqrtS, np.linalg.inv(wfn1.sqrtS_inv))
            return Wfn(wfn1.basis, wfn1.xyz, C, E, np.full(wfn1.o.shape, 2), wfn1.s, wfn1.i)

        case 1,2:
            n_occ = ((np.sum(wfn1.o, axis=1)/2).astype(np.int32), np.sum(wfn2.o, axis=1).astype(np.int32))
            C, E = (np.concat(arrs, axis=0) for arrs in zip(*[_spin_NDO(wfn1.c[0,:n_occ[0][0]], wfn2.c[i,:n_occ[1][i]], wfn1.S, wfn1.sqrtS, wfn1.sqrtS_inv) for i in range(2)]))
            return Wfn(wfn1.basis, wfn1.xyz, C, E, np.ones(wfn2.o.shape), wfn2.s, wfn2.i)

        case 2,1:
            n_occ = (np.sum(wfn1.o, axis=1).astype(np.int32), (np.sum(wfn2.o, axis=1)/2).astype(np.int32))
            C, E = (np.concat(arrs, axis=0) for arrs in zip(*[_spin_NDO(wfn1.c[i,:n_occ[0][i]], wfn2.c[0,:n_occ[1][0]], wfn1.S, wfn1.sqrtS, wfn1.sqrtS_inv) for i in range(2)]))
            return Wfn(wfn1.basis, wfn1.xyz, C, E, np.ones(wfn1.o.shape), wfn1.s, wfn1.i)

        case 2,2:
            n_occ = (np.sum(wfn1.o, axis=1).astype(np.int32), np.sum(wfn2.o, axis=1).astype(np.int32))
            C, E = (np.concat(arrs, axis=0) for arrs in zip(*[_spin_NDO(wfn1.c[i,:n_occ[0][i]], wfn2.c[i,:n_occ[1][i]], wfn1.S, wfn1.sqrtS, wfn1.sqrtS_inv) for i in range(2)]))
            return Wfn(wfn1.basis, wfn1.xyz, C, E, np.ones(wfn1.o.shape), wfn1.s, wfn1.i)
            
    raise ValueError("Unrecognize coefficient shape: {wfn1.c.shape} & {wfn2.c.shape}")