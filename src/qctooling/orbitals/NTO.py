import numpy as np
import numpy.typing as npt
from typing import Tuple, List
from ..classes import Wfn, Excited_State

def generate_NTOs(wfn: Wfn, state: Excited_State) -> Wfn:
    T_mats = state.get_MO_Tdens(wfn)
    C_nto = np.zeros((len(T_mats), *wfn.c.shape[1:]))
    E_nto = np.zeros((len(T_mats), *wfn.e.shape[1:]))
    O_nto = np.zeros(E_nto.shape)
    S_nto = np.zeros(E_nto.shape, dtype=np.bool)
    for i, t in enumerate(T_mats):
        c = (wfn.c[i] if wfn.c.shape[0] >= i else wfn.c[0])
        n_occ = np.sum((wfn.o[i]/wfn.o[i,0] if wfn.c.shape[0] >= i else wfn.o[0]/wfn.o[0,0]) != 0)
        U, s, Vt = np.linalg.svd(t, full_matrices=False)
        C_nto[i, n_occ-1::-1] = (U.T @ c[:n_occ])
        C_nto[i, n_occ:2*n_occ] = (Vt @ c[n_occ:])
        E_nto[i, n_occ-1::-1] = E_nto[i, n_occ:2*n_occ] = np.square(s)
        O_nto[i, :n_occ] = (1 if O_nto.shape[0] == 2 else 2)
        S_nto[i, :] = (0 if i == 0 else 1)
        assert np.all(np.isclose(C_nto[i, :2*n_occ] @ wfn.S @ C_nto[i, :2*n_occ].T, np.eye(2*n_occ))), f"non orthogonal NTOs detected"

    return Wfn(wfn.basis, wfn.xyz, C_nto, O_nto, E_nto, S_nto, np.full(E_nto.shape, "1a"))