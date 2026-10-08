import numpy as np
import numpy.typing as npt
from typing import Tuple, List, Callable, Sequence
from ..classes import Wfn, Excited_State, Xyz

import time

def tden_analysis_AO(wfn: Wfn, states: Sequence[Excited_State]) -> npt.NDArray:
    omegas = np.zeros(shape=(len(states), states[0].n_spin_states(), wfn.n_orb, wfn.n_orb), dtype=np.float64)
    for i, spin_states in enumerate(states):
        for j, tdens in enumerate(spin_states.get_AO_Tdens(wfn)):
            omegas[i,j] = np.square(wfn.sqrtS @ tdens @ wfn.sqrtS)

    return omegas

#TODO: to slow prallelise later, no time now
def tden_analysis(
        wfn: Wfn,
        states: Sequence[Excited_State],
        frag_func: Callable[[Xyz, str], List[List[int]]],
        cut_str: str = "",
        ) -> npt.NDArray:

    start = time.time()
    
    omegas = tden_analysis_AO(wfn,states)
    print(time.time() - start)

    ao_atom_idx = wfn.basis_str("{i}").astype(np.int16)
    ao_atom_mapping = np.zeros((len(ao_atom_idx), len(wfn.xyz.elements)))
    ao_atom_mapping[np.arange(len(ao_atom_idx)),ao_atom_idx] = 1
    print(time.time() - start)
    # omegas = ao_atom_mapping.T @ omegas @ ao_atom_mapping
    omegas = np.einsum("af, ...ab,bg->...fg", ao_atom_mapping, omegas, ao_atom_mapping, optimize = True)
    print(time.time() - start)

    fragments = frag_func(wfn.xyz, cut_str)
    P = np.zeros((len(fragments), omegas.shape[-1]))
    for f, aos in enumerate(fragments):
        P[f, aos] = 1
    print(time.time() - start)
    ret = np.einsum("fa, ...ab,gb->...fg", P, omegas, P)
    print(time.time() - start)
    return ret
