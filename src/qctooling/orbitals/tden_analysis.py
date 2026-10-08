"""Transition-density analysis using Löwdin-partitioned omega contributions.

For each AO transition density D, the AO contribution matrix is computed as
Omega = (S^(1/2) D S^(1/2)) elementwise-squared, then summed into atom and
fragment blocks. This is the Löwdin-partitioned form (formula 2), not the
formulas based on (DS)*(SD) or the symmetrized D/SDS expression.

Method reference: F. Plasser and H. Lischka, J. Chem. Theory Comput. 8,
2777-2789 (2012), DOI: 10.1021/ct300307c.
"""

import numpy as np
import numpy.typing as npt
from typing import Tuple, List, Callable, Sequence
from ..classes import Wfn, Excited_State, Xyz

def tden_analysis_AO(wfn: Wfn, states: Sequence[Excited_State]) -> npt.NDArray:
    """Build AO omega matrices using Löwdin partitioning (formula 2)."""
    omegas = np.zeros(shape=(len(states), states[0].n_spin_states(), wfn.n_orb, wfn.n_orb), dtype=np.float64)
    for i, spin_states in enumerate(states):
        for j, tdens in enumerate(spin_states.get_AO_Tdens(wfn)):
            omegas[i,j] = np.square(wfn.sqrtS @ tdens @ wfn.sqrtS)

    return omegas

def tden_analysis(
        wfn: Wfn,
        states: Sequence[Excited_State],
        frag_func: Callable[[Xyz, str], List[List[int]]],
        cut_str: str = "",
        ) -> npt.NDArray:
    """Calculate fragment-resolved transition-density contributions.

    Uses Löwdin partitioning: See Plasser and Lischka,
    JCTC 8, 2777-2789 (2012), DOI: 10.1021/ct300307c.
    """
    print("Transition-density analysis: Löwdin partitioning: "
          "Plasser and Lischka, JCTC 8, 2777-2789 (2012), DOI: 10.1021/ct300307c")
    omegas = tden_analysis_AO(wfn, states)
    ao_atom_idx = wfn.basis_str("{i}").astype(np.int16)
    ao_atom_mapping = np.zeros((len(ao_atom_idx), len(wfn.xyz.elements)))
    ao_atom_mapping[np.arange(len(ao_atom_idx)), ao_atom_idx] = 1
    omegas = np.einsum(
        "af, ...ab,bg->...fg",
        ao_atom_mapping,
        omegas,
        ao_atom_mapping,
        optimize=True,
    )

    fragments = frag_func(wfn.xyz, cut_str)
    P = np.zeros((len(fragments), omegas.shape[-1]))
    for f, atoms in enumerate(fragments):
        P[f, atoms] = 1
    ret = np.einsum("fa, ...ab,gb->...fg", P, omegas, P)
    return ret
