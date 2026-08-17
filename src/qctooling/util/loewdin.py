import numpy as np
from ..classes import Wfn

def LoewDensity(wfn: Wfn):
    U, s, Vh = np.linalg.svd(wfn.S_matrix, hermitian=True)

    S_root = U @ np.sqrt(s) @ Vh