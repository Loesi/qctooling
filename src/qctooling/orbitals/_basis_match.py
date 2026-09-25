
from ..classes import Wfn

mismatchingBasisError = Exception(f"mismatching Basis between two wfns required to share a basis")

def checkBasisMatch(wfn1: Wfn, wfn2: Wfn):
    for b1, b2 in zip(wfn1.basis, wfn2.basis):
        if b1 != b2:
            raise mismatchingBasisError