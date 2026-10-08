from typing import TypeVar
import numpy as np

Numeric = TypeVar("Numeric", float, np.ndarray)

def eh2ev(eh: Numeric) -> Numeric:
    return eh * 27.2114


def au2ang(au: Numeric) -> Numeric:
    return au * 0.529177

def ang2au(au: Numeric) -> Numeric:
    return au / 0.529177