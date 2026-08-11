from typing import TypeVar, overload
import numpy as np

Numeric = TypeVar("Numeric", float, np.ndarray)

def eh2ev(eh: Numeric) -> Numeric:
    return eh * 27.2114