from dataclasses import dataclass
from typing import Tuple, Literal
import numpy as np
import numpy.typing as npt

elements = [
    "H"                                                                                                 , "He",
    "Li", "Be"                                                            , "B" , "C" , "N" , "O" , "F" , "Ne",
    "Na", "Mg"                                                            , "Al", "Si", "P" , "S" , "Cl", "Ar",
    "K" , "Ca", "Sc", "Ti", "V" , "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y" , "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I" , "Xe",
    ]

@dataclass(frozen=True)
class Xyz:
    elements: np.ndarray[Tuple[int], np.dtype[np.str_]]
    coordinates: np.ndarray[Tuple[int, Literal[3]], np.dtype[np.float64]]