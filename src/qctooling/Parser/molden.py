
import pathlib, io
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from typing import Literal, Tuple, List, Callable, Dict
from pydantic import BaseModel

from ..classes import Basis_grp, l2orb, Wfn, Xyz
from .parserBase import ParserBase

keywords = ["[ATOMS]", "[GTO]", "[MO]", "[Pseudo]"]

def read_atoms(file: io.TextIOWrapper) -> Tuple[Xyz, str]:
    elements = []
    coordiantes = []
    for line in file:
        if any(line.startswith(k) for k in keywords):
            return Xyz(np.array(elements), np.array(coordiantes)), line
        vals = line.split()
        elements.append(vals[0])
        coordiantes.append([float(c) for c in vals[3:]])
                
    raise ValueError("End of Atoms section not found")

def read_pseudo(file: io.TextIOWrapper):
    return None, file.readline()
    pass

normalization_funcs: Dict[str, Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64], int], npt.NDArray[np.float64]]] = {
    "orca": lambda a, c, l: (
        c / (np.pow(2 * a / np.pi, 3/4) * np.pow(4 * a, l/2)) if l !=4 else
        c * np.sqrt(3) / (np.pow(2 * a / np.pi, 3/4) * np.pow(4 * a, l/2))
    ),
    "multiwfn": lambda a, c, l: c,
}

def read_gto(file: io.TextIOWrapper, program: str) -> Tuple[List[Basis_grp], str]:
    basis: list[Basis_grp] = []
    run: bool = True
    line = ""
    norm_func = normalization_funcs[program]
    while run:
        line = file.readline()
        if line.startswith("["):
            run = False
            continue
        if line.strip() == "":
            continue

        atom_idx = int(line.split()[0])-1
        shell_counter = np.arange(5)
        for l in file:
            if l.strip() == "":
                break
            vals = l.split()
            l, n_prim = l2orb.index(vals[0]), int(vals[1])
            alpha, coeff = np.empty(shape=(n_prim)), np.empty(shape=(n_prim))
            for i in range(n_prim):
                vals = file.readline().split()
                alpha[i], coeff[i] = vals

            shell_counter[l] += 1
            basis.append(Basis_grp(
                atom_idx, shell_counter[l], l, alpha, norm_func(alpha, coeff, l)
            ))

    tags = []
    while not any(line.startswith(k) for k in keywords):
        tags.append(line[1:3])
        line = file.readline()

    print("Still need logic go implement cartesian basis functions. though i dont think my programs use them")
    return basis, line

def read_mo(
    file: io.TextIOWrapper, n_ao: int
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.bool], npt.NDArray[np.str_]]:
    """Symmetry labels (e.g. "1a") stored as full strings, not truncated."""

    coeffs = np.empty(shape=(2, n_ao, n_ao), dtype=np.float64)
    occ = np.empty(shape=(2, n_ao), dtype=np.float64)
    energy = np.empty(shape=(2, n_ao), dtype=np.float64)
    spin = np.empty(shape=(2, n_ao), dtype=np.bool)
    irrep = np.empty(shape=(2, n_ao), dtype="U16")

    i = 0
    line = file.readline()
    while i < 2:
        for j in range(n_ao):
            k = 0
            while k != 4:
                if "Sym=" in line:
                    irrep[i, j] = line.split("=")[-1].strip()
                elif "Ene=" in line:
                    energy[i,j] = line.split("=")[-1].strip()
                elif "Spin=" in line:
                    if line.split("=")[-1].strip() == "Alpha":
                        spin[i,j] = True
                    elif line.split("=")[-1].strip() == "Beta":
                        spin[i,j] = False
                    else:
                        raise ValueError(f"Corrupted molden file. Spin label was {line.split()[-1]} but expected 'Alpha' or 'Beta'.")
                elif "Occup=" in line:
                    occ[i,j] = line.split("=")[-1].strip()

                elif line.split()[0] == "1":
                    k = 4
                    continue

                line = file.readline()
                k += 1

            for l in range(n_ao):
                coeffs[i,j,l] = line.split()[-1]
                line = file.readline()

        i+= 1

        if i == 1 and (line.strip() == "" or line.startswith("[")):
            coeffs = coeffs[0]
            occ = occ[0]
            energy = energy[0]
            spin = spin[0]
            irrep = irrep[0]

            i = 2

    return coeffs, occ, energy, spin, irrep

def post_correction(wfn: Wfn, program: str) -> Wfn:
    match program:
        case "orca":
            csum = np.cumsum([b.n_orb for b in wfn.basis])
            for i, b in enumerate(wfn.basis):
                if b.l < 3:
                    continue
                wfn.C[:, :, csum[i-1]+5 : csum[i]] *= -1

        case _:
            pass

    return wfn


def read_molden(path: pathlib.Path, program: Literal['orca', 'multiwfn']) -> Wfn:
    """
    Parse basis fcts und LCAO coeffs from molden.
    program: to differentiate if normalized (like orca) or unnormalized (like multiwfn) primitive coefficients are expected
    """

    match program:
        case 'orca':
            is_norm_coeffs: bool = True
            is_dflt_order:bool = True
        case 'multiwfn':
            is_norm_coeffs: bool = False
            is_dflt_order:bool = True
        case _:
            raise ValueError(f"settings for program {program} not known, if the primitive coefficients are normalized use 'orca', if they are not use 'multiwfn'")

    with path.open("r") as f:
        line = f.readline()
        while not line.startswith("[Atoms]"):
            line = f.readline()

        xyz, line = read_atoms(f)
        if line.startswith("[PSEUDO]"):
            pseudo, line = read_pseudo(f)
        basis, line = read_gto(f, program)
        n_AO = sum(b.n_orb for b in basis)
        coeffs, occ, energy, spin, irrep = read_mo(f, n_AO)

    wfn = Wfn(basis, xyz, coeffs, occ, energy, spin, irrep)
    wfn = post_correction(wfn, program)
    return wfn
    
class MoldenParser(ParserBase):
    program: Literal['orca', 'multiwfn'] = 'orca'

    def parse(self):
        candidates = [self.path / (self.baseName + suf) for suf in [".molden.input", ".molden"]]
        for c in candidates:
            if c.exists():
                return read_molden(c, self.program)

        raise FileNotFoundError(
            f"Either .molden.input nor .molden found with basename {self.baseName} in {self.path.resolve()}"
        )
            