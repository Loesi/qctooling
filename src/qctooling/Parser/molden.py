
import pathlib, io
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from typing import Literal, Tuple, List
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

def read_gto(file: io.TextIOWrapper, is_norm: bool, is_dflt_order: bool) -> Tuple[List[Basis_grp], str]:
    basis: list[Basis_grp] = []
    run: bool = True
    line = ""
    while run:
        line = file.readline()
        if line.startswith("["):
            run = False
            continue

        atom_idx = int(line.split()[0])
        shell_counter = np.arange(5)
        for l in file:
            if l == "\n":
                break
            vals = l.split()
            l, n_prim = l2orb.index(vals[0]), int(vals[1])
            alpha, coeff = np.empty(shape=(n_prim)), np.empty(shape=(n_prim))
            for i in range(n_prim):
                vals = file.readline().split()
                alpha[i], coeff[i] = vals

            shell_counter[l] += 1
            basis.append(Basis_grp(atom_idx, shell_counter[l], l,alpha, coeff ))

    tags = []
    while not any(line.startswith(k) for k in keywords):
        tags.append(line[1:3])
        line = file.readline()

    print("Still need logic go implement cartesian basis functions. though i dont think my programs use them")
    return basis, line

def read_mo(
    file: io.TextIOWrapper, n_ao: int
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.bool], npt.NDArray[np.str_]]:

    coeffs = np.empty(shape=(2, n_ao, n_ao), dtype=np.float64)
    occ = np.empty(shape=(2, n_ao), dtype=np.float64)
    energy = np.empty(shape=(2, n_ao), dtype=np.float64)
    spin = np.empty(shape=(2, n_ao), dtype=np.bool)
    irrep = np.empty(shape=(2, n_ao), dtype=np.str_)

    i = 0
    while i < 2:
        line = file.readline()
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

        if line == "\n":
            coeffs = coeffs[0]
            occ = occ[0]
            energy = energy[0]
            spin = spin[0]
            irrep = irrep[0]

            i = 2

    return coeffs, occ, energy, spin, irrep


def read_molden(path: pathlib.Path, program: Literal['orca', 'multiwfn']) -> Tuple[Wfn, Xyz]:
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
        basis, line = read_gto(f, is_norm_coeffs, is_dflt_order)
        n_AO = sum(b.n_orb for b in basis)
        coeffs, occ, energy, spin, irrep = read_mo(f, n_AO)

        wfn = Wfn(basis, coeffs, occ, energy, spin, irrep)

        return wfn, xyz
    
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
            