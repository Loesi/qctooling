
import pathlib, io, logging, time
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from typing import Literal, Tuple, List, Callable, Dict
from pydantic import BaseModel

from ..classes import Basis_grp, l2orb, Wfn, Xyz
from .parserBase import ParserBase

keywords = ["[ATOMS]", "[GTO]", "[MO]", "[Pseudo]"]

def read_atoms(file: io.TextIOWrapper, log: logging.Logger) -> Tuple[Xyz, str]:
    elements = []
    coordiantes = []
    for line in file:
        if any(line.startswith(k) for k in keywords):
            log.info(f"Detected {len(elements)} atoms")
            return Xyz(np.array(elements), np.array(coordiantes)), line
        vals = line.split()
        elements.append(vals[0])
        coordiantes.append([float(c) for c in vals[3:]])
                
    raise ValueError("End of Atoms section not found")

def read_pseudo(file: io.TextIOWrapper, log: logging.Logger) -> str:
    line = file.readline()
    while not line.startswith("["):
        try:
            atom, a_idx, valenz = line.split()
            log.warning(f"Detected use of ECP for {atom}{a_idx}. Only {valenz} electrons in SCF")
        except:
            continue
    return line

normalization_funcs: Dict[str, Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64], int], npt.NDArray[np.float64]]] = {
    "orca": lambda a, c, l: (
        c / (np.pow(2 * a / np.pi, 3/4) * np.pow(4 * a, l/2)) if l !=4 else
        c * np.sqrt(3) / (np.pow(2 * a / np.pi, 3/4) * np.pow(4 * a, l/2))
    ),
    "multiwfn": lambda a, c, l: c,
}

def read_gto(file: io.TextIOWrapper, program: str, log: logging.Logger) -> Tuple[List[Basis_grp], str]:
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
    log.info(f"Detected {len(basis)} basis grps with {sum([b.n_orb for b in basis])} to basis functions")

    tags = []
    while not any(line.startswith(k) for k in keywords):
        tags.append(line[1:3])
        line = file.readline()
    log.info(f"Detected gto tags {tags}")
    log.warning(f"found gto tags but tags are not applied")

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
            coeffs = coeffs[[0]]
            occ = occ[[0]]
            energy = energy[[0]]
            spin = spin[[0]]
            irrep = irrep[[0]]

            i = 2

    return coeffs, occ, energy, spin, irrep

def post_correction(wfn: Wfn, program: str, log: logging.Logger) -> Wfn:
    match program:
        case "orca":
            csum = np.cumsum([b.n_orb for b in wfn.basis])
            for i, b in enumerate(wfn.basis):
                if b.l < 3:
                    continue
                wfn.C[:, :, csum[i-1]+5 : csum[i]] *= -1
            log.info("Applied post_correction for orca: phase flip for |m| > 2")

        case "multiwfn":
            pass
            log.info("Applied post_correction for mulitwfn: no correction")

        case _:
            raise ValueError(f"no post_correction defined for program {program}")

    return wfn


def read_molden(path: pathlib.Path, program: Literal['orca', 'multiwfn'], log: logging.Logger) -> Wfn:
    """
    Parse basis fcts und LCAO coeffs from molden.
    program: to differentiate if normalized (like orca) or unnormalized (like multiwfn) primitive coefficients are expected
    """
    start = time.time()
    with path.open("r") as f:
        line = f.readline()
        while not line.startswith("[Atoms]"):
            line = f.readline()

        xyz, line = read_atoms(f, log)
        if line.startswith("[PSEUDO]"):
            line = read_pseudo(f, log)
        basis, line = read_gto(f, program, log)
        n_AO = sum(b.n_orb for b in basis)
        coeffs, occ, energy, spin, irrep = read_mo(f, n_AO)

    wfn = Wfn(basis, xyz, coeffs, occ, energy, spin, irrep)
    wfn = post_correction(wfn, program, log)
    log.debug(f"reading of {path.resolve()} took {time.time() - start} s")
    return wfn
    
class MoldenParser(ParserBase):
    program: Literal['orca', 'multiwfn'] = 'orca'

    def parse(self):
        candidates = [self.path / (self.baseName + suf) for suf in [".molden.input", ".molden"]]
        for c in candidates:
            if c.exists():
                with self._local_path(c) as p:
                    return read_molden(p, self.program, self.logger)

        raise FileNotFoundError(
            f"Either .molden.input nor .molden found with basename {self.baseName} in {self.path.resolve()}"
        )
            