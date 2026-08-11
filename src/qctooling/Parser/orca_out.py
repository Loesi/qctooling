import pathlib, io, logging
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Tuple, List, Any, Dict, Callable
from pydantic import BaseModel

from ..classes import Basis_grp, l2orb, Wfn, Xyz, State, Abs
from ..util import eh2ev
from .parserBase import ParserBase

# idk about a loewdin parser as i can generate loewdin form molden file
class OrcaSection(str, Enum):
    STATES = "States"
    ABSORPTION = "ABSORPTION"
    NEVPT2 = "NEVPT2"
    SINGLE_POINT_ENERGY = "SinglePointEnergy"
    GIBBS_FREE_ENERGY = "GibbsFreeEnergy"


detectStrings: Dict[OrcaSection, List[str]] = {
    OrcaSection.STATES: [
        "CAS-SCF STATES FOR BLOCK",
        "TD-DFT/TDA EXCITED STATES",
    ],
    OrcaSection.ABSORPTION: [
        "ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS",
    ],
    OrcaSection.NEVPT2: [
        "CASSCF (NEVPT2 diagonal energies) UV, CD spectra and dipole moments",
    ],
    OrcaSection.SINGLE_POINT_ENERGY: [
        "FINAL SINGLE POINT ENERGY",
    ],
    OrcaSection.GIBBS_FREE_ENERGY: [
        "Final Gibbs free energy",
    ]
}

def parse_States(file: io.TextIOWrapper, header: str, logger: logging.Logger) -> List[State]:
    states: List[State] = []

    if "CAS-SCF STATES FOR BLOCK" in header:
        logger.info("Identified CAS-SCF State Section")
        multi = int(header.split()[6])
        n_roots = int(header.split()[8])

        file.readline()
        file.readline()

        for i in range (n_roots):
            line = file.readline()
            idx = int(line.split()[1])
            assert idx == 1, f"lost track of roots while parsing"
            energy = eh2ev(float(line.split()[3]))
            contri = []
            line = file.readline()
            while not line.startswith("ROOT"):
                contri.append((float(line.split()[0]), line.split()[-1]))
                line = file.readline()
            states.append(State(idx, multi, "A", energy, contri))

    elif "TD-DFT/TDA EXCITED STATES" in header:
        logger.info("Identified TDDFT State Section")
        line = file.readline()
        while not line.startswith("STATE"):
            if line == "":
                raise ValueError("Unexpected end of file while parsing states")
            line = file.readline()

        while line.startswith("STATE"):
            idx = int(line.split()[1][:-1])
            energy = float(line.split()[5])
            multi = int(line.split()[-1])
            line = file.readline()
            contri = []
            while line != "\n":
                if line == "":
                    raise ValueError("Unexpected end of file while parsing state contributions")
                contri.append((float(line.split()[4]), line.split(":")[0].strip()))
                logger.debug(line)
                line = file.readline()
            states.append(State(idx, multi, "A", energy, contri))
            line = file.readline()
        
    logger.info(f"Read {len(states)} states")
    return states

def parse_Absorption(file: io.TextIOWrapper, header: str, logger: logging.Logger) -> Abs:
    logger.info("Identified Absorption section")
    line = header
    for i in range(5):
        line = file.readline()

    don, acc, energy, fosz = [], [] ,[], []
    while line != "\n":
        vals = line.split()
        don.append(vals[0])
        acc.append(vals[2])
        energy.append(float(vals[3]))
        fosz.append(float(vals[6]))

        line = file.readline()

    logger.info(f"Read {len(don)} transitions")
    return Abs(np.array(don, dtype=np.str_), np.array(acc, dtype=np.str_), np.array(energy, dtype=np.float64), np.array(fosz, dtype=np.float64))

def parse_NEVPT2(file: io.TextIOWrapper, header: str, logger: logging.Logger) -> Abs:
    for l in file:
        if l.startswith(detectStrings[OrcaSection.ABSORPTION][0]):
            return parse_Absorption(file, l, logger)

    raise ValueError("Misformed file")


sectionParseFunc: Dict[OrcaSection, Callable[[io.TextIOWrapper, str, logging.Logger], Any]] = {
    OrcaSection.STATES: parse_States,
    OrcaSection.ABSORPTION: parse_Absorption,
    OrcaSection.NEVPT2: parse_NEVPT2,
    OrcaSection.SINGLE_POINT_ENERGY: lambda f, l, log: (
        log.info(f"Identified SinglePointEnergy"),
        float(l.split()[-1])
        )[-1],
    OrcaSection.GIBBS_FREE_ENERGY: lambda f, l, log: (
        log.info(f"Identified GibbsFreeEnergy"),
        float(l.split()[-1])
        )[-1],
}

# verify setup
assert set(detectStrings.keys()) >= set(OrcaSection), f"Missing detection Strings for Sections: {set(OrcaSection) - set(detectStrings.keys())}"


def parseOutFile(path: pathlib.Path, sections: List[OrcaSection], logger: logging.Logger) -> dict[OrcaSection, Any]:
    output = {}
    detect_calls: Dict[str, OrcaSection] = {s: sec for sec in sections for s in detectStrings[sec]}
    check_found = np.zeros(shape=(len(sections)), dtype=bool)
    with path.open('r') as f:
        for l in f:
            striped_line = l.lstrip()
            for s, sec in detect_calls.items():
                if striped_line.startswith(s):
                    output[sec] = sectionParseFunc[sec](f, striped_line, logger)
                    check_found[sections.index(sec)] = True

    if np.any(~check_found):
        missing = [s for i, s in enumerate(sections) if not check_found[i]]
        raise ValueError(f"Sections {missing} not found in {path}")
    return output

class OrcaParser(ParserBase):
    sections: List[OrcaSection]

    def parse(self):
        candidates = [self.path / (self.baseName + suf) for suf in [".out", ".oout"]]
        for c in candidates:
            if c.exists():
                return parseOutFile(c, self.sections, self.logger)

        raise FileNotFoundError(
            f"Either .out nor .oout found with basename {self.baseName} in {self.path.resolve()}"
        )
