import struct, pathlib, logging
from typing import List,  Optional
import numpy as np
import numpy.typing as npt

from .parserBase import ParserBase
from ..classes import CIstate
from ..util import catch_errors

@catch_errors
def parse_cis(file_path: pathlib.Path, logger: logging.Logger) -> List[CIstate]:
    with file_path.open("rb") as f:
        info = struct.unpack("i"*9, f.read(4*9))
        n_trans = info[0]
        restricted = True if info[5] == -1 else False
        logger.info(f"Detected {"restricted" if restricted else "unrestricted"} cis data.")
        if restricted:
            offset = (info[1],)
        else:
            offset = (info[1], info[5])

        NDA, NDB = info[2] - info[1] + 1, info[6] - info[5] + 1
        NAA, NAB = info[4] - info[3] + 1, info[8] - info[7] + 1

        nvals_a, nvals_b = NDA * NAA, NDB * NAB

        states: List[CIstate] = []
        for i in range(n_trans):
            _, _, m, _, r, _ = struct.unpack("i"*6, f.read(4*6))
            e, _ = struct.unpack("dd", f.read(8*2))

            if restricted:
                logger.debug(f"Added restricted state {i}")
                X = (
                    np.array(struct.unpack("d"*nvals_a, f.read(nvals_a * 8))).reshape(NDA,NAA),
                )
            else:
                logger.debug(f"Added unrestricted state {i}")
                X = (
                    np.array(struct.unpack("d"*nvals_a, f.read(nvals_a * 8))).reshape(NDA,NAA),
                    np.array(struct.unpack("d"*nvals_b, f.read(nvals_b * 8))).reshape(NDB,NAB),
                )
            states.append(CIstate(m, r, e, X, offset))

        logger.info(f"Read {n_trans} {"restricted" if restricted else "unrestricted"} cis states.")
        return states


class CISParser(ParserBase):
    file_stream: bool = True

    def parse(self) -> Optional[List[CIstate]]:
        p = self.path / (self.baseName + ".cis")
        result = parse_cis(p, self.logger)
        if isinstance(result, Exception):
            self.logger.error(f"Error occured when parsing cis: {result}")
            self.logger.debug("Traceback for above error:", exc_info=(type(result), result, result.__traceback__))
            return None
        else:
            self.logger.info(f"Successfully read CIS from {p}")
            return result