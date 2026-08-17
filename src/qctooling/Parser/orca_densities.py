### read density info
import struct, pathlib
from typing import Union, List, Tuple, Generator, Any
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, field_validator

from .parserBase import ParserBase

def parse_densinfo(info_path: pathlib.Path) -> List[Tuple[str, int, int]]:
    with info_path.open("rb") as f:
        data = f.read()

        file_size = len(data)
        sections = int(file_size / 661)
        header_len = int(file_size % 661)

        header = data[:header_len]
        print_sec = struct.unpack('<i', header[:4])[0]
        assert sections == print_sec, f"expected {print_sec} but got {sections}"
        sys_name = header[4:4+header[4:].find(b'\x00')]
    bytes_info = [data[header_len+i*661:header_len+(i+1)*661] for i in range(int(file_size / 661))]

    section_info = []
    for idx, d in enumerate(bytes_info):
        assert len(d) == 661, f"Unexpected section length in densinfo found; expected 661, got {len(d)}"
        null_pos = d.find(b'\x00')
        
        if null_pos != -1:
            section_name = d[0:null_pos].decode('utf-8', errors='ignore')
        else:
            section_name = "UNKNOWN_SECTION"

        assert d[520:528] == b'\xff'*8, f"got: {d[520:528]}"
        assert d[565:581] == b'\xff'*16, f"got: {d[565:581]}"

        rows, cols = struct.unpack('<II', d[528:536])
        section_info.append((section_name, rows, cols))
            
    return section_info

def parse_densities(file_path: pathlib.Path, layout: List[Union[Tuple[int, int], int]]) -> Generator[npt.NDArray[np.float64], None, None]:
    bytes_per_float = 8  # double precision 
    with file_path.open("rb") as f:
        for item in layout:
            if isinstance(item, int):
                bytes_to_skip = item * bytes_per_float
                f.seek(bytes_to_skip, 1)
            elif isinstance(item, tuple):
                rows, cols = item
                bytes_to_read = cols * rows * bytes_per_float
                raw = f.read(bytes_to_read)

                if len(raw) != bytes_to_read:
                    raise EOFError(f"Unexpected end of file while reading matrix of size {rows}x{cols}. Expected {bytes_to_read} bytes, but got {len(raw)} bytes.")
                
                array = np.frombuffer(raw, dtype=np.float64).reshape((rows,cols))
                yield array

            else:
                raise TypeError(f"Invalid item type in layout list: {type(item)}. Must be tuple or int.")
            
        extra_byte = f.read(1)
        if extra_byte != b'':
            current_pos = f.tell() - 1  # -1 because we just read a byte
            f.seek(0, 2)               # Seek to the absolute end of the file
            total_size = f.tell()
            remaining_bytes = total_size - current_pos
            
            raise ValueError(
                f"Layout fully processed, but the end of the file was not reached! There are still {remaining_bytes} unparsed bytes left in '{file_path.name}'."
            )

class DensityParser(ParserBase):
    path: pathlib.Path
    baseName: str
    densinfos: List[Tuple[str, int, int]] = []

    def model_post_init(self, __context: Any) -> None:
        self.densinfos = parse_densinfo(self.path / (self.baseName + ".densitiesinfo"))

    @property
    def densities(self):
        return [d[0] for d in self.densinfos]

    def parse_densities(self, densNames: List[str]) -> List[npt.NDArray[np.float64]]:
        dens_idx = []
        unavail_dens = []
        for n in densNames:
            try:
                dens_idx.append(self.densities.index(n))
            except:
                unavail_dens.append(n)

        if len(unavail_dens) != 0:
            raise ValueError(f"Densities {unavail_dens} not available: only {self.densities} are available")

        with self._local_path(self.path / (self.baseName + ".densities")) as dens_path:
            req_dens = parse_densities(
                dens_path,
                [d[1:] if i in dens_idx else d[1]*d[2] for i,d in enumerate(self.densinfos)],
                )

        return [d for d in req_dens]
