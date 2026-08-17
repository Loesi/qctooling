"""
ctypes loader and interface for the native molden parser (native/molden_helper.c).

Search order:
1. ``MOLDEN_HELPER_PATH`` environment variable -- explicit path (used during dev)
2. A library bundled in the ``_native`` directory at build time
"""

from __future__ import annotations

import ctypes
import importlib.resources
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import numpy.typing as npt

_LIB_NAMES = {
    "win32": ["molden_helper.dll"],
    "darwin": ["libmolden_helper.dylib"],
    "linux": ["libmolden_helper.so"],
}


class _MoldenResult(ctypes.Structure):
    """C mirror of `typedef struct MoldenResult` in molden_helper.h."""

    _fields_ = [
        ("status", ctypes.c_int32),
        ("error", ctypes.c_char_p),
        ("n_atom", ctypes.c_int32),
        ("elements", ctypes.POINTER(ctypes.c_char_p)),
        ("coords", ctypes.POINTER(ctypes.c_double)),
        ("n_shell", ctypes.c_int32),
        ("shell_atom", ctypes.POINTER(ctypes.c_int32)),
        ("shell_n", ctypes.POINTER(ctypes.c_int32)),
        ("shell_l", ctypes.POINTER(ctypes.c_int32)),
        ("shell_nprim", ctypes.POINTER(ctypes.c_int32)),
        ("shell_off", ctypes.POINTER(ctypes.c_int32)),
        ("alpha", ctypes.POINTER(ctypes.c_double)),
        ("coeff", ctypes.POINTER(ctypes.c_double)),
        ("n_tags", ctypes.c_int32),
        ("tags", ctypes.POINTER(ctypes.c_char_p)),
        ("n_ao", ctypes.c_int32),
        ("n_spin", ctypes.c_int32),
        ("C", ctypes.POINTER(ctypes.c_double)),
        ("occ", ctypes.POINTER(ctypes.c_double)),
        ("ene", ctypes.POINTER(ctypes.c_double)),
        ("spin", ctypes.POINTER(ctypes.c_int32)),
        ("irrep", ctypes.POINTER(ctypes.c_char_p)),
    ]


@dataclass(frozen=True)
class MoldenData:
    """Raw parsed molden data with simple arrays/tuples ready to be wrapped."""

    elements: npt.NDArray[np.str_]
    coords: npt.NDArray[np.float64]
    basis: List[Tuple[int, int, int, int, npt.NDArray[np.float64], npt.NDArray[np.float64]]]
    tags: List[str]
    C: npt.NDArray[np.float64]
    occ: npt.NDArray[np.float64]
    ene: npt.NDArray[np.float64]
    spin: npt.NDArray[np.bool_]
    irrep: npt.NDArray[np.str_]

    @property
    def n_ao(self) -> int:
        return self.C.shape[-1]

    @property
    def n_spin(self) -> int:
        return self.C.shape[0]


def _candidate_paths():
    override = os.environ.get("MOLDEN_HELPER_PATH")
    if override:
        yield override

    pkg = __package__
    if pkg:
        import pathlib

        mod = sys.modules[pkg]
        # Editable installs can expose the package across several __path__
        # entries (e.g. src + site-packages), so search all of them.
        for entry in getattr(mod, "__path__", []):
            libdir = pathlib.Path(entry) / "_native"
            for name in _LIB_NAMES[sys.platform]:
                path = libdir / name
                if path.is_file():
                    yield str(path)


_lib = None


def load_molden_helper() -> ctypes.CDLL:
    """Locate and load the native molden parser, returned as ``ctypes.CDLL``."""
    global _lib
    if _lib is not None:
        return _lib

    tried = []
    for path in _candidate_paths():
        try:
            lib = ctypes.CDLL(path)
            break
        except OSError as exc:
            tried.append(f"{path} ({exc})")
    else:
        msg = "Could not load molden_helper."
        if tried:
            msg += " Tried:\n  " + "\n ".join(tried)
        msg += (
            "\nBuild the package with its CMake/scikit-build-core backend or set"
            "MOLDEN_HELPER_PATH to the path of a compiled libmolden_helper."
        )
        raise OSError(msg)

    lib.molden_parse.restype = ctypes.POINTER(_MoldenResult)
    lib.molden_parse.argtypes = [ctypes.c_char_p]
    lib.molden_result_free.argtypes = [ctypes.POINTER(_MoldenResult)]
    _lib = lib
    return lib


def _strs(ptr: ctypes._Pointer, n: int) -> List[str]:
    return [ptr[i].decode("utf-8") for i in range(n)]


def parse_molden(path: os.PathLike[str]) -> MoldenData:
    """Parse a molden file with the native parser, returning a ``MoldenData``.

    Coefficients and exponents are returned raw (unnormalized); apply
    per-program normalization when wrapping into higher-level objects.
    """
    lib = load_molden_helper()
    p = lib.molden_parse(os.fspath(path).encode("utf-8"))
    if not p:
        raise RuntimeError("molden_parse returned NULL")
    res = p.contents
    try:
        if res.status:
            err = res.error.decode("utf-8") if res.error else f"status {res.status}"
            raise ValueError(err)

        n_atom = res.n_atom
        elements = np.array(_strs(res.elements, n_atom), dtype=str)

        coords = np.ctypeslib.as_array(res.coords, shape=(n_atom, 3)).copy()

        n_shell = res.n_shell
        nprim_all = np.ctypeslib.as_array(res.shell_nprim, shape=(n_shell,)).copy()
        off_all = np.ctypeslib.as_array(res.shell_off, shape=(n_shell,)).copy()
        atom_all = np.ctypeslib.as_array(res.shell_atom, shape=(n_shell,)).copy()
        n_all = np.ctypeslib.as_array(res.shell_n, shape=(n_shell,)).copy()
        l_all = np.ctypeslib.as_array(res.shell_l, shape=(n_shell,)).copy()
        alpha_all = np.ctypeslib.as_array(res.alpha, shape=(int(nprim_all.sum()),)).copy()
        coeff_all = np.ctypeslib.as_array(res.coeff, shape=(int(nprim_all.sum()),)).copy()

        basis = []
        for i in range(n_shell):
            off, nprim = int(off_all[i]), int(nprim_all[i])
            basis.append(
                (
                    int(atom_all[i]),
                    int(n_all[i]),
                    int(l_all[i]),
                    nprim,
                    alpha_all[off : off + nprim],
                    coeff_all[off : off + nprim],
                )
            )

        tags = _strs(res.tags, res.n_tags)

        n_ao, n_spin = res.n_ao, res.n_spin
        C = np.ctypeslib.as_array(res.C, shape=(n_spin, n_ao, n_ao)).copy()
        occ = np.ctypeslib.as_array(res.occ, shape=(n_spin, n_ao)).copy()
        ene = np.ctypeslib.as_array(res.ene, shape=(n_spin, n_ao)).copy()
        spin = np.ctypeslib.as_array(res.spin, shape=(n_spin, n_ao)).copy().astype(bool)
        irrep = np.array(
            [_strs(res.irrep[i * n_ao : (i + 1) * n_ao], n_ao) for i in range(n_spin)],
            dtype=str,
        )

        return MoldenData(elements, coords, basis, tags, C, occ, ene, spin, irrep)
    finally:
        lib.molden_result_free(p)
