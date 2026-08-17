"""
Loader for libcint shared library

Search order:
1. ``LIBCINT_PATH`` environment variable -- explicit path to cimpiled libcint (used during dev)
2. A libcin library bundled in the ``_libcint`` directory at build time
"""

from __future__ import annotations

import ctypes
import importlib.resources
import os
import sys

_LIB_NAMES = {
    "win32": ["cint.dll"],
    "darwin": ["libcint.dylib"],
    "linux": ["libcint.so", "libcint.so.6"],
}

def _candidate_paths():
    override = os.environ.get("LIBCINT_PATH")
    if override:
        yield override

    pkg = __package__
    if pkg:
        libdir = importlib.resources.files(pkg).joinpath("_libcint")
        for name in _LIB_NAMES[sys.platform]:
            path = libdir.joinpath(name)
            if path.is_file():
                yield str(path)

def load_libcint() -> ctypes.CDLL:
    """Locate and load the libcint shared libary, returned as ``ctypes.CDLL``"""
    tried = []
    for path in _candidate_paths():
        try:
            return ctypes.CDLL(path)
        except OSError as exc:
            tried.append(f"{path} ({exc})")
    msg = "Could not load libcint."
    if tried:
        msg += " Tried:\n  " + "\n ".join(tried)
    msg += (
        "\nBuild the package with its CMAKE/scikit-bild-core backend or set"
        "LIBCINT_PATH to the path of a compiled libcint shared library."
    )
    raise OSError(msg)