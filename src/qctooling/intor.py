"""Minimal ctypes binding to libcint for one-electron integrals.

A libcint molecular/basis description consists of three plain arrays,
``atm``, ``bas`` and ``env`` (layout documented in ``include/cint.h.in`` of
the libcint source).  This module builds those arrays from numpy data and
evaluates one-electron integrals shell-block by shell-block through the
low-level ``int1e_<name>_sph`` / ``int1e_<name>_cart`` entry points, using
the full 10-argument signature with ``dims = opt = cache = NULL``.

Only ``numpy`` and ``ctypes`` are required at runtime.
"""

from __future__ import annotations

import ctypes
from functools import wraps
import inspect
import numpy as np
import numpy.typing as npt
from typing import List, Sequence, Tuple, Optional, overload

from ._libcint import load_libcint

# ---- env layout (from include/cint.h.in) ----
PTR_EXPCUTOFF = 0
PTR_ENV_START = 20

# atm slots
CHARGE_OF, PTR_COORD, NUC_MOD_OF, PTR_ZETA, PTR_FRAC_CHARGE = range(5)
ATM_SLOTS = 6  # + RESERVE_ATMSLOT

# bas slots
ATOM_OF, ANG_OF, NPRIM_OF, NCTR_OF, KAPPA_OF, PTR_EXP, PTR_COEFF = range(7)
BAS_SLOTS = 8  # + RESERVE_BASLOT

NP_INT = np.int32

# libcint index (within a spherical shell of angular momentum l) of the
# function at position j of the molden/ORCA ordering (m = 0, +1, -1, +2, -2,
# ...).  libcint orders a shell as ascending m for l >= 2 but as px, py, pz
# for l = 1, hence the p entry differs from the ascending-m pattern.
_MOLDEN_PERM: List[Tuple[int, ...]] = [
    (0,),                            # l = 0
    (0, 1, 2),                       # l = 1  pz, px, py
    (2, 3, 1, 4, 0),                 # l = 2  dz2, dxz, dyz, dx2y2, dxy
    (3, 4, 2, 5, 1, 6, 0),           # l = 3  f0, f1, f-1, f2, f-2, f3, f-3
    (4, 5, 3, 6, 2, 7, 1, 8, 0),     # l = 4  g0, g1, g-1, g2, g-2, g3, g-3, g4, g-4
]

_cint = None


def _lib() -> "ctypes.CDLL":

    global _cint
    if _cint is None:
        lib = load_libcint()
        _setup_argtypes(lib)
        _cint = lib
    return _cint


# ---------------------------------------------------------------------------
# low-level libcint helpers
# ---------------------------------------------------------------------------

_CFUNC_CGTO = [
    np.ctypeslib.ndpointer(NP_INT, ndim=2),
]


def _setup_argtypes(lib: "ctypes.CDLL") -> None:
    lib.CINTcgto_spheric.restype = ctypes.c_int
    lib.CINTcgto_spheric.argtypes = [ctypes.c_int] + _CFUNC_CGTO
    lib.CINTcgto_cart.restype = ctypes.c_int
    lib.CINTcgto_cart.argtypes = [ctypes.c_int] + _CFUNC_CGTO
    lib.CINTtot_cgto_spheric.restype = ctypes.c_int
    lib.CINTtot_cgto_spheric.argtypes = [
        np.ctypeslib.ndpointer(NP_INT, ndim=2),
        ctypes.c_int,
    ]
    lib.CINTtot_cgto_cart.restype = ctypes.c_int
    lib.CINTtot_cgto_cart.argtypes = [
        np.ctypeslib.ndpointer(NP_INT, ndim=2),
        ctypes.c_int,
    ]
    lib.CINTshells_spheric_offset.restype = None
    lib.CINTshells_spheric_offset.argtypes = [
        np.ctypeslib.ndpointer(NP_INT, ndim=1),
        np.ctypeslib.ndpointer(NP_INT, ndim=2),
        ctypes.c_int,
    ]
    lib.CINTshells_cart_offset.restype = None
    lib.CINTshells_cart_offset.argtypes = [
        np.ctypeslib.ndpointer(NP_INT, ndim=1),
        np.ctypeslib.ndpointer(NP_INT, ndim=2),
        ctypes.c_int,
    ]
    lib.CINTgto_norm.restype = ctypes.c_double
    lib.CINTgto_norm.argtypes = [ctypes.c_int, ctypes.c_double]


def CINTgto_norm(l: int, alpha: float) -> float:
    """Normalization factor of the primitive GTO ``r^l exp(-alpha r^2)``."""
    return float(_lib().CINTgto_norm(int(l), float(alpha)))


def _cgto(l: int, nctr: int, *, cart: bool) -> int:
    return (l * 2 + 1) * nctr if not cart else (l + 1) * (l + 2) // 2 * nctr


# ---------------------------------------------------------------------------
# molecular/basis description
# ---------------------------------------------------------------------------

# Largest number of one-electron integral components a driver call returns
# (components are stored contiguously in the output buffer).
_INT1E_NCOMP = {
    "ovlp": 1,
    "kin": 1,
    "nuc": 1,
    "r": 3,
    "r2": 1,
    "rr": 9,
}

_INTEGRAL_CACHE = {}


def _int1e_fn(name: str, cart: bool):
    key = (name, cart)
    fn = _INTEGRAL_CACHE.get(key)
    if fn is None:
        symbol = f"int1e_{name}{'_cart' if cart else '_sph'}"
        try:
            fn = getattr(_lib(), symbol)
        except AttributeError:
            raise AttributeError(
                f"libcint integral {symbol!r} is not available in the loaded library"
            ) from None
        fn.restype = ctypes.c_int
        fn.argtypes = [
            np.ctypeslib.ndpointer(np.float64, ndim=1),
            ctypes.c_void_p,  # dims  (NULL -> block dims from shells)
            np.ctypeslib.ndpointer(NP_INT, ndim=1),  # shls (2 ints)
            np.ctypeslib.ndpointer(NP_INT, ndim=2),  # atm
            ctypes.c_int,  # natm
            np.ctypeslib.ndpointer(NP_INT, ndim=2),  # bas
            ctypes.c_int,  # nbas
            np.ctypeslib.ndpointer(np.float64, ndim=1),  # env
            ctypes.c_void_p,  # opt  (NULL)
            ctypes.c_void_p,  # cache (NULL, libcint mallocs internally)
        ]
        _INTEGRAL_CACHE[key] = fn
    return fn


class IntEnv:
    """libcint molecular/basis description (opaque ``atm``, ``bas``, ``env``).

    Instances are cheap views over three numpy arrays; create them with
    :func:`make_intenv`.
    """

    __slots__ = ("atm", "bas", "env")

    def __init__(self, atm: np.ndarray, bas: np.ndarray, env: np.ndarray):
        self.atm = np.ascontiguousarray(atm, dtype=NP_INT)
        self.bas = np.ascontiguousarray(bas, dtype=NP_INT)
        self.env = np.ascontiguousarray(env, dtype=np.float64)

    @property
    def natm(self) -> int:
        return self.atm.shape[0]

    @property
    def nbas(self) -> int:
        return self.bas.shape[0]

    def shell_dim(self, i: int, *, cart: bool = False) -> int:
        """Number of AO functions contributed by shell ``i``."""
        l = int(self.bas[i, ANG_OF])
        nctr = int(self.bas[i, NCTR_OF])
        return _cgto(l, nctr, cart=cart)

    def ao_loc(self, *, cart: bool = False) -> np.ndarray:
        """Cumulative AO offsets per shell, length ``nbas + 1``."""
        loc = np.zeros(self.nbas + 1, dtype=NP_INT)
        lib = _lib()
        fn = lib.CINTshells_cart_offset if cart else lib.CINTshells_spheric_offset
        tot = lib.CINTtot_cgto_cart if cart else lib.CINTtot_cgto_spheric
        fn(loc, self.bas, self.nbas)
        loc[-1] = tot(self.bas, self.nbas)
        return loc

    @property
    def nao(self) -> int:
        """Number of AO functions (real spherical by default)."""
        return int(self.ao_loc()[-1])


def _iter_shells(shells):
    if isinstance(shells, dict):
        for atom, shells_on_atom in shells.items():
            for l, (exps, coeffs) in shells_on_atom.items():
                yield atom, l, exps, coeffs
    else:
        for shell in shells:
            yield shell


def make_intenv(
    coords: npt.ArrayLike,
    shells: Sequence[Tuple[int,int,npt.ArrayLike,npt.ArrayLike]],
    *,
    charges: Optional[npt.ArrayLike]=None,
    normalize: Optional[bool] = True,
) -> IntEnv:
    """Build a libcint :class:`IntEnv` from plain numpy data.

    Parameters
    ----------
    coords : (natm, 3) array_like
        Atom Cartesian coordinates in **Bohr**.
    shells : sequence or dict
        One entry per basis shell.  A sequence of tuples
        ``(atom_index, l, exponents, coeffs)`` where ``exponents`` has shape
        ``(nprim,)`` and ``coeffs`` has shape ``(nprim, nctr)``; ``nctr``
        contracted functions share the ``nprim`` primitives (libcint general
        contraction).  A dict mapping ``atom_index -> {l: (exps, coeffs)}``
        is accepted as well.
    charges : (natm,) array_like, optional
        Nuclear charges; only needed for integrals that involve the nuclei
        (e.g. ``int1e_nuc``).
    normalize : bool, default=True
        Multiply every contraction coefficient by the primitive GTO
        normalization factor ``CINTgto_norm(l, alpha)``.  Set to False when
        your coefficients are already in the normalization convention you
        want to reproduce.

    Returns
    -------
    IntEnv
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (natm, 3)")
    natm = coords.shape[0]

    if charges is None:
        charges = np.zeros(natm, dtype=NP_INT)
    charges = np.asarray(charges, dtype=NP_INT)
    if charges.size != natm:
        raise ValueError("charges must have one entry per atom")

    shells = list(_iter_shells(shells))
    if not shells:
        raise ValueError("no shells given")


    sizes = []
    for atom, l, exps, coeffs in shells:
        exps = np.asarray(exps, dtype=np.float64)
        coeffs = np.asarray(coeffs, dtype=np.float64)
        if exps.ndim != 1 or coeffs.ndim != 2 or coeffs.shape[0] != exps.size:
            print(exps.ndim)
            print(exps)
            print(coeffs)
            raise ValueError(
                "coeffs must have shape (exps.size, nctr); got"
                f" exps{exps.shape} coeffs{coeffs.shape} for shell ({atom},{l})"
            )
        sizes.append((int(atom), int(l), exps, coeffs, exps.size, coeffs.size))

    env_size = PTR_ENV_START + 3 * natm
    env_size += sum(nprim + coeffs.size for _, _, _, coeffs, nprim, _ in sizes)

    atm = np.zeros((natm, ATM_SLOTS), dtype=NP_INT)
    bas = np.zeros((len(shells), BAS_SLOTS), dtype=NP_INT)
    env = np.zeros(env_size, dtype=np.float64)

    off = PTR_ENV_START
    for i in range(natm):
        atm[i, CHARGE_OF] = charges[i]
        atm[i, PTR_COORD] = off
        env[off : off + 3] = coords[i]
        off += 3

    for r, (atom, l, exps, coeffs, nprim, nc) in enumerate(sizes):
        bas[r, ATOM_OF] = atom
        bas[r, ANG_OF] = l
        bas[r, NPRIM_OF] = nprim
        bas[r, NCTR_OF] = coeffs.shape[1]
        bas[r, KAPPA_OF] = 0  # spinor-related, unused for _sph/_cart
        bas[r, PTR_EXP] = off
        env[off : off + nprim] = exps
        off += nprim
        bas[r, PTR_COEFF] = off
        if normalize:
            norm = np.array([CINTgto_norm(l, a) for a in exps], dtype=np.float64)
            coeffs = coeffs * norm[:, None]
        # libcint stores a general-contracted shell's coefficients
        # contraction-major: c[ic*nprim + ip] for contraction ic, primitive ip
        # (see CINTprim_to_ctr_* in src/g1e.c); coeffs is (nprim, nctr).
        env[off : off + nc] = coeffs.ravel(order="F")
        off += nc

    return IntEnv(atm, bas, env)


# ---------------------------------------------------------------------------
# one-electron integral driver
# ---------------------------------------------------------------------------


def int1e(name: str, intenv: IntEnv, *, cart: bool = False) -> np.ndarray:
    """Compute a one-electron integral matrix in the AO basis.

    ``name`` selects the operator and maps to the libcint entry point
    ``int1e_<name>_sph`` / ``int1e_<name>_cart`` (the latter when
    ``cart=True``).  Known, pre-registered names with their number of
    independent components::

        ovlp  1   overlap S
        kin   1   kinetic energy
        nuc   1   electron-nucleus attraction
        r     3   dipole moment (one matrix per Cartesian component)
        r2    1   <i| r^2 |j>
        rr    9   <i| r_a r_b |j>

    Unregistered names default to one component.

    Returns a ``(nao, nao)`` array, or ``(ncomp, nao, nao)`` when the
    integral has more than one component.
    """
    if not isinstance(intenv, IntEnv):
        raise TypeError("intenv must be an IntEnv (see make_intenv)")
    fn = _int1e_fn(name, cart)

    nbas = intenv.nbas
    dims = np.fromiter(
        (intenv.shell_dim(i, cart=cart) for i in range(nbas)),
        dtype=NP_INT,
        count=nbas,
    )
    ao_loc = intenv.ao_loc(cart=cart)
    nao = int(ao_loc[-1])
    ncomp = _INT1E_NCOMP.get(name, 1)

    if ncomp == 1:
        out = np.zeros((nao, nao))
    else:
        out = np.zeros((ncomp, nao, nao))

    shls = np.empty(2, dtype=NP_INT)
    for j in range(nbas):

        for i in range(j + 1):
            shls[0], shls[1] = i, j
            di, dj = dims[i], dims[j]
            buf = np.zeros(ncomp * di * dj)
            fn(
                buf,
                None,
                shls,
                intenv.atm,
                intenv.natm,
                intenv.bas,
                intenv.nbas,
                intenv.env,
                None,
                None,
            )
            i0, j0 = ao_loc[i], ao_loc[j]
            for c in range(ncomp):
                blk = buf[c * di * dj : (c + 1) * di * dj].reshape(di, dj, order="F")
                if ncomp == 1:
                    out[i0 : i0 + di, j0 : j0 + dj] = blk
                    if i != j:
                        out[j0 : j0 + dj, i0 : i0 + di] = blk.T
                else:
                    out[c, i0 : i0 + di, j0 : j0 + dj] = blk
                    if i != j:
                        out[c, j0 : j0 + dj, i0 : i0 + di] = blk.T

    if cart:
        return out

    order = np.empty(nao, dtype=NP_INT)
    i = 0
    for r in range(nbas):
        l = int(intenv.bas[r, ANG_OF])
        nctr = int(intenv.bas[r, NCTR_OF])
        d = 2 * l + 1
        perm = np.asarray(_MOLDEN_PERM[l], dtype=NP_INT)
        for ic in range(nctr):
            base = i + ic * d
            order[base : base + d] = np.arange(base, base + d, dtype=NP_INT)[perm]
        i += nctr * d

    if ncomp == 1:
        return out[np.ix_(order, order)]
    return out[np.ix_(np.arange(ncomp), order, order)]

def accepts_make_intenv(fc):
    base = inspect.signature(make_intenv)
    keep = {p.name: p for p in base.parameters.values()}
    own = [p for p in inspect.signature(fc).parameters.values() if p.name != "intenv"]
    new_sig = inspect.signature(fc).replace(parameters=[
        keep["coords"], keep["shells"], keep["charges"], keep["normalize"], *own,
    ])

    @wraps(fc)
    def dispatch(coords, shells=None, *, charges=None, normalize=True, **kwargs):
        if isinstance(coords, IntEnv):            # power-user overload
            return fc(coords, **kwargs)
        if shells is None:
            raise TypeError("make_intenv requires 'shells'")
        return fc(                               # raw-data overload: just converts
            make_intenv(coords, shells, charges=charges, normalize=normalize),
            **kwargs,
        )

    dispatch.__signature__ = new_sig
    return dispatch

@overload
def overlap(
    coords: npt.ArrayLike,
    shells: Sequence[Tuple[int,int,npt.ArrayLike,npt.ArrayLike]],
    *,
    charges: Optional[npt.ArrayLike] = None,
    normalize: Optional[bool] = True,
    cart: bool = False,
) -> np.ndarray:
    """AO overlap matrix S."""
@overload
def overlap(intenv: IntEnv, *, cart: bool = False) -> np.ndarray:
    """Overlap from a prebuilt :class:`IntEnv`."""
@accepts_make_intenv
def overlap(intenv: IntEnv, *, cart: bool = False) -> np.ndarray:
    """AO overlap matrix S."""
    return int1e("ovlp", intenv, cart=cart)

@overload
def kinetic(
    coords: npt.ArrayLike,
    shells: Sequence[Tuple[int,int,npt.ArrayLike,npt.ArrayLike]],
    *,
    charges: Optional[npt.ArrayLike] = None,
    normalize: Optional[bool] = True,
    cart: bool = False,
) -> np.ndarray:
    """AO kinetic-energy matrix T = <mu| -1/2 nabla^2 |nu>."""
@overload
def kinetic(intenv: IntEnv, *, cart: bool = False) -> np.ndarray:
    """Overlap from a prebuilt :class:`IntEnv`."""
@accepts_make_intenv
def kinetic(intenv: IntEnv, *, cart: bool = False) -> np.ndarray:
    """AO kinetic-energy matrix T = <mu| -1/2 nabla^2 |nu>."""
    return int1e("kin", intenv, cart=cart)

@overload
def nuclear(
    coords: npt.ArrayLike,
    shells: Sequence[Tuple[int,int,npt.ArrayLike,npt.ArrayLike]],
    *,
    charges: Optional[npt.ArrayLike] = None,
    normalize: Optional[bool] = True,
    cart: bool = False,
) -> np.ndarray:
    """AO electron-nucleus attraction matrix, charges from make_intenv."""
@overload
def nuclear(intenv: IntEnv, *, cart: bool = False) -> np.ndarray:
    """Overlap from a prebuilt :class:`IntEnv`."""
@accepts_make_intenv
def nuclear(intenv: IntEnv, *, cart: bool = False) -> np.ndarray:
    """AO electron-nucleus attraction matrix, charges from make_intenv."""
    return int1e("nuc", intenv, cart=cart)