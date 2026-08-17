"""Golden tests: native C molden parser vs the reference Python parser."""

import os
from pathlib import Path

import numpy as np
import pytest

from qctooling._molden_helper import load_molden_helper, parse_molden
from qctooling.Parser.molden import post_correction, read_molden

DEV = Path(__file__).resolve().parent.parent / "dev"

try:
    load_molden_helper()
    _HAVE_NATIVE = True
except OSError:
    _HAVE_NATIVE = False

pytestmark = pytest.mark.skipif(
    not _HAVE_NATIVE,
    reason="native molden_helper library not built",
)


def _normalize(alpha, coeff, l):
    """Orca normalization, matching molden.py normalization_funcs['orca']."""
    base = np.power(2 * alpha / np.pi, 3 / 4) * np.power(4 * alpha, l / 2)
    if l == 4:
        return coeff * np.sqrt(3) / base
    return coeff / base


def _native_to_wfn(d, program):
    from qctooling.classes import Basis_grp, Wfn, Xyz

    basis = []
    for atom_idx, n, l, nprim, alpha, coeff in d.basis:
        assert len(alpha) == nprim and len(coeff) == nprim
        norm = coeff if program == "multiwfn" else _normalize(alpha, coeff, l)
        basis.append(Basis_grp(atom_idx, n, l, alpha, norm))
    xyz = Xyz(d.elements, d.coords)
    wfn = Wfn(basis, xyz, d.C, d.occ, d.ene, d.spin, d.irrep)
    return post_correction(wfn, program)


@pytest.mark.parametrize(
    ("file", "program"),
    [
        ("MnCl5.molden.input", "orca"),
        ("MnCl5_mwfn.molden", "multiwfn"),
    ],
)
def test_native_matches_python(file, program):
    path = DEV / file
    ref = read_molden(path, program)
    native = _native_to_wfn(parse_molden(path), program)

    np.testing.assert_array_equal(native.xyz.elements, ref.xyz.elements)
    np.testing.assert_allclose(native.xyz.coordinates, ref.xyz.coordinates, rtol=0, atol=1e-14)

    assert len(native.basis) == len(ref.basis)
    for nb, rb in zip(native.basis, ref.basis):
        assert nb.atom_idx == rb.atom_idx
        assert nb.n == rb.n
        assert nb.l == rb.l
        np.testing.assert_allclose(nb.alpha, rb.alpha, rtol=0, atol=1e-14)
        np.testing.assert_allclose(nb.coeff, rb.coeff, rtol=0, atol=1e-14)

    np.testing.assert_allclose(native.C, ref.C, rtol=0, atol=1e-12)
    np.testing.assert_allclose(native.O, ref.O, rtol=0, atol=1e-14)
    np.testing.assert_allclose(native.E, ref.E, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(native.S, ref.S)
    np.testing.assert_array_equal(native.I, ref.I)


def test_tags():
    assert parse_molden(DEV / "MnCl5.molden.input").tags == ["[5D]", "[7F]", "[9G]"]
    assert parse_molden(DEV / "MnCl5_mwfn.molden").tags == ["[5D]"]


def test_shapes():
    d = parse_molden(DEV / "MnCl5.molden.input")
    assert d.elements.shape == (6,)
    assert d.coords.shape == (6, 3)
    assert len(d.basis) == 80
    assert d.C.shape == (2, 230, 230)
    assert d.occ.shape == d.ene.shape == d.spin.shape == d.irrep.shape == (2, 230)
