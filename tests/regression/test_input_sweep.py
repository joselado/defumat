"""Every input in QE's ``pw_*`` test suite is read, and refused only by name.

The other regression files compare *numbers*. This one compares nothing: it
walks all 252 inputs under ``test-suite/pw_*`` and asserts only that each one
either builds a :class:`~pypresso.system.builder.System` or is refused with a
``NotImplementedError`` that names the feature. It is here because that sweep
found six of the bugs this file was written alongside, and none of them was
reachable from a curated case list:

* two inputs whose ``ATOMIC_SPECIES`` mass is a Fortran ``d``-literal
  (``16.D0``), which ``float()`` does not accept -- the *cards* were not using
  ``fortran_float`` even though its docstring says they need it;
* one input writing two assignments on a line separated by blanks
  (``ecutrho=100.0  nbnd = 8``), which the namelist regex swallowed whole -- and
  where the swallowed text still converts, that failure is silent (``nosym =
  .true. noinv = .true.`` read as ``nosym`` **False**);
* one input asking for ``vdw_corr = 'dft-d2'``, a spelling ``set_vdw_corr`` does
  not recognise, where QE warns and runs on and this code raised;
* one input using Wyckoff positions, refused with "ibrav is required" rather than
  with the reason;
* and the seven ``&control``/``&system`` switches -- ``tot_charge``, ``tefield``,
  ``dipfield``, ``lelfield``, ``assume_isolated``, ``twochem``,
  ``one_atom_occupations`` -- that changed the physics and were read by nobody.

**The assertion is about the failure mode.** A `ValueError`, a `KeyError` or a
`TypeError` out of this sweep means the reader could not cope with valid QE
input; a `NotImplementedError` means it understood the input and said what it
lacks. The first is a bug and the second is the design, so only the second is
allowed -- and only where :data:`EXPECTED_REFUSALS` names the *reason*, which is
what stops a refusal drifting onto the wrong cause the way
`lattice-wyckoff-sio2.in`'s did. Adding an entry is how a new refusal is
declared; a case that starts building again fails until its entry goes.

The whole sweep takes about eight seconds -- it stops at ``build_system`` and
runs no SCF.
"""

import warnings

import pytest

from pypresso.io.pwin import read_pw_input
from pypresso.system import build_system
from tests.conftest import QE_ROOT

pytestmark = pytest.mark.regression

#: Inputs whose refusal is expected and whose reason must appear in the message.
#: Each is a feature `PLAN.md` or `CLAUDE.md` lists as out of scope; the pairing
#: is what stops a refusal drifting onto the wrong cause, which is exactly how
#: `lattice-wyckoff-sio2.in` came to report "ibrav is required".
EXPECTED_REFUSALS = {
    # Wyckoff input (P6, outstanding).
    "lattice-wyckoff-sio2.in": "space_group",
    # DFT+U variants `pypresso.hubbard` refuses by name.
    "lda+U+V-user_ns.in": "intersite",
    "lda+U+V_background.in": "intersite",
    "lda+U+V_force_stress_ortho.in": "intersite",
    "lda+U+V_noncol_ortho.in": "intersite",
    "lda+U_kind1_collin.in": "Liechtenstein",
    "lda+U_kind1_noncollin.in": "Liechtenstein",
    "lda+U_background_one_channel.in": "background",
    "lda+U_orbital_resolved.in": "orbital-resolved",
    # The seven &control/&system switches that change the physics.
    "atom-occ1.in": "one_atom_occupations",
    "atom-occ2.in": "one_atom_occupations",
    "cluster1.in": "assume_isolated",
    "cluster2.in": "tot_charge",
    "cluster3.in": "assume_isolated",
    "cluster4.in": "tot_charge",
    "cluster5.in": "tot_charge",
    "cluster6.in": "tot_charge",
    "cluster_gs.in": "tot_charge",
    "cluster_gs50.in": "tot_charge",
    "cluster_rs.in": "tot_charge",
    "cluster_rs50.in": "tot_charge",
    "2dcutoff.in": "assume_isolated",
    "dipole.in": "tefield",
    "relax-el.in": "tefield",
    "electric-1.in": "lelfield",
    "electric-2.in": "lelfield",
}

#: QE evaluates arithmetic in an input field (``clib/eval_infix.c``), so
#: ``celldm(1) = 4/3`` and ``nat = 1-1`` are legal. `pw_eval` is the directory
#: that exists to test it and nothing else here needs it, so it is a gap rather
#: than a refusal: the parser reports a number it could not read.
NEEDS_INFIX_ARITHMETIC = {"eval_infix.in", "eval_infix-2.in"}


def _inputs():
    suite = QE_ROOT / "test-suite"
    if not suite.is_dir():
        return [pytest.param(None, marks=pytest.mark.skip(reason="QE tree absent"))]
    found = [
        pytest.param(path, id=f"{path.parent.name}/{path.name}")
        for path in sorted(suite.glob("pw_*/*.in"))
        if not path.name.startswith("benchmark")
    ]
    assert found, "no inputs discovered"
    return found


@pytest.mark.parametrize("path", _inputs())
def test_every_qe_input_is_read_or_refused_by_name(path):
    if path.name in NEEDS_INFIX_ARITHMETIC:
        pytest.skip("eval_infix arithmetic in an input field is not implemented")

    with warnings.catch_warnings():
        # ``vdw_corr = 'dft-d2'`` warns here exactly as ``set_vdw_corr`` does.
        warnings.simplefilter("ignore")
        try:
            build_system(read_pw_input(path))
        except NotImplementedError as refusal:
            expected = EXPECTED_REFUSALS.get(path.name)
            assert expected is not None, (
                f"{path.name} is refused and is not in EXPECTED_REFUSALS; either "
                f"it has stopped being readable or the entry is missing: {refusal}"
            )
            assert expected in str(refusal), (
                f"{path.name} is refused for the wrong reason: expected "
                f"{expected!r} in {str(refusal)!r}"
            )
        else:
            assert path.name not in EXPECTED_REFUSALS, (
                f"{path.name} now builds; drop its EXPECTED_REFUSALS entry"
            )
