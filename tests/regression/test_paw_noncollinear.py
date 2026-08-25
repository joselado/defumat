"""P33 check: PAW's one-centre gradient correction with a noncollinear magnetization.

``PW/src/paw_onecenter.f90``'s ``PAW_gcxc_potential`` resolves each direction of
the radial quadrature onto the local spin axis before evaluating the functional
(``compute_rho_spin_lm``) and attaches the splitting back to ``m-hat`` afterwards
(``compute_pot_nonc``). This file checks the transcription the only way that is
sharp: against the *collinear* branch, on a magnetization that happens to point
along one axis, where the two must agree as algebra rather than to a tolerance.

The trap the refusal that used to stand here named is real and is what the
implementation does differently from the collinear path: the rotated channel
densities' **multipoles are recomputed from their grid values**, because the
rotation runs through ``|m|`` and no combination of the stored multipoles is the
expansion of the result.
"""

from pathlib import Path

import numpy as np
import pytest

import jax.numpy as jnp

from pypresso.io.pwin import read_pw_input
from pypresso.paw.gradient import onecenter_gradient_correction
from pypresso.paw.onecenter import build_paw
from pypresso.pseudo import read_upf
from pypresso.scf import run_scf
from pypresso.system import build_system
from pypresso.xc.functional import get_functional

pytestmark = [pytest.mark.regression, pytest.mark.slow]

#: An oxygen atom in a box: PAW, PBE, and a triplet ground state, so the
#: magnetization it polarises to is real rather than seeded and held.
OXYGEN = """ &control
    calculation = 'scf'
 /
 &system
    ibrav=1, celldm(1)=10.0,
    nat=1, ntyp=1,
    ecutwfc=40.0, ecutrho=320.0,
    occupations='smearing', smearing='gaussian', degauss=0.01
 /
 &electrons
    conv_thr = 1.0d-10
 /
ATOMIC_SPECIES
 O  15.999  O.pbe-kjpaw.UPF
ATOMIC_POSITIONS (crystal)
 O 0.00 0.00 0.00
K_POINTS gamma
"""


def _oxygen(tmp_path, pseudo_dir, **overrides):
    path = tmp_path / "o.in"
    path.write_text(OXYGEN)
    data = read_pw_input(path)
    data.namelists["system"].update(overrides)
    system = build_system(data)
    pseudos = tuple(
        read_upf(pseudo_dir / species.pseudo_file)
        for species in system.structure.species
    )
    return system, pseudos


def test_the_two_branches_agree_as_algebra(tmp_path, pseudo_dir):
    """``(n, 0, 0, m)`` through the ``nspin = 4`` branch is ``(up, down)``.

    Fed the same physical state, the one-centre gradient correction must return
    the same energy and the same potential whichever representation it is handed.
    It returns a **bit-identical** energy, a potential agreeing to 5.5e-11
    relative -- the residue is the multipole round-trip the noncollinear branch
    does and the collinear one does not -- and transverse components that are
    exactly zero, which is the statement that a functional of ``|m|`` cannot
    produce a torque.
    """
    system, pseudos = _oxygen(tmp_path, pseudo_dir)
    paw = build_paw(pseudos, system.structure, get_functional("PBE")).species[0]

    r = np.asarray(paw.r)
    profile = np.exp(-2 * r) * r**2
    channels = np.zeros((2, paw.nlm, len(r)))
    channels[0, 0], channels[1, 0] = profile, 0.6 * profile
    for lm in (1, 2, 3, 4):
        channels[0, lm] = 0.08 * profile * np.cos(lm)
        channels[1, lm] = 0.05 * profile * np.sin(lm)

    ylm = paw.angular.ylm[:, : paw.nlm]
    core = jnp.asarray(np.asarray(paw.core_ae))
    up, down = jnp.asarray(channels[0]), jnp.asarray(channels[1])

    collinear_lm = jnp.stack([up, down])
    v2, e2 = onecenter_gradient_correction(
        collinear_lm, jnp.einsum("xl,slr->sxr", ylm, collinear_lm), core, paw
    )
    noncollinear_lm = jnp.stack([up + down, 0 * up, 0 * up, up - down])
    v4, e4 = onecenter_gradient_correction(
        noncollinear_lm, jnp.einsum("xl,slr->sxr", ylm, noncollinear_lm), core, paw,
        axis=np.array([0.0, 0.0, 1.0]),
    )

    assert float(e4) == float(e2)
    scale = float(jnp.abs(v2[0]).max())
    assert float(jnp.abs(v4[0] - 0.5 * (v2[0] + v2[1])).max()) < 1.0e-9 * scale
    assert float(jnp.abs(v4[3] - 0.5 * (v2[0] - v2[1])).max()) < 1.0e-9 * scale
    assert float(jnp.abs(v4[1]).max()) == 0.0
    assert float(jnp.abs(v4[2]).max()) == 0.0


def test_a_magnetic_paw_gga_run_matches_the_collinear_one(tmp_path, pseudo_dir):
    """An oxygen atom polarises to 2 mu_B either way, at the same energy.

    End to end, which the algebra test above cannot be: the SCF, the mixer, the
    spinor eigensolver and the one-centre terms together. The two regimes agree
    to 2.8e-6 Ry, and that is *not* this branch's error -- the same comparison
    under LDA, which never enters it, differs by 3.1e-6. An isolated atom at
    Gamma with smearing is simply a system the two eigensolver paths converge
    slightly differently on.
    """
    collinear = run_scf(
        *_oxygen(tmp_path, pseudo_dir, nspin=2,
                 starting_magnetization={(1,): 0.5}),
        nbnd=8, conv_thr=1.0e-10, max_iterations=200, mixing_beta=0.3, tstress=False,
    )
    spinor = run_scf(
        *_oxygen(tmp_path, pseudo_dir, noncolin=True,
                 starting_magnetization={(1,): 0.5}, angle1={(1,): 0.0}),
        nbnd=8, conv_thr=1.0e-10, max_iterations=200, mixing_beta=0.3, tstress=False,
    )

    assert collinear.converged and spinor.converged
    assert collinear.magnetization == pytest.approx(2.0, abs=1.0e-3)
    assert spinor.magnetization_vector[2] == pytest.approx(2.0, abs=1.0e-3)
    assert spinor.total_energy == pytest.approx(collinear.total_energy, abs=1.0e-5)
