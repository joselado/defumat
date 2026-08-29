"""P30 check: a Tran-Blaha SCF end to end.

The unit tests (``tests/unit/test_mgga.py``) pin the *functional* against
analytic limits. These pin the *calculation*: that ``tau`` is what it claims to
be, that the two spin regimes agree, that the gap opens by the amount it opened
by when this was written, and that every regime the implementation does not
cover is refused rather than run.

**There is no QE reference here and there cannot be one.** ``pw.x`` reaches
TB09 only through libxc, and then with ``c = 1`` and a zero Laplacian
(``XClib/xc_wrapper_mgga.f90`` declares its Laplacian argument "not used in QE"
and ``dft_setting_routines.f90`` hands libxc its default parameter list) -- so
what QE computes under ``input_dft = 'tb09'`` is Becke-Johnson without a
Laplacian, which is a different functional. The comparison that *is* available
is against the published band gaps, and it is in ``PLAN.md`` P30.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from pypresso.io.pwin import read_pw_input
from pypresso.pseudo import read_upf
from pypresso.scf import run_scf
from pypresso.scf.driver import Calculation
from pypresso.system import build_system
from pypresso.units import RY_TO_EV
from pypresso.workflows import run_bands

pytestmark = [pytest.mark.regression, pytest.mark.slow]

#: How many valence bands two silicon atoms have.
NOCC = 4


def _system(testsuite: Path, pseudo_dir: Path, **overrides):
    """QE's own two-atom silicon, with ``&system`` variables overridden.

    An **indexed** override (``starting_magnetization(1)``) is translated into
    the representation :func:`~pypresso.io.pwin.parse_pw_input` produces --
    a dict of index tuples under the *base* key -- rather than injected as a
    literal string key, which is a key nothing reads. That was a silent no-op:
    an override was written in the test, was never applied, and the run used the
    default.
    """
    data = read_pw_input(testsuite / "pw_scf" / "scf.in")
    namelist = data.namelists["system"]
    for key, value in overrides.items():
        if value is None:
            continue
        if key.endswith(")") and "(" in key:
            name, _, index = key[:-1].partition("(")
            entries = dict(namelist.get(name) or {})
            entries[tuple(int(i) for i in index.split(","))] = value
            namelist[name] = entries
        else:
            namelist[key] = value
    system = build_system(data)
    pseudos = tuple(
        read_upf(pseudo_dir / species.pseudo_file)
        for species in system.structure.species
    )
    return system, pseudos


@lru_cache(maxsize=None)
def _converged(testsuite: Path, pseudo_dir: Path, dft, mbj_c=None, tstress=False,
               **overrides):
    system, pseudos = _system(testsuite, pseudo_dir, input_dft=dft, mbj_c=mbj_c,
                              **overrides)
    with pytest.warns(RuntimeWarning) if dft else _nothing():
        # ``nbnd = 10`` and not 8, which is what the electron count alone would
        # ask for. **The highest band of an ``nbnd`` window does not converge
        # under this functional where it does under LDA**: at ``nbnd = 8`` band
        # 8 comes out 4.9e-3 Ry away from a dense diagonalisation of the *same*
        # Hamiltonian while every band below it is within 5e-7, and at 10 the
        # whole window is within 1e-6. Davidson resolves the top of its window
        # last, and mBJ's potential -- which carries the structure of
        # ``|grad rho|/rho`` and ``sqrt(tau/rho)`` -- mixes that band with the
        # ones just outside it far more than a local potential does. The density
        # is unaffected (``c`` agrees to every digit either way), so this is a
        # rule for reading eigenvalues, not a convergence failure.
        result = run_scf(system, pseudos, nbnd=10, conv_thr=1.0e-10,
                         max_iterations=90, tstress=tstress)
    return system, pseudos, result


class _nothing:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _gap(result) -> float:
    """The smallest gap over the k-points the SCF itself ran on.

    Not the fundamental gap -- ``scf.in`` has two k-points and neither is at the
    conduction-band minimum -- but a number computed identically for every
    functional, which is what a regression test wants. The fundamental gap on a
    band path is in the notebook.
    """
    ev = np.asarray(result.eigenvalues) * RY_TO_EV
    return float(ev[:, NOCC:].min() - ev[:, :NOCC].max())


def test_tau_integrates_to_the_band_kinetic_energy(qe_testsuite, pseudo_dir):
    """``int tau dr = sum_i w_i |k+G|^2 |c_G|^2``, exactly.

    The one identity that pins the kinetic energy density without reference to
    anything else. It catches the whole class of error the builder is exposed to
    -- the ``1/Omega`` normalisation, the weights, the ``i(k+G)`` factor, the
    scatter into the FFT box and the smooth-to-dense lift -- because every one
    of them changes the integral. A symmetrisation cannot: ``sym_rho`` averages
    over operations that preserve the cell, so this also says the symmetrisation
    -- which ``sum_band.f90`` applies to ``kin_g`` and this code applies too --
    did not lose charge.
    """
    system, pseudos, result = _converged(qe_testsuite, pseudo_dir, "tb09")
    calculation = Calculation(system, pseudos)

    cell = calculation.system.cell
    tau = np.asarray(result.tau)
    integral = cell.volume / tau[0].size * tau.sum()

    weights = jnp.asarray(result.occupations)
    if weights.ndim == 2:
        weights = weights[None]
    kinetic = calculation.kinetic  # (nk, npwx), |k+G|^2 in Ry
    band = float(jnp.sum(
        weights[..., None] * kinetic[None, :, None, :]
        * jnp.abs(result.wavefunctions) ** 2
    ))
    assert integral == pytest.approx(band, abs=1.0e-12)


def test_the_two_spin_regimes_agree(qe_testsuite, pseudo_dir):
    """``nspin = 2`` with no magnetization is ``nspin = 1``, to machine precision.

    The single test that exercises the whole polarized path -- the per-channel
    ``tau``, the spin-scaled Thomas-Fermi guess, the halving that an unpolarized
    density needs and a pair of channels does not, and the fact that ``c`` is an
    average over the **total** density and so must come out bit-identical.

    It is also the test that found the bug it exists for: the channel split was
    written for QE's ``(total, magnetization)`` storage where this package uses
    ``(up, down)``, which left the eigenvalues 7 eV out. Nothing in the
    unpolarized runs could see it.
    """
    smearing = dict(occupations="smearing", smearing="gaussian", degauss=0.02)
    _, _, one = _converged(qe_testsuite, pseudo_dir, "tb09", **smearing)
    # Zero, explicitly: an unpolarized ``nspin = 2`` run reproducing
    # ``nspin = 1`` is the whole point of the test, and ``build_system``
    # requires an LSDA input to say which it wants. ``_system`` translates the
    # indexed spelling into the parser's own representation -- it used to inject
    # the string as a key, which nothing read.
    _, _, two = _converged(
        qe_testsuite, pseudo_dir, "tb09", nspin=2,
        **{"starting_magnetization(1)": 0.0}, **smearing
    )

    assert two.meta_c == pytest.approx(one.meta_c, abs=1.0e-12)
    assert two.total_energy == pytest.approx(one.total_energy, abs=1.0e-10)
    eigenvalues = np.asarray(two.eigenvalues)
    for channel in eigenvalues:
        assert channel == pytest.approx(np.asarray(one.eigenvalues), abs=1.0e-12)
    # tau_up + tau_down is the unpolarized tau, which is the check that the
    # per-channel accumulation shares the density's weights.
    assert np.asarray(two.tau).sum(axis=0) == pytest.approx(
        np.asarray(one.tau)[0], abs=1.0e-12
    )


def test_the_gap_opens_over_the_local_functional(qe_testsuite, pseudo_dir):
    """LDA to Tran-Blaha on QE's own silicon: 2.26 eV to 2.75 eV.

    The headline. Both numbers are at ``ecutwfc = 12`` on two k-points, so
    neither is a converged silicon gap -- what is pinned is the *shift*, and
    that it comes with the ``c`` it should. On a converged setup (30 Ry, a 6x6x6
    grid, a band path) the same comparison is 0.49 eV to 1.13 eV against an
    experimental 1.17; see ``PLAN.md`` P30.
    """
    _, _, lda = _converged(qe_testsuite, pseudo_dir, None)
    _, _, tb09 = _converged(qe_testsuite, pseudo_dir, "tb09")

    assert lda.converged and tb09.converged
    assert lda.meta_c is None
    assert _gap(lda) == pytest.approx(2.2645, abs=2.0e-3)
    assert _gap(tb09) == pytest.approx(2.7454, abs=2.0e-3)
    assert tb09.meta_c == pytest.approx(1.00845, abs=1.0e-4)


def test_becke_johnson_is_tran_blaha_at_c_equal_one(qe_testsuite, pseudo_dir):
    """``bj06`` and ``tb09`` with ``mbj_c = 1`` are the same calculation.

    Which is also the statement that ``pw.x``'s ``input_dft = 'tb09'`` is this
    row and not the one above it: libxc's default ``c`` is 1 and QE never sets
    it. The gap difference between the two is what QE's TB09 is missing.
    """
    _, _, bj06 = _converged(qe_testsuite, pseudo_dir, "bj06")
    _, _, imposed = _converged(qe_testsuite, pseudo_dir, "tb09", mbj_c=1.0)

    assert bj06.meta_c == 1.0
    assert imposed.total_energy == pytest.approx(bj06.total_energy, abs=1.0e-9)
    assert _gap(imposed) == pytest.approx(_gap(bj06), abs=1.0e-6)
    # ...and it is a smaller gap than the self-consistent c gives, because
    # c = 1.008 > 1 and the gap grows with c.
    _, _, tb09 = _converged(qe_testsuite, pseudo_dir, "tb09")
    assert _gap(bj06) < _gap(tb09)


def test_spin_orbit_runs(qe_testsuite, pseudo_dir):
    """``noncolin`` and ``lspinorb`` reach a converged TB09 SCF.

    Only that they run and land where the collinear calculation lands: silicon
    has no magnetization and a scalar-relativistic dataset has no spin-orbit
    term, so a spinor run of it is two copies of the same problem and its
    eigenvalues are the Kramers-doubled scalar ones. The agreement is ~5e-5 eV
    rather than machine precision, and the reason is measured rather than
    assumed -- see :func:`test_the_spinor_tau_is_the_collinear_one`.
    """
    _, _, collinear = _converged(qe_testsuite, pseudo_dir, "tb09")
    _, _, spinor = _converged(qe_testsuite, pseudo_dir, "tb09", noncolin=True)

    assert spinor.converged
    assert spinor.meta_c == pytest.approx(collinear.meta_c, abs=1.0e-4)
    doubled = np.asarray(spinor.eigenvalues)[:, ::2]
    reference = np.asarray(collinear.eigenvalues)[:, : doubled.shape[1]]
    assert doubled == pytest.approx(reference, abs=1.0e-5)


def test_a_band_path_needs_the_converged_tau(qe_testsuite, pseudo_dir):
    """A fixed-density run under a meta-GGA is refused without ``tau``.

    ``tau`` is a property of the occupied states over the whole zone, and a band
    path has no occupations at all -- so unlike the density it cannot be rebuilt
    from what the run is handed. Refused for the same reason PAW's ``becsum``
    is, and it works when given.
    """
    system, pseudos, result = _converged(qe_testsuite, pseudo_dir, "tb09")
    with pytest.raises(NotImplementedError, match="kinetic energy density"):
        run_bands(system, pseudos, result.density, nbnd=10)

    bands = run_bands(system, pseudos, result.density, nbnd=10, tau=result.tau)
    # The SCF's own k-points, so the bands must reproduce its eigenvalues -- to
    # the residual and no tighter. The SCF's eigenvalues come from the potential
    # of the *input* (mixed) density and this one from the output density's, so
    # the two differ by the self-consistency error; at ``conv_thr = 1e-10`` that
    # is ~7e-7 Ry here and ~5e-7 for LDA, which is the scale to hold it to. It
    # was 4.9e-3 before ``nbnd`` was raised, and the cause was the unconverged
    # top band and not ``tau``.
    assert np.asarray(bands.eigenvalues) == pytest.approx(
        np.asarray(result.eigenvalues), abs=5.0e-6
    )


def _test_fields(grid):
    """A smooth, positive, spin-polarised density and ``tau`` on the grid.

    Analytic rather than converged: these are algebra tests of the spin
    bookkeeping, and a made-up field exercises it as well as a real one while
    costing nothing.
    """
    x = np.indices(grid) / np.array(grid)[:, None, None, None]
    charge = (0.4 + 0.25 * np.cos(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1])
              + 0.1 * np.cos(2 * np.pi * x[2]))
    moment = 0.15 * np.cos(2 * np.pi * x[0]) + 0.05
    up, down = 0.5 * (charge + moment), 0.5 * (charge - moment)
    # ``abs`` before the fractional power: ``down`` dips slightly negative where
    # the charge is smallest, and ``x**(5/3)`` of a negative is a NaN that
    # propagates through the whole comparison.
    factor = 0.6 * (6 * np.pi**2) ** (2.0 / 3.0)
    return (charge, moment, up, down,
            factor * np.abs(up) ** (5.0 / 3.0), factor * np.abs(down) ** (5.0 / 3.0))


def test_the_spinor_tau_is_the_collinear_one(qe_testsuite, pseudo_dir):
    """Two spinor bands ``(psi, 0)`` and ``(0, psi)`` give the scalar ``tau``.

    The decisive test of :func:`~pypresso.scf.density.spinor_band_kinetic_density`,
    and it is an *identity* rather than an agreement: embedding a scalar band
    into a spinor at half the weight, twice, is the same state, so the two
    builders must return the same array. They do, to 3e-17.

    It is worth having as algebra rather than as two SCF runs. Those agree only
    to ~5e-6 Ry, because ``tau`` weights the wavefunction by ``|k+G|^2`` and so
    amplifies whatever the two eigensolver paths left differing -- and mBJ is
    nonlinear in ``tau`` on top. That is noise, not a discrepancy, and this test
    is how one tells.
    """
    from pypresso.scf.density import kinetic_energy_density, spinor_kinetic_energy_density

    system, pseudos, result = _converged(qe_testsuite, pseudo_dir, "tb09")
    calculation = Calculation(system, pseudos)
    psi = result.wavefunctions
    weights = jnp.asarray(result.occupations)[None]
    grid, cell = calculation.basis.smooth.grid, calculation.system.cell

    collinear = kinetic_energy_density(
        psi, calculation.fft_index, grid, weights, cell,
        calculation.kplusg, calculation.k_batch,
    )
    zero = jnp.zeros_like(psi[0])
    spinor = jnp.concatenate([
        jnp.concatenate([psi[0], zero], axis=-1),
        jnp.concatenate([zero, psi[0]], axis=-1),
    ], axis=1)
    halved = jnp.concatenate([0.5 * weights[0], 0.5 * weights[0]], axis=1)
    spun = spinor_kinetic_energy_density(
        spinor, calculation.fft_index, grid, halved, cell,
        calculation.kplusg, 1, calculation.k_batch,
    )
    assert float(jnp.abs(collinear - spun).max()) < 1.0e-15


def test_the_local_spin_frame_reduces_to_the_collinear_branch(qe_testsuite, pseudo_dir):
    """A magnetization along ``z`` must give back the two-channel answer exactly.

    ``v_0 = (v_up + v_dw)/2`` and ``v_z = (v_up - v_dw)/2``, with the transverse
    components identically zero -- which is the statement that a functional of
    ``|m|`` cannot produce a torque, and the check that the rotation, the ``tau``
    projection and the sign convention all line up with the collinear code.
    """
    from pypresso.basis.fft import r_to_g
    from pypresso.scf.potential import _noncollinear_meta_exchange, meta_exchange

    system, pseudos, _ = _converged(qe_testsuite, pseudo_dir, "tb09")
    calculation = Calculation(system, pseudos)
    gvectors, cell = calculation.basis.dense, calculation.system.cell
    functional = calculation.functional
    charge, moment, up, down, tau_up, tau_down = _test_fields(gvectors.grid)

    collinear_rho = jnp.asarray(np.stack([up, down]))
    collinear_tau = jnp.asarray(np.stack([tau_up, tau_down]))
    collinear_g = jax.vmap(r_to_g, in_axes=(0, None))(collinear_rho, gvectors.fft_index)
    v, c = meta_exchange(collinear_rho, collinear_g, collinear_tau, gvectors, cell,
                         functional)

    zeros = np.zeros_like(charge)
    v_nc, c_nc = _noncollinear_meta_exchange(
        jnp.asarray(np.stack([charge, zeros, zeros, moment])), gvectors, cell,
        functional, None,
        jnp.asarray(np.stack([tau_up + tau_down, zeros, zeros, tau_up - tau_down])),
        np.array([0.0, 0.0, 1.0]),
    )

    assert float(c_nc) == float(c)
    assert float(jnp.abs(v_nc[0] - 0.5 * (v[0] + v[1])).max()) < 1.0e-13
    assert float(jnp.abs(v_nc[3] - 0.5 * (v[0] - v[1])).max()) < 1.0e-13
    assert float(jnp.abs(v_nc[1]).max()) == 0.0
    assert float(jnp.abs(v_nc[2]).max()) == 0.0


@pytest.mark.parametrize("axis", [(1.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.3, -0.7, 0.5)])
def test_the_noncollinear_potential_rotates_with_the_magnetization(
    qe_testsuite, pseudo_dir, axis
):
    """Turn ``m``, and ``v`` turns with it -- nothing is wired to ``z``.

    The scalar part is invariant, the vector part has the same length along the
    new axis as it had along ``z``, and there is no transverse component. This
    is the property that a hard-coded component or a mislaid sign breaks and
    that the ``z``-axis test above cannot see.
    """
    from pypresso.scf.potential import _noncollinear_meta_exchange

    system, pseudos, _ = _converged(qe_testsuite, pseudo_dir, "tb09")
    calculation = Calculation(system, pseudos)
    gvectors, cell = calculation.basis.dense, calculation.system.cell
    charge, moment, _, _, tau_up, tau_down = _test_fields(gvectors.grid)

    def evaluate(direction):
        direction = np.asarray(direction, dtype=float)
        direction = direction / np.linalg.norm(direction)
        rho = np.stack([charge] + [component * moment for component in direction])
        tau = np.stack(
            [tau_up + tau_down]
            + [component * (tau_up - tau_down) for component in direction]
        )
        v, c = _noncollinear_meta_exchange(
            jnp.asarray(rho), gvectors, cell, calculation.functional, None,
            jnp.asarray(tau), direction,
        )
        return np.asarray(v), float(c), direction

    reference, c_reference, _ = evaluate((0.0, 0.0, 1.0))
    rotated, c_rotated, direction = evaluate(axis)
    parallel = np.tensordot(direction, rotated[1:], axes=(0, 0))
    transverse = rotated[1:] - direction[:, None, None, None] * parallel

    assert c_rotated == pytest.approx(c_reference, abs=1.0e-14)
    assert np.abs(rotated[0] - reference[0]).max() < 1.0e-13
    assert np.abs(parallel - reference[3]).max() < 1.0e-13
    assert np.abs(transverse).max() < 1.0e-14


def test_paw_recovers_the_all_electron_coefficient(qe_testsuite, pseudo_dir):
    """PAW measures ``c = 1.11`` where the norm-conserving dataset measures 1.00.

    The point of the one-centre terms in one number. Tran and Blaha's ``c``
    averages ``|grad rho|/rho`` over the cell, and that ratio is largest in the
    core -- which a norm-conserving pseudopotential has removed and a PAW
    augmentation charge puts back. The published all-electron value for silicon
    is **1.12**; this cell gives 1.107 with PAW and 1.000 with
    ``Si.pz-vbc``, on the same grid and cutoff.

    The gap follows: 0.589 -> 1.285 eV with PAW against 0.645 -> 1.163 with
    norm-conserving, where experiment is 1.17.
    """
    paw_system, paw_pseudos = _system(
        qe_testsuite, pseudo_dir, input_dft="tb09", ecutwfc=12.0,
    )
    paw_pseudos = (read_upf(pseudo_dir / "Si.pz-n-kjpaw_psl.0.1.UPF"),)
    with pytest.warns(RuntimeWarning):
        paw = run_scf(paw_system, paw_pseudos, nbnd=10, conv_thr=1.0e-9,
                      max_iterations=120, tstress=False)

    assert paw.converged
    assert paw.meta_c == pytest.approx(1.107, abs=0.02)
    # ...and it is above the norm-conserving one, which is the whole claim.
    _, _, norm_conserving = _converged(qe_testsuite, pseudo_dir, "tb09")
    assert paw.meta_c > norm_conserving.meta_c + 0.05


def test_forces_and_stress_are_refused(qe_testsuite, pseudo_dir):
    """No energy, no derivative of one.

    ``tstress = .true.`` is in ``scf.in``, so this also checks that an optional
    diagnostic degrades to a warning rather than failing the run -- QE's own
    convention for a combination it cannot do.
    """
    from pypresso.forces import compute_forces
    from pypresso.stress import compute_stress
    from pypresso.forces.energy import state_from_result

    system, pseudos, result = _converged(qe_testsuite, pseudo_dir, "tb09",
                                         tstress=None)
    assert result.stress is None  # tstress in the input, warned and skipped

    calculation = Calculation(system, pseudos)
    state = state_from_result(result)
    for compute in (compute_forces, compute_stress):
        with pytest.raises(NotImplementedError, match="not the derivative of an energy"):
            compute(calculation, state)

    # **And the analytic route, which this test did not cover and which had no
    # check at all.** ``analytic_forces`` is a transcription of QE's six
    # expressions and never touches ``energy_at``, where the refusal lives, so
    # ``method='analytic'`` under a potential-only functional came back with a
    # force -- smooth, translation-corrected, symmetrised and meaningless.
    # ``compute_forces`` makes the check before dispatching now.
    with pytest.raises(NotImplementedError, match="not the derivative of an energy"):
        compute_forces(calculation, state, method="analytic")


@pytest.mark.parametrize(
    ("label", "overrides", "pseudo"),
    [("ultrasoft", {}, "Si.pz-n-rrkjus_psl.0.1.UPF")],
)
def test_unsupported_regimes_are_refused_by_name(
    qe_testsuite, pseudo_dir, label, overrides, pseudo
):
    """What is still not written, refused before any work is done.

    Ultrasoft, and **only** ultrasoft: the augmentation charge corrects the
    density with no kinetic counterpart anywhere, so a plain ultrasoft dataset
    has nothing to reconstruct ``tau`` from inside the sphere. A PAW one has the
    partial waves and is supported (P32).

    Neither of ``pw.x``'s two meta-GGA refusals is reproduced whole:
    ``PW/src/setup.f90`` stops on "Meta-GGA not implemented with USPP/PAW" and
    on "Non-collinear Meta-GGA not implemented", and here PAW works (P32) and so
    does noncollinear magnetism with spin-orbit coupling (P31).
    """
    system, pseudos = _system(qe_testsuite, pseudo_dir, input_dft="tb09", **overrides)
    if pseudo is not None:
        pseudos = (read_upf(pseudo_dir / pseudo),)
    with pytest.raises(NotImplementedError):
        Calculation(system, pseudos)
