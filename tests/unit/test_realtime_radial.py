"""The table of ``g_l(q^2)`` the real-time projectors are built from.

:mod:`defumat.realtime.radial` replaces the direct radial transform inside the
time loop by a Chebyshev series in ``q^2``, so it is checked against the
transform it replaces: the radial functions on a grid the nodes do not contain,
the projectors themselves on a cell at a shifted k-point, and the derivative a
current reads at ``k + G = 0``, where the table has no guard to lose.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defumat.io.pwin import read_pw_input
from defumat.pseudo import read_upf
from defumat.realtime.radial import column_layout, max_error, radial_table
from defumat.scf import Calculation
from defumat.system import build_system

pytestmark = pytest.mark.unit

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"


@pytest.mark.parametrize("name, ecut", [
    ("Si.pz-vbc.UPF", 12.0), ("C.pz-rrkjus.UPF", 40.0),
    ("Pt.pbe-n-kjpaw_psl.0.1.UPF", 40.0), ("As.pz-bhs.UPF", 30.0)])
def test_the_table_is_the_transform(pseudo_dir, name, ecut):
    """Within 2e-13 of the largest ``g_l`` on a grid off the nodes.

    Measured 1.1e-13, 1.4e-13, 4.0e-14 and 7.2e-14 for the four datasets on
    ``[0, (sqrt(ecut) + 1)^2]``, which is the transform's own round-off: the
    Chebyshev coefficients fall to about 1e-14 of the largest by the twentieth
    term and stay there.
    """
    pseudo = read_upf(pseudo_dir / name)
    table = radial_table((pseudo,), 270.0, (np.sqrt(ecut) + 1.0) ** 2)
    assert max_error(table, (pseudo,), 270.0) < 2e-13


def _calculation(pseudo_dir, case):
    system = build_system(read_pw_input(CASES / f"{case}.in"))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    return Calculation(system, pseudos)


def test_the_columns_sit_where_the_projector_core_puts_them(pseudo_dir):
    """The table's columns against ``ProjectorCore.columns`` on a two-species cell.

    ``column_layout`` reproduces ``build_projector_core``'s ordering by datasets
    and channels; were it to differ, the columns would be right numbers in the
    wrong places and the nonlocal potential scrambled, so the comparison is
    element by element on zincblende AlAs, whose two datasets have different
    channel counts.
    """
    calculation = _calculation(pseudo_dir, "alas-shg")
    core = calculation.projector_core
    radius = float(np.max(np.linalg.norm(np.asarray(core.kg), axis=-1)))
    table = radial_table(calculation.pseudos, calculation.system.cell.volume,
                         (radius + 0.5) ** 2)
    columns = np.asarray(table.columns(core.kg))
    mask = np.asarray(core.mask)[..., None]
    expected = np.asarray(core.columns)
    scale = np.abs(expected).max()
    assert np.abs(np.where(mask, columns - expected, 0.0)).max() < 1e-12 * scale
    datasets, beta_of, _, _ = column_layout(calculation.pseudos)
    assert len(beta_of) == expected.shape[-1] and len(datasets) == 2


def test_the_table_differentiates_at_the_origin_like_the_rewritten_rows(pseudo_dir):
    """Second derivative of the columns at ``k + G = 0``, table against core.

    The projector core's origin rows are a solid harmonic times the
    transform's Taylor series (``pseudo.projectors._with_origin_rows``), and the
    table is a solid harmonic times a Chebyshev series, so on that row the two
    are independent polynomial representations of one function and their
    second derivatives along ``x`` must agree.
    """
    calculation = _calculation(pseudo_dir, "si2-nosym")
    core = calculation.projector_core
    origin = np.argwhere(np.asarray(np.sum(core.kg * core.kg, axis=-1)) <= 1e-8)[0]
    ik, ig = int(origin[0]), int(origin[1])
    table = radial_table(calculation.pseudos, calculation.system.cell.volume, 30.0)
    from defumat.pseudo.projectors import (
        _origin_columns, _with_origin_rows)
    from defumat.realtime.radial import column_layout as layout

    datasets, beta_of, lm_of, l_of = layout(calculation.pseudos)
    series = _origin_columns(datasets, beta_of, l_of, calculation.system.cell.volume)
    lmax = max(p.lmax for p in datasets)
    guarded = jnp.asarray(core.columns[ik:ik + 1, ig:ig + 1])

    def by_table(x):
        kg = jnp.zeros((1, 1, 3)).at[0, 0, 0].set(x)
        return table.columns(kg)[0, 0]

    def by_series(x):
        kg = jnp.zeros((1, 1, 3)).at[0, 0, 0].set(x)
        return _with_origin_rows(guarded, kg, series, jnp.asarray(lm_of), lmax)[0, 0]

    second = lambda f: jax.jacfwd(jax.jacfwd(f))(0.0)
    a, b = np.asarray(second(by_table)), np.asarray(second(by_series))
    assert np.abs(a).max() > 1e-2, "the l = 0 curvature is not zero"
    np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12 * np.abs(a).max())


def test_a_second_evaluation_outside_jit_compiles_nothing(pseudo_dir):
    """The columns of a second k-chunk, evaluated eagerly, reuse the first one's programs.

    The setup of every k-chunk evaluates the table outside any ``jit``, and
    with the Clenshaw loop a ``fori_loop`` over a closure that compiled two
    ``jit(scan)`` programs a chunk (248 in an 8^3 third harmonic on a CPU).
    The counter is validated on the first evaluation, which must compile.
    """
    import logging

    pseudo = read_upf(pseudo_dir / "Si.pz-vbc.UPF")
    table = radial_table((pseudo,), 270.0, 25.0)
    rng = np.random.default_rng(0)
    chunks = [jnp.asarray(rng.normal(size=(2, 37, 3))) for _ in range(3)]
    count = [0]

    class Counter(logging.Handler):
        def emit(self, record):
            if "Finished XLA compilation" in record.getMessage():
                count[0] += 1

    handler = Counter()
    logger = logging.getLogger("jax")
    previous = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    jax.config.update("jax_log_compiles", True)
    counts = []
    try:
        jax.clear_caches()
        for kg in chunks:
            before = count[0]
            table.columns(kg).block_until_ready()
            counts.append(count[0] - before)
    finally:
        jax.config.update("jax_log_compiles", False)
        logger.removeHandler(handler)
        logger.setLevel(previous)
    assert counts[0] > 0, "the counter saw the first evaluation compile"
    assert counts[1] == counts[2] == 0, counts
