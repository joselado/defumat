"""A separate step length for the magnetization, VASP's ``AMIX_MAG``.

`pw.x` uses one ``alphamix`` for every component of ``mix_type``, and this is a
deliberate departure from that, so what the tests have to pin is narrow and
exact: **the magnetization's step changes and the charge's does not**. A knob
that quietly moved the charge would be a different operator wearing the same
name, and on a metal it is the charge's step that decides whether the run
converges at all.
"""
import numpy as np
import pytest

from defumat.scf.mixing import AndersonMixer, LinearMixer, get_mixer


def collinear(up, down):
    """``(up, down)``, which is how a collinear density is carried here."""
    return np.stack([np.full(4, up), np.full(4, down)])


def test_unset_is_pw_x_and_changes_nothing():
    """The default has to be a literal no-op, not a no-op to round-off."""
    mixer = LinearMixer(beta=0.3)
    mixer.shape = (2, 4)
    residual = np.arange(8, dtype=float)

    assert mixer.beta_mag is None
    assert np.array_equal(mixer.step(residual), 0.3 * residual)


def test_the_magnetization_moves_and_the_charge_does_not():
    """The whole claim, at ``nspin_mag = 4`` where the channels are direct."""
    mixer = LinearMixer(beta=0.3)
    mixer.shape = (4, 4)
    mixer.beta_mag = 0.9
    residual = np.arange(16, dtype=float)

    stepped = mixer.step(residual).reshape(4, 4)
    plain = (0.3 * residual).reshape(4, 4)

    assert np.allclose(stepped[0], plain[0])
    for channel in (1, 2, 3):
        assert np.allclose(stepped[channel], 3.0 * plain[channel])


def test_a_collinear_density_is_rotated_before_it_is_scaled():
    """``(up, down)`` is not ``(charge, magnetization)``.

    Scaling channel 1 of a collinear residual would scale *down*, which changes
    the charge as well and is a different operator. The pair has to be rotated,
    scaled and rotated back -- exactly what Kerker does around its own screening.
    The falsifier is a residual with **zero magnetization**: a correct rescale
    leaves it untouched whatever ``beta_mag`` is, and the naive one does not.
    """
    mixer = LinearMixer(beta=0.5)
    mixer.shape = (2, 4)
    mixer.beta_mag = 2.0

    # Pure charge: up == down, so the moment is zero and nothing may move.
    residual = collinear(1.0, 1.0).reshape(-1)
    assert np.allclose(mixer.step(residual), 0.5 * residual)

    # Pure moment: up == -down, so the charge is zero and everything scales.
    residual = collinear(1.0, -1.0).reshape(-1)
    assert np.allclose(mixer.step(residual), 2.0 * residual)

    # And a mixture separates correctly rather than approximately.
    residual = collinear(3.0, 1.0).reshape(-1)
    stepped = mixer.step(residual).reshape(2, 4)
    charge, moment = stepped[0] + stepped[1], stepped[0] - stepped[1]
    assert np.allclose(charge, 0.5 * 4.0)      # beta * (up + down)
    assert np.allclose(moment, 2.0 * 2.0)      # beta_mag * (up - down)


def test_a_nonmagnetic_run_is_untouched():
    mixer = LinearMixer(beta=0.3)
    mixer.shape = (1, 4)
    mixer.beta_mag = 0.9
    residual = np.arange(4, dtype=float)
    assert np.allclose(mixer.step(residual), 0.3 * residual)


def test_the_tail_beyond_the_density_keeps_the_charge_step():
    """``becsum``, ``ns`` and ``tau`` are not given the magnetic step.

    Giving them one would be a second departure with no measurement behind it.
    The tail is whatever the flat vector carries past ``prod(shape)``.
    """
    mixer = LinearMixer(beta=0.3)
    mixer.shape = (4, 2)
    mixer.beta_mag = 0.9
    residual = np.ones(8 + 5, dtype=float)

    stepped = mixer.step(residual)

    assert np.allclose(stepped[8:], 0.3)


def test_anderson_takes_it_through_its_own_step():
    """The rescale has to survive the extrapolation, not bypass it.

    Anderson's first iteration is a plain step, so it is the one that can be
    checked in closed form; what matters is that it goes through ``step`` at all,
    because a mixer that took its own scalar would silently ignore the knob.
    """
    mixer = AndersonMixer(beta=0.3, history=4)
    mixer.shape = (4, 2)
    mixer.beta_mag = 0.9

    rho_in = np.zeros((4, 2))
    rho_out = np.ones((4, 2))
    mixed = mixer.mix(rho_in, rho_out)

    assert np.allclose(mixed[0], 0.3)
    assert np.allclose(mixed[1:], 0.9)


def test_the_adaptive_mixer_is_refused_rather_than_ignored():
    """Its step is already one length per component, so a second is undefined.

    The flag is the same one that gates a preconditioner, because it is the same
    property: a step that is not one scalar times the residual.
    """
    assert get_mixer("adaptive").accepts_precondition is False
    assert get_mixer("anderson").accepts_precondition is True


def test_the_scale_is_exact_at_a_ratio_of_one():
    """``beta_mag == beta`` must reproduce ``pw.x`` bit for bit, not nearly.

    Otherwise a run with the knob set to its own default would differ from one
    without it, and every comparison against QE would carry an unstated term.
    """
    mixer = LinearMixer(beta=0.37)
    mixer.shape = (4, 3)
    residual = np.linspace(-1.0, 1.0, 12)

    plain = mixer.step(residual).copy()
    mixer.beta_mag = 0.37
    assert np.array_equal(mixer.step(residual), plain)
