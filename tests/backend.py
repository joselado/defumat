"""What a test may ask of the platform it runs on.

The push gate was first run with a card as JAX's backend on 2026-10-05
(``PLAN.md`` P131) and 19 tests failed without one wrong number among them:
they assumed the CPU. Two helpers cover the two kinds that are not about a
compiled program's text.

* :func:`on_a_cpu` pins :func:`defumat.batching._backend`, which every
  platform default reads when it is called, for a test whose claim **is** a
  CPU default (a dial, a route, a printed setting). On a CPU it changes
  nothing.
* :func:`assert_same` asserts the same bits on a CPU, where a refactor or a
  second call is promised to give them, and agreement to round-off on an
  accelerator, where a different reduction order or an atomic scatter-add
  with colliding indices does not repeat its last bit.
"""

import jax
import numpy as np


def on_a_cpu(monkeypatch) -> None:
    """Make every platform default resolve as it does on a CPU."""
    from defumat import batching

    monkeypatch.setattr(batching, "_backend", lambda: "cpu")


def assert_same(actual, expected, atol: float, err_msg: str = "") -> None:
    """Equal bit for bit on a CPU, within ``atol`` on an accelerator."""
    if jax.default_backend() == "cpu":
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected),
                                      err_msg=err_msg)
    else:
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected),
                                   rtol=0, atol=atol, err_msg=err_msg)
