"""A pytest plugin that prints every component the spectrum-versus-orders test compares, so a pass
carries its numbers. Loaded with -p svo_report; the test itself is unchanged."""
import time

from defumat import Calculator

KEYS = ((1, 1), (2, 2), (2, 0), (3, 3), (3, 1))
_last = {}


def _report():
    spectrum, orders = _last["get_nonlinear_spectrum"], _last["get_harmonic_orders"]
    refs = {k: complex(orders.component(*k, axis=0)) for k in KEYS}
    for k in KEYS:
        value = complex(spectrum.component(*k, axis=0)[0])
        scale = max(abs(v) for kk, v in refs.items() if kk[0] == k[0])
        print(f"svo {k} hierarchy {value:.6e} orders {refs[k]:.6e} "
              f"rel {abs(value - refs[k]) / scale:.3e}", flush=True)
    print("svo norm drift", orders.norm_drift, "iterations", spectrum.iterations, flush=True)


def _wrap(name):
    original = getattr(Calculator, name)

    def wrapped(self, *args, **kwargs):
        start = time.time()
        result = original(self, *args, **kwargs)
        print(f"svo {name} {time.time() - start:.1f} s", flush=True)
        _last[name] = result
        if name == "get_harmonic_orders":
            _report()
        return result

    setattr(Calculator, name, wrapped)


_wrap("get_nonlinear_spectrum")
_wrap("get_harmonic_orders")
