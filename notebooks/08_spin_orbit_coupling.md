# Spin-orbit coupling

`noncolin = .true.` makes a wavefunction a two-component spinor of length `2 npwx`, so
there is **one** Hamiltonian on a space twice as large rather than two Hamiltonians;
`lspinorb = .true.` puts the `j`-resolved projectors of a fully-relativistic dataset into
it. Three platinum benchmarks — ultrasoft LDA, ultrasoft PBE and PAW — match Quantum
ESPRESSO to **≤1.3e-8 Ry**, and bismuthene's half-electronvolt gap is made of nothing but
the coupling.

**Keep three numbers apart**, because collapsing them is the mistake that makes a
spin-orbit run allocate a magnetization it does not have: `nspin` says which regime (1, 2
or 4), `npol` how many components a *wavefunction* has, and `nspin_mag` how many a
*density* has — which is **one** for a nonmagnetic spin-orbit run, exactly as for an
unpolarized one.

A state is a two-component spinor and the nonlocal term is a matrix in spin space, built
from the $j$-resolved projectors of the dataset:

$$|\psi_{n\mathbf k}\rangle =
  \begin{pmatrix} \psi^{\uparrow}_{n\mathbf k} \\[2pt]
                   \psi^{\downarrow}_{n\mathbf k} \end{pmatrix},
\qquad
\hat V_{\rm NL} = \sum_{I}\sum_{ij}\sum_{\alpha\beta}
   D^{I,\,\alpha\beta}_{ij}\;
   |\beta^I_i\rangle\langle\beta^I_j| \otimes |\alpha\rangle\langle\beta|$$

$D^{\alpha\beta}_{ij}$ is assembled from the Clebsch-Gordan coefficients that couple
$(l, m_l, \sigma)$ to $(j, m_j)$ -- QE's `fcoef` -- which is the whole of where the
coupling enters.

Phase P14.


```python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pypresso import Calculator
from pypresso.io import read_qe_output
from pypresso.pseudo import read_upf
from pypresso.units import RY_TO_EV

QE = Path("../quantum_espresso/qe-7.5-ReleasePack/qe-7.5/test-suite")
PSEUDO, CASES = Path("../tests/data/pseudo"), Path("../tests/data/qe")
plt.rcParams.update({"figure.dpi": 120, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3})


def load(path):
    return Calculator.from_file(path, pseudo_dir=PSEUDO, announce=False,
                                conv_thr=1e-10)


# A fully-relativistic dataset keeps two projectors where a scalar one keeps their average:
# j = l - 1/2 and j = l + 1/2, which is where the coupling physically lives.
relativistic = read_upf(PSEUDO / "Bi.rel-pbe-dn-rrkjus_psl.1.0.0.UPF")
scalar = read_upf(PSEUDO / "Bi.pbe-dn-rrkjus_psl.1.0.0.UPF")
for name, pseudo in (("Bi.rel-pbe-dn-rrkjus", relativistic), ("Bi.pbe-dn-rrkjus", scalar)):
    print("%-22s %2d projectors  has_so=%-5s  (l, j) = %s"
          % (name, pseudo.nbeta, pseudo.has_so,
             [(p.l, p.j) for p in pseudo.projectors]))

fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0))
for ax, l, label in ((axes[0], 1, "6p"), (axes[1], 2, "5d")):
    seen = set()
    for projector in relativistic.projectors:
        if projector.l != l or projector.j in seen:
            continue
        seen.add(projector.j)
        ax.plot(relativistic.r, projector.beta, label="j = %g" % projector.j)
    beta = next(p for p in scalar.projectors if p.l == l)
    ax.plot(scalar.r, beta.beta, "k--", lw=1.0, label="scalar-relativistic")
    ax.set_xlim(0, 4.0); ax.set_xlabel("r [bohr]")
    ax.set_ylabel(r"$r\,\beta(r)$"); ax.set_title("Bi %s:  l = %d" % (label, l))
    ax.legend(fontsize=8)
fig.suptitle("Fully-relativistic projectors split by j; the scalar one is their average",
             fontsize=10)
fig.tight_layout()
```

    Bi.rel-pbe-dn-rrkjus   10 projectors  has_so=True   (l, j) = [(0, 0.5), (0, 0.5), (1, 0.5), (1, 1.5), (1, 0.5), (1, 1.5), (2, 1.5), (2, 2.5), (2, 1.5), (2, 2.5)]
    Bi.pbe-dn-rrkjus        6 projectors  has_so=False  (l, j) = [(0, None), (0, None), (1, None), (1, None), (2, None), (2, None)]



    
![png](08_spin_orbit_coupling_files/08_spin_orbit_coupling_1_1.png)
    


## The identity that gates the whole spinor path

Before any spin-orbit number is believed: run a spinor calculation with a *scalar*
dataset, where the two components cannot mix, and the collinear answer must come back
term by term with every eigenvalue simply doubled. This is silicon, `noncolin = .true.`
and nothing else changed.


```python
import tempfile

text = (QE / "pw_scf" / "scf.in").read_text()
marker = text.lower().index("&system") + len("&system")
smeared = (text[:marker]
           + "\n    occupations='smearing', smearing='gaussian', degauss=0.02,\n"
           + text[marker:])
work = Path(tempfile.mkdtemp())
(work / "collinear.in").write_text(smeared)
(work / "noncollinear.in").write_text(
    smeared[:marker] + "\n    noncolin = .true.,\n" + smeared[marker:])

collinear_calc = load(work / "collinear.in")
spinor_calc = load(work / "noncollinear.in")
for label, s in (("collinear   ", collinear_calc.system),
                 ("noncollinear", spinor_calc.system)):
    print("%s:  nspin=%d npol=%d nspin_mag=%d"
          % (label, s.nspin, s.npol, s.nspin_mag))
print("             ^ nspin_mag is one: there is no magnetization to carry")

collinear = collinear_calc.get_scf(max_iterations=80)
spinor = spinor_calc.get_scf(max_iterations=80)
print("\n%14s %18s %18s %12s" % ("term", "collinear", "noncollinear", "difference"))
for term, value in collinear.energy_terms.items():
    print("%14s %18.12f %18.12f %12.1e"
          % (term, value, spinor.energy_terms[term], spinor.energy_terms[term] - value))
print("%14s %18.12f %18.12f %12.1e"
      % ("TOTAL", collinear.total_energy, spinor.total_energy,
         spinor.total_energy - collinear.total_energy))

doubled = np.repeat(collinear.eigenvalues, 2, axis=1)
n = doubled.shape[1]
print("\neigenvalues: %d bands -> %d, max |difference from doubling| = %.2e eV"
      % (collinear.eigenvalues.shape[1], spinor.eigenvalues.shape[1],
         np.abs(spinor.eigenvalues[:, :n] - doubled).max() * RY_TO_EV))
```

    collinear   :  nspin=1 npol=1 nspin_mag=1
    noncollinear:  nspin=4 npol=2 nspin_mag=1
                 ^ nspin_mag is one: there is no magnetization to carry


    
              term          collinear       noncollinear   difference
      one-electron     4.833721601574     4.833721601574     -6.2e-15
           hartree     1.084391608208     1.084391608208      2.4e-15
                xc    -4.812850203017    -4.812850203017      0.0e+00
             ewald   -16.899758577223   -16.899758577223      0.0e+00
          smearing    -0.000000000418    -0.000000000418     -1.0e-19
             TOTAL   -15.794495570876   -15.794495570876     -3.6e-15
    
    eigenvalues: 8 bands -> 16, max |difference from doubling| = 5.44e-14 eV


## Platinum, against Quantum ESPRESSO

Three of QE's own benchmarks, on the three kinds of dataset that carry `j`. Note the
k-point weights: they sum to **one**, not two — a spinor band holds one electron.


```python
rows = []
for name, stem, label in (("spinorbit.in", "pw_spinorbit-spinorbit", "ultrasoft, LDA"),
                          ("spinorbit-pbe.in", "pw_spinorbit-spinorbit-pbe",
                           "ultrasoft, PBE"),
                          ("spinorbit-paw.in", "pw_spinorbit-spinorbit-paw", "PAW, PBE")):
    reference = read_qe_output(CASES / f"reference.out.{stem}")
    result = load(QE / "pw_spinorbit" / name).get_scf(max_iterations=100)
    eigen = np.abs(np.asarray(result.eigenvalues) * RY_TO_EV
                   - np.squeeze(np.asarray(reference.eigenvalues))).max()
    rows.append((label, result, reference, eigen))
    if name == "spinorbit.in":
        pt = result

print("%16s %18s %18s %10s %12s"
      % ("dataset", "pypresso (Ry)", "QE 7.5 (Ry)", "dE", "max de (eV)"))
for label, result, reference, eigen in rows:
    print("%16s %18.9f %18.9f %10.1e %12.1e"
          % (label, result.total_energy, reference.total_energy,
             result.total_energy - reference.total_energy, eigen))

levels = np.asarray(pt.eigenvalues)
print("\nmax Kramers splitting over all k: %.1e eV"
      % (np.abs(levels[:, 0::2] - levels[:, 1::2]).max() * RY_TO_EV))
print("(inversion and time reversal both hold, so every level is doubly degenerate)")
```

             dataset      pypresso (Ry)        QE 7.5 (Ry)         dE  max de (eV)
      ultrasoft, LDA      -69.491529507      -69.491529520    1.3e-08      3.6e-04
      ultrasoft, PBE      -90.199533906      -90.199533910    3.8e-09      2.4e-04
            PAW, PBE     -753.342691622     -753.342691630    8.4e-09      1.0e-04
    
    max Kramers splitting over all k: 1.1e-13 eV
    (inversion and time reversal both hold, so every level is doubly degenerate)


## Forces and the stress of a spinor

Until P46 none of the above could be differentiated: `noncolin` had a validated ground state
and no force, no stress and no relaxation. `GAPS.md` sized the fix at two substitutions —
the nonlocal quadratic form with `dvan_so` and the orthonormality constraint with `qq_so`,
both complex $2\times2$ matrices in spin space. That was right about the physics and wrong
about the size, for a reason this project states as a convention: **`nspin`, `npol` and
`nspin_mag` are three different numbers.** A spinor is *one* coefficient vector of length
$2n_{\rm pw}$, so the frozen state is `(1, nk, nbnd, 2 npwx)` and even the kinetic term has
to read the coefficient vector's own $|k+G|^2$. The layout was the larger half.

`dvan_so` is the **bare** $D$, for the same reason the collinear branch takes `dion` and not
`deeq`: `newd_nc` sandwiches the self-consistent $\int V_{\rm eff} Q_{ij}$ between `fcoef`
and *adds* it, so the split survives one spin index up. Taking `deeq_nc` double-counts — and
the only thing that catches it is the identity that `energy_at` reproduces the SCF total,
because a finite difference of the wrong functional agrees with its own gradient perfectly.

The figure is that identity's derivative: the force from one `jax.grad` against a central
difference of the *converged* energy, along a bond of doubled fcc platinum.


```python
platinum = Calculator.from_file(CASES / "pt2-soc-force.in", pseudo_dir=PSEUDO,
                                announce=False, conv_thr=1e-10)
p0 = np.asarray(platinum.system.structure.positions)

shifts = np.array([0.0, 0.04, 0.08, 0.12, 0.16])
energies, forces = [], []
for u in shifts:
    moved = p0.copy()
    moved[1, 0] += u
    here = platinum.with_positions(moved)
    energies.append(here.get_scf().total_energy)
    forces.append(float(np.asarray(here.get_forces().forces)[1, 0]))
energies, forces = np.array(energies), np.array(forces)

# -dE/du by a central difference of the converged energies, at the interior points.
fd = -(energies[2:] - energies[:-2]) / (shifts[2:] - shifts[:-2])
print("   u      F (autodiff)     -dE/du        difference")
for u, f, d in zip(shifts[1:-1], forces[1:-1], fd):
    print("%5.2f    %+11.6f    %+11.6f    %9.1e" % (u, f, d, abs(f - d)))

fig, ax = plt.subplots(figsize=(4.4, 3.2))
ax.plot(shifts, forces, "-o", ms=4, label=r"$F$, one $\nabla$ of the energy")
ax.plot(shifts[1:-1], fd, "kd", ms=7, mfc="none", mew=1.2,
        label=r"$-dE/du$, converged runs")
ax.axhline(0.0, lw=0.8, color="0.6")
ax.set_xlabel("displacement of atom 2 along $a_1$  [bohr]")
ax.set_ylabel("force on atom 2  [Ry/bohr]")
ax.set_title("Pt$_2$ with spin-orbit coupling:\nthe spinor force is the energy's derivative")
ax.legend(fontsize=8)
fig.tight_layout()
plt.show()
```

       u      F (autodiff)     -dE/du        difference
     0.04      +0.046206      +0.046195      1.1e-05
     0.08      +0.040065      +0.040081      1.5e-05
     0.12      +0.033999      +0.034043      4.4e-05



    
![png](08_spin_orbit_coupling_files/08_spin_orbit_coupling_7_1.png)
    


The two agree to about $10^{-5}$ Ry/bohr, which is the central difference's own floor at
this step — and neither side involves Quantum ESPRESSO. Against `pw.x`, which computes both
quantities for a spinor run, the force agrees to **8.9e-7 Ry/bohr** on a four-atom
noncollinear hydrogen chain, **7.5e-6** on this ultrasoft platinum and **7.3e-7** on its PAW
twin, with the stress on those three plus the three `pw_spinorbit` cases above to
**$\le 1.2\times10^{-6}$ Ry/bohr³**. Three of those needed no new reference: QE's own
spin-orbit inputs already carry `tstress`, so a PAW spin-orbit stress had been sitting there
all along and the refusal was the only thing in the way.

Relaxation came with the force and needed nothing — P15's BFGS puts this displaced platinum
back in **8 ionic steps**, to `max|F|` = 8.96e-5 Ry/bohr and the two atoms 0.499945 of a cell
apart, against the half symmetry requires. What is asserted there is the *separation*:
subtracting the mean force leaves a rigid translation of the whole crystal free, and the
optimizer uses it.

**What stays refused, each naming its own missing term.** `method='analytic'` — QE's
`force_us`/`stres_knl` are transcriptions with no spinor form. Anything through the
Sternheimer solver, so no spinor phonons. The elastic constants and electrostriction, which
reach the energy functional *directly* rather than through the Sternheimer guard — which is
why the spinor path is opt-in (`spinors=True`) rather than merely allowed, since deleting
the refusal would have opened a third derivative for a regime whose first-order
wavefunctions do not exist. And the force on an atom of a **spin spiral**: its two components
live on different plane-wave spheres, so the nonlocal term needs the projectors of both, and
$dE/dq$ is what a spiral has instead.

## Bismuthene: a gap made entirely of spin-orbit coupling

Two bismuth atoms in a honeycomb layer. Without the coupling the bands cross near K in a
Dirac point; with it they do not, and what opens is a gap of about half an electronvolt —
the quantum spin Hall insulator whose invariant notebook 10 computes.

Run at the test size (20 Ry, 6x6x1). The converged pair (35 Ry, 12x12x1) is committed
beside it with its own QE reference and takes about forty minutes at a peak of 9.4 GB;
the small pair is what the regression tests check.


```python
SIZE = "-small"
CORNERS = {"Gamma": 0, "M": 4, "K": 7, "Gamma'": 12}

results, bands = {}, {}
for tag in ("nosoc", "soc"):
    calc = load(CASES / f"bismuthene-{tag}{SIZE}.in")
    system = calc.system
    results[tag] = calc.get_scf(max_iterations=100)
    # Only the k-path comes from the bands input; the density is already here,
    # and so is the Fermi level the plot puts at zero.
    path_system = load(CASES / f"bismuthene-{tag}{SIZE}-bands.in").system
    bands[tag] = calc.get_bands(kpoints=path_system.kpoints, nbnd=path_system.nbnd)
    reference = read_qe_output(CASES / f"reference.out.bismuthene-{tag}{SIZE}")
    nelec = sum(calc.pseudos[t].z_valence for t in system.structure.types)
    occupied = int(round(nelec / (1 if system.nspin == 4 else 2)))
    levels = bands[tag].eigenvalues_ev
    direct = levels[:, occupied] - levels[:, occupied - 1]
    print("%6s: nspin=%d npol=%d, E = %.9f Ry (QE %.1e), smallest direct gap %.4f eV"
          % (tag, system.nspin, system.npol, results[tag].total_energy,
             results[tag].total_energy - reference.total_energy, direct.min()))
```

     nosoc: nspin=1 npol=1, E = -296.198423399 Ry (QE -3.4e-05), smallest direct gap 0.1361 eV


       soc: nspin=4 npol=2, E = -295.610317533 Ry (QE -3.5e-05), smallest direct gap 0.6295 eV



```python
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharey=True)
for ax, tag, title in ((axes[0], "nosoc", "scalar-relativistic: no spin-orbit"),
                       (axes[1], "soc", "fully relativistic: with spin-orbit")):
    x = bands[tag].path_length
    ax.plot(x, bands[tag].eigenvalues_ev - results[tag].fermi_energy * RY_TO_EV,
            color="#1f77b4", lw=1.0)
    ax.axhline(0.0, color="0.4", lw=0.8, ls="--")
    corners = list(CORNERS.values())
    for c in corners[1:-1]:
        ax.axvline(x[c], color="0.7", lw=0.8)
    ax.set_xticks([x[c] for c in corners])
    ax.set_xticklabels([r"$\Gamma$", "M", "K", r"$\Gamma$"])
    ax.set_xlim(x[0], x[-1]); ax.set_ylim(-4.0, 4.0); ax.set_title(title, fontsize=9)
axes[0].set_ylabel(r"$E - E_F$  [eV]")
fig.suptitle("Bismuthene: the spin-orbit coupling gaps the Dirac point at K", fontsize=10)
fig.tight_layout()
```


    
![png](08_spin_orbit_coupling_files/08_spin_orbit_coupling_11_0.png)
    


---
**The detail:** `PLAN.md` §3 P14, and P46 for the forces and the stress — `fcoef` and why `init_us_1` zeroes its cross-radial
entries *after* building `dvan_so` (one array used for both is a correct `dvan_so` and a
silently wrong `qq_so`), `transform_qq_so`, and `vloc_psi_nc`.
**The tests:** `tests/regression/test_spinorbit.py`,
`tests/unit/test_spinorbit_coefficients.py`.
