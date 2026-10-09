# Plan: the pooled density in G (OPEN.md Part XXIII item 26, the part left)

Written 2026-10-09 on branch `open-e`, for a fable review before any code. Nothing here is
implemented, and its first step is a cluster submission that is the user's to approve.

## What is done and what is left

Done at `d19a211` (2026-10-04): every pool computes the layout locally and one byte-packed
all-gather carries the eigenvalues, step counts and unsettled counts, 15 collectives an
iteration to 7, bit-identical per rank at two and three pools. Not done, by the entry's own
account:

1. **Packing the raw sums into one all-reduce** (smooth density, `becsum` per species, `tau`,
   `ns`). Bit-identical at two pools only: at three, gloo's CPU all-reduce adds in an order
   that depends on the element's position in the buffer, and 1839 of 25140 sums moved in the
   last bit.
2. **The G-space payload**: all-reduce the smooth density as its half-sphere coefficients and
   broadcast the mixed density in G rather than as the dense real-space box, which
   `multihost_utils.broadcast_one_to_all` turns into a full ring all-reduce of zeros on gloo.
   About 0.52 of a box, so roughly half the bytes, and on a dual-grid run the dense box is
   replaced by the smooth sphere.

## Why it is not done here

The bound is entirely a multi-node one: 868 ms per 83 MB all-reduce between two Triton nodes
on `eth0` (`PLAN.md` P124), against 25 ms an iteration, about 1.3 per cent, at six pools on
one machine (si32). D22 is one machine with 16 cores, so nothing measured there can move the
decision, and `ib0` (`DEFUMAT_POOL_INTERFACE=ib0`), unmeasured, may shrink the whole item.

## Steps

1. **Measure the wire first** (the user's approval, Triton): `tools/parallel/comm_bench.py
   --sizes-mb 40 83 110 216` across two Milan nodes, once on `eth0` and once with
   `DEFUMAT_POOL_INTERFACE=ib0`. If `ib0` brings 83 MB under about 100 ms, the payload work
   is worth less than one per cent on any cell on record and the item closes as measured.
2. **If it does not**: implement (2) above behind the existing `Pools` API.
   `JaxCommunicator.broadcast` of the mixed state sends `rho` in G on the smooth sphere
   (the mixer already holds it there when `mixing_space='g'`, Part XXIII item 22), and the
   all-reduce of the raw smooth density goes through `r_to_g` on each pool before the sum.
   Reassociating, so the acceptance is energies at a fixed iteration count to 1e-10 Ry
   between the real-space and G routes at two and three pools, not bit-identity.
3. **(1) at two pools only**, where it is exact, keyed on `pools.size == 2`, or not at all;
   the 1839-of-25140 last-bit move at three pools is the reason the general form is out.

## Measurement

`tools/parallel/pool_time.py` on a large-box pooled input across two nodes before and after,
the per-iteration collective time read from its log, and the energy at a fixed iteration
count compared between the arms.
