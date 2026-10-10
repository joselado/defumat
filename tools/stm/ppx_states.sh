#!/bin/bash
# Every single-state density of al-metal-nosym through pp.x (plot_num = 7, one
# run, 64 x 6 files psi_K###_B###), and the zero-bias STM image (plot_num = 5),
# for tools/stm/ppx_state_sum.py. RUN must hold a pw.x run of
# tests/data/qe/al-metal-nosym.in with prefix='al', outdir='./out'.
#
# usage: RUN=<dir> PPX=<path to pp.x> bash tools/stm/ppx_states.sh
set -e
cd "$RUN"
cat > pp7.in <<EOF
 &inputpp
 prefix='al', outdir='./out', filplot='psi', plot_num=7,
 kpoint(1)=1, kpoint(2)=64, kband(1)=1, kband(2)=6, lsign=.false.
 /
EOF
cat > pp5.in <<EOF
 &inputpp
 prefix='al', outdir='./out', filplot='al.stm.b20', plot_num=5, sample_bias=0.0
 /
EOF
export OMP_NUM_THREADS=1
"$PPX" < pp7.in > pp7.out 2>&1
"$PPX" < pp5.in > pp5.out 2>&1
ls psi_K* | wc -l
