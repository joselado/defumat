#!/bin/bash
# A7: Elk's setup inside task 460 on the 3x3x3 mesh, measured with the same k-set as the
# timed run. A 2-step run with the pulse of the timed one is reduced with the crystal's
# whole D3 (9 points), since findsymlat checks A(t) only over the steps it runs and the
# pulse is 1e-11 of its peak there; here the pulse peaks at t = 0 with a 90-degree phase,
# so A(0) = A0 along c and the group is C3, 15 points, as in the timed run.
d=/l/ladovj1/review/hspin-elk
E=/l/ladovj1/review/harmonics-elk/bin/elk
cd $d
source /l/ladovj1/venv/bin/activate
until [ -f $d/A6.done ]; do sleep 30; done
for ne in 16 8; do
  src=$d/a4-elk-n$ne; [ $ne = 8 ] && src=$d/a6-elk-n8
  s=$d/a7-elk-n$ne-s2; rm -rf $s; mkdir -p $s
  cp $src/STATE.OUT $s/
  python3 write_elk.py $s/elk.in 450 3 --lorbcnd --nempty $ne --sppath $d/species/ --tstime 0.25 --dtimes 0.125 --peak 0 --phase 90
  $d/run_capped.sh a7-elk-450s2-n$ne $d bash -c "cd $s && $E > t450.stdout 2>&1"
  python3 write_elk.py $s/elk.in 460 3 --lorbcnd --nempty $ne --sppath $d/species/ --tstime 0.25 --dtimes 0.125 --peak 0 --phase 90
  $d/run_capped.sh a7-elk-460s2-n$ne $d bash -c "cd $s && $E > td.stdout 2>&1"
done
touch $d/A7.done
