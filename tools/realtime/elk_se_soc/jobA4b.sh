#!/bin/bash
# A4b: the timed pair on the 3x3x3 mesh (the c-field's C3 wedge, 15 points), one run at a time
# on core 0, after the user's Blender render (PID 92077) ends or at 23:50. Elk at nempty 16
# per atom with lorbcnd (the basis at which the in-pulse current agrees to 7 per cent on
# 2x2x2): task 0 timed, 450 alone, a 2-step 450+460 for the setup, 460 alone timed.
# defumat: se_time.py hxc, dt 0.125, 2000 steps, block 400. A silicon load control before
# and after. Then an nempty-8 Elk row on the same mesh.
d=/l/ladovj1/review/hspin-elk
E=/l/ladovj1/review/harmonics-elk/bin/elk
cd $d
source /l/ladovj1/venv/bin/activate
deadline=$(date -d "today 23:50" +%s)
while kill -0 92077 2>/dev/null && [ $(date +%s) -lt $deadline ]; do sleep 30; done
date -Is > $d/A4pair.start
$d/run_capped.sh a4-si-control $d python3 -u si_control.py
elkrow() {  # nempty tag
  local ne=$1 tag=$2 w=$d/$2-elk-n$1 s=$d/$2-elk-n$1-s2
  rm -rf $w; mkdir -p $w
  python3 write_elk.py $w/elk.in 0 3 --lorbcnd --nempty $ne --sppath $d/species/
  $d/run_capped.sh $tag-elk-gs-n$ne $d bash -c "cd $w && $E > gs.stdout 2>&1"
  python3 elk_gap.py $w/EIGVAL.OUT > $w/gap.json 2>&1
  cp $w/INFO.OUT $w/INFO-gs.OUT
  python3 write_elk.py $w/elk.in 450 3 --lorbcnd --nempty $ne --sppath $d/species/ --tstime 250.0 --dtimes 0.125
  $d/run_capped.sh $tag-elk-450-n$ne $d bash -c "cd $w && $E > t450.stdout 2>&1"
  rm -rf $s; cp -r $w $s
  python3 write_elk.py $s/elk.in 450,460 3 --lorbcnd --nempty $ne --sppath $d/species/ --tstime 0.25 --dtimes 0.125
  $d/run_capped.sh $tag-elk-460s2-n$ne $d bash -c "cd $s && $E > td.stdout 2>&1"
  python3 write_elk.py $w/elk.in 460 3 --lorbcnd --nempty $ne --sppath $d/species/ --tstime 250.0 --dtimes 0.125
  $d/run_capped.sh $tag-elk-460-n$ne $d bash -c "cd $w && $E > td.stdout 2>&1"
}
elkrow 16 a4
$d/run_capped.sh a4-defumat-hxc $d python3 -u se_time.py 20 3 0.125 2000 400 hxc $d/a4-defumat-hxc.npz
python3 compare_current.py $d/a4-defumat-hxc.npz $d/a4-elk-n16/JTOT_TD.OUT $d/a4-elk-n16/AFIELDT.OUT > $d/a4-compare-n16.json 2>&1
python3 coef.py $d/a4-defumat-hxc.npz $d/a4-elk-n16/JTOT_TD.OUT > $d/a4-coef-n16.txt 2>&1
$d/run_capped.sh a4-si-control-after $d python3 -u si_control.py
touch $d/A4pair.done
elkrow 8 a6
python3 coef.py $d/a4-defumat-hxc.npz $d/a4-elk-n16/JTOT_TD.OUT $d/a6-elk-n8/JTOT_TD.OUT > $d/a6-coef.txt 2>&1
touch $d/A6.done
