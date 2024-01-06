#!/usr/bin/env bash

ml purge
ml "libGLU/9.0.0-fosscuda-2018b"

python3 run_render.py 0 2 &

#step_size=2
#for i in `seq 0 $step_size 50`;
#do
#python3 run_render.py $i $step_size &
#done
