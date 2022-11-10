#!/bin/bash

for i in {1..9}
do
    echo Make folder ${i}_ai4mat
    cp -r template_ai4mat ${i}_ai4mat
    cd ${i}_ai4mat

    for f in janus_ctrl janus_chimera janus_hv janus_random
    do
    	echo Submitting ${i} run for ${f}
        cd $f
        sbatch janus_job
        cd ..
    done
    cd ..

done
