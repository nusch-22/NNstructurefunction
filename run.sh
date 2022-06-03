#!/bin/bash

# Replace the below with the run card name
runcard=runcard_2

cp template.sh template_run.sh
sed -i 's/runcard/'"$runcard"'/' template_run.sh

qsub -q smefit -W group_list=smefit -l walltime=03:00:00 -l nodes=1:ppn=4 -l pvmem=15gb template_run.sh
