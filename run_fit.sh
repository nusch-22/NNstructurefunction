#!/usr/bin/env bash

cd /data/theorie/nmortaza/NNstructurefunction

# Replace the below with the fitname and number of replicas
name=CHORUS_2
n_rep=1000

# create replicas
./create_replicas.py $name $n_rep

for id in $(seq 1 $n_rep)
do

cp template_fit.sh ./replicas/rep_$id.sh

sed -i 's/id/'"$id"'/' ./replicas/rep_$id.sh
sed -i 's/name/'"$name"'/' ./replicas/rep_$id.sh

qsub -q smefit -W group_list=smefit -l walltime=01:00:00 -l nodes=1:ppn=4 -l pvmem=15gb ./replicas/rep_$id.sh

done
