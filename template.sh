#!/bin/bash

source activate nnpdf40

export OMP_STACKSIZE=16000                                                                                                                                                                                
export OMP_NUM_THREADS=4                                                                                                                                                                                  
export KMP_BLOCKTIME=0                                                                                                                                                                                    
export KMP_AFFINITY=granularity=fine,verbose,compact,1,0

# Replace the below with the path to where your
# nnpdf40 run card is located
cd /data/theorie/nmortaza/NNstructurefunction
./run_hyperopt.py runcard.yaml 
