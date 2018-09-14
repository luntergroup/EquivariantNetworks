#!/bin/bash
#$ -S /bin/bash
#$ -q gpu8.q

cd /users/lunter/bras3910/paper_revisions/EquivariantNetworks
module load python/3.4.6-gcc4.8.2
module load gcc/4.8.2-devtoolset-2
module load cuda/8.0
export LD_LIBRARY_PATH="/apps/well/cudnn/8.0-linux-x64-v5.1/lib64:${LD_LIBRARY_PATH}"

python sim_train.py params/small_data/sim_asym_no_drop_0.5_dat.json 50
# Run from command line with
 
