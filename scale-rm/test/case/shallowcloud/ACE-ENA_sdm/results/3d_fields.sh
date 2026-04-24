#!/bin/bash
#PBS -q SQUID
#PBS --group=hp250136 
#PBS -b 1 
#PBS -l cpunum_job=40
#PBS -l elapstim_req=24:00:00
#PBS -N 3d_fields
#PBS -M yinchongzhi@gmail.com

source /etc/profile.d/modules.sh
cd ${PBS_O_WORKDIR}

#-----------program execution------------

module purge
source ~/miniforge3/etc/profile.d/conda.sh
conda activate sdm_env

# run
python ace-ena_3D_fields.py
