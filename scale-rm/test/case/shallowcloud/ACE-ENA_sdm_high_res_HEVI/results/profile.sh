#!/bin/bash
#PBS -q SQUID
#PBS --group=hp250136 
#PBS -b 1 
#PBS -l cpunum_job=40
#PBS -l elapstim_req=24:00:00
#PBS -N profile
#PBS -M yinchongzhi@gmail.com

source /etc/profile.d/modules.sh
cd ${PBS_O_WORKDIR}

#-----------program execution------------

module purge
source ~/miniforge3/etc/profile.d/conda.sh

conda activate sdm_env

# run
python ace-ena_time_series_and_vertical_profiles.py
