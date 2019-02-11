#!/bin/bash
#SBATCH -J firstjob 
#SBATCH -N 1

module purge
module load intel/18 python/3.6.4 
source /home/lv71235/mmatak/qnn-aas-py3/venv/bin/activate
python /home/lv71235/mmatak/qnn-aas-py3/launcher.py

