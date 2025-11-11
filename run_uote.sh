#!/bin/bash
#SBATCH --job-name=UOTe
#SBATCH --time=04:00:00
#SBATCH --ntasks=4
#SBATCH --output=uote_%j.out
#SBATCH --error=uote_%j.err

module load quantum-espresso
python /shared/home/lun364/ap275_final_project_2025/uote_phonon.py
