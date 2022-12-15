#!/bin/bash

#SBATCH -p spgpu
#SBATCH --account=ahowens1
#SBATCH --job-name=flow
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --error=error_zs_itm_patches_declaration_clipmod_k_20.txt


#SBATCH --gres=gpu:1
#SBATCH --time=00-20:00:00
#SBATCH --mem-per-cpu=11G

# The job command(s):
source ~/.bashrc
conda activate vqa                                        #TODO

python zs_itm_patches_declaration_clipmod.py > out_zs_itm_patches_declaration_clipmod_k_20.txt
