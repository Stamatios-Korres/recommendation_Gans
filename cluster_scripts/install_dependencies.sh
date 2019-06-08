#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=Standard
#SBATCH --gres=gpu:1
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-07:59:59

source /home/${STUDENT_ID}/miniconda3/bin/activate mlp

conda install -c conda-forge tensorflow 
pip install --upgrade torch
pip install tb-nightly
pip install future