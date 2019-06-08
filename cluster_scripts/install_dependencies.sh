#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=Standard
#SBATCH --gres=gpu:1
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-07:59:59

export CUDA_HOME=/opt/cuda-9.0.176.1/

export CUDNN_HOME=/opt/cuDNN-7.0/

export STUDENT_ID=$(whoami)

export DATASET_DIR=${TMP}/datasets/

source /home/${STUDENT_ID}/miniconda3/bin/activate mlp

conda install -c conda-forge tensorflow 
pip install --upgrade torch
pip install tb-nightly
pip install future