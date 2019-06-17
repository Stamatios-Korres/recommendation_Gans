#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=Interactive
#SBATCH --gres=gpu:1
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-01:59:59

export CUDA_HOME=/opt/cuda-9.0.176.1/

export CUDNN_HOME=/opt/cuDNN-7.0/

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

mkdir -p /disk/scratch/${STUDENT_ID}

export TMPDIR=/disk/scratch/${STUDENT_ID}/

export TMP=/disk/scratch/${STUDENT_ID}/

mkdir -p ${TMP}/datasets/

export DATASET_DIR=${TMP}/datasets/

rsync -ua --progress /home/${STUDENT_ID}/recommendations/datasets/movielens/ /disk/scratch/${STUDENT_ID}/datasets/movielens

source /home/${STUDENT_ID}/miniconda3/bin/activate mlp

echo 'Activated mlp'

cd /home/${STUDENT_ID}/recommendations/

echo "Changed to recommendation folder. Calling python"

python3 slate_generation.py   --use_gpu "True"  \
                              --training_epochs 100 \
                              --learning_rate 0.0002 \
                              --loss 'mse' \
                              --batch_size 256 --dataset '1M' \
                              --experiment_name "GANs_10m_mse" --on_cluster 'True'
