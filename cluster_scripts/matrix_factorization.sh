#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=Standard
#SBATCH --gres=gpu:1
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-02:59:59

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

# rsync -ua --progress /home/${STUDENT_ID}/recommendations/datasets/ /disk/scratch/${STUDENT_ID}/data

source /home/${STUDENT_ID}/miniconda3/bin/activate mlp

cd /home/${STUDENT_ID}/recommendations/

python mf_spotlight.py --use_gpu "True" \
                       --embedding_dim 200 --training_epochs 300 \
 		               --learning_rate 0.001 --l2_regularizer 0.0 \
                       --batch_size 512 --dataset '20M' \
                       --experiment_name "matrix_model"

