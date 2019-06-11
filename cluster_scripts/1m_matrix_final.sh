#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=LongJobs
#SBATCH --gres=gpu:1
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-03:59:59

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



python3 mf_spotlight.py  --use_gpu "True" \
                         --embedding_dim 16 --training_epochs 100 \
                         --learning_rate 0.001 --l2_regularizer 1e-5 \
                         --batch_size 256 --dataset '1M' \
                         --k 5 --neg_examples 5 \
                         --experiment_name "25_matrix_model_1M_1e-5" --on_cluster 'True'

python3 mf_spotlight.py  --use_gpu "True" \
                         --embedding_dim 25 --training_epochs 50 \
                         --learning_rate 0.001 --l2_regularizer 1e-5 \
                         --batch_size 256 --dataset '1M' \
                         --k 5 --neg_examples 5 \
                         --experiment_name "25_matrix_model_1M_1e-5" --on_cluster 'True'


python3 mf_spotlight.py  --use_gpu "True" \
                         --embedding_dim 50 --training_epochs 50 \
                         --learning_rate 0.001 --l2_regulaarizer 1e-5  \
                         --batch_size 256 --dataset '1M' \
                         --k 5 --neg_examples 5 \
                         --experiment_name "50_matrix_model_1M_1e-5" --on_cluster 'True'



