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
                              --training_epochs 5 \
                              --learning_rate 0.002 \
                              --k 3 --slate_size 3\
                              --batch_size 3 --dataset '20M' \
                              --gan_embedding_dim 10 --gan_hidden_layer 16 \
                              --experiment_name "GANs_20m_exp1" --on_cluster 'True'

python3 slate_generation.py   --use_gpu "True"  \
                              --training_epochs 5 \
                              --learning_rate 0.002 \
                              --k 3 --slate_size 3\
                              --batch_size 3 --dataset '20M' \
                              --gan_embedding_dim 10 --gan_hidden_layer 40 \
                              --experiment_name "GANs_20m_exp2" --on_cluster 'True'

python3 slate_generation.py   --use_gpu "True"  \
                              --training_epochs 5 \
                              --learning_rate 0.001 \
                              --k 3 --slate_size 3\
                              --batch_size 3 --dataset '20M' \
                              --gan_embedding_dim 10 --gan_hidden_layer 80 \
                              --experiment_name "GANs_20m_exp3" --on_cluster 'True'      
                              
python3 slate_generation.py   --use_gpu "True"  \
                              --training_epochs 5 \
                              --learning_rate 0.002 \
                              --k 3 --slate_size 3\
                              --batch_size 3 --dataset '20M' \
                              --gan_embedding_dim 10 --gan_hidden_layer 40 \
                              --experiment_name "GANs_20m_exp4" --on_cluster 'True'     

python3 slate_generation.py   --use_gpu "True"  \
                              --training_epochs 5 \
                              --learning_rate 0.01 \
                              --k 3 --slate_size 3\
                              --batch_size 10 --dataset '20M' \
                              --gan_embedding_dim 10 --gan_hidden_layer 32 \
                              --experiment_name "GANs_20m_exp5" --on_cluster 'True'                                                                         
