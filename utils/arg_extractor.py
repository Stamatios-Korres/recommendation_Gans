import argparse
import json
import os
import torch
import sys

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

learning_rate = 0.001
l2_regularizer = .0
batch_size = 1024

def get_args():
    """
    Returns a namedtuple with arguments extracted from the command line.
    :return: A namedtuple with arguments
    """
    parser = argparse.ArgumentParser(
        description='Welcome to the MLP course\'s Pytorch training and inference helper script')

    parser.add_argument('--use_gpu', nargs="?", type=str2bool, default=False,
                        help='A flag indicating whether we will use GPU acceleration or not')

    parser.add_argument('--feedback', type=str, default="implicit", help="implicit/explicit")    

    parser.add_argument('--dataset', type=str, default="20M", help="100K/1M/10M/20M")    

    parser.add_argument('--experiment_name', type=str, default="matrix_model", help="Name of resulting experiment")    
    
    parser.add_argument('--embedding_dim', type=int, default=30, help="latents dimensions of matrix factorization models")
    
    parser.add_argument('--training_epochs', type=int, default=100, help="training epochs")

    parser.add_argument('--batch_size', type=int, default=512, help="training epochs")

    parser.add_argument('--learning_rate', type=float, default=0.01, help=" learning rate")

    parser.add_argument('--l2_regularizer', type=float, default=0.0, help=" learning rate")


                
    
    # parser.add_argument('--unfrozen_layers', type=int, default=5, help="number of layers to be trained on transfer learning. HINT: they will freeze 2 times the number of layers")
    
    # parser.add_argument('--adv_train', type=str2bool,default = False, help="specify whether or not to perform adversarial training")
    # parser.add_argument('--adversary', type=str, default="fgsm", help="fgsm/pgd")
    # parser.add_argument('--gamma', type=float, default=0.1, help="optimizer lr gamma")
    # parser.add_argument('--step_size', type=int, default=25, help="optimizer step size to apply gamma")


    args = parser.parse_args()
    return args