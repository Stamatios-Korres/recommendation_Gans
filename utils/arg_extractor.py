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


def get_args():
    """
    Returns a namedtuple with arguments extracted from the command line.
    :return: A namedtuple with arguments
    """
    parser = argparse.ArgumentParser(
        description='Welcome to the MLP course\'s Pytorch training and inference helper script')

    # parser.add_argument('--batch_size', nargs="?", type=int, default=100, help='Batch_size for experiment')
    # parser.add_argument('--continue_from_epoch', nargs="?", type=int, default=-1, help='Batch_size for experiment')
    # parser.add_argument('--dataset_name', type=str, help='Dataset on which the system will train/eval our model')
    # parser.add_argument('--seed', nargs="?", type=int, default=7112018,
    #                     help='Seed to use for random number generator for experiment')
    # parser.add_argument('--trained_on', type=str, default="None", help="A string indicating the methods under which the network sas adversarially trained")
                        
    #parser.add_argument('--image_num_channels', nargs="?", type=int, default=1,
    #                    help='The channel dimensionality of our image-data')
    #parser.add_argument('--image_height', nargs="?", type=int, default=28, help='Height of image data')
    #parser.add_argument('--image_width', nargs="?", type=int, default=28, help='Width of image data')
    #parser.add_argument('--dim_reduction_type', nargs="?", type=str, default='strided_convolution',
    #                    help='One of [strided_convolution, dilated_convolution, max_pooling, avg_pooling]')
    #parser.add_argument('--num_layers', nargs="?", type=int, default=4,
                        # help='Number of convolutional layers in the network (excluding '
                        #      'dimensionality reduction layers)')
    #parser.add_argument('--num_filters', nargs="?", type=int, default=64,
                        # help='Number of convolutional filters per convolutional layer in the network (excluding '
                        #      'dimensionality reduction layers)')
    # parser.add_argument('--model', type=str, help='Network architecture for training')
    # parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    # parser.add_argument('--num_epochs', nargs="?", type=int, default=100, help='The experiment\'s epoch budget')
    # parser.add_argument('--experiment_name', nargs="?", type=str, default="exp_1",
    #                     help='Experiment name - to be used for building the experiment folder')
    parser.add_argument('--use_gpu', nargs="?", type=str2bool, default=False,
                        help='A flag indicating whether we will use GPU acceleration or not')
    # parser.add_argument('--gpu_id', type=str, default="None", help="A string indicating the gpu to use")
    # parser.add_argument('--weight_decay_coefficient', nargs="?", type=float, default=1e-05,
    #                     help='Weight decay to use for Adam')
    # parser.add_argument('--filepath_to_arguments_json_file', nargs="?", type=str, default=None,
    #                      help='')
    # parser.add_argument('--source_net', type=str, default="pretrained", help="pretrained/cifar10/cifa100")
    # parser.add_argument('--feature_extraction', type=str2bool, default=True, help="Feature extraction or finetuning")


    
    # parser.add_argument('--unfrozen_layers', type=int, default=5, help="number of layers to be trained on transfer learning. HINT: they will freeze 2 times the number of layers")
    
    # parser.add_argument('--adv_train', type=str2bool,default = False, help="specify whether or not to perform adversarial training")
    # parser.add_argument('--adversary', type=str, default="fgsm", help="fgsm/pgd")
    # parser.add_argument('--gamma', type=float, default=0.1, help="optimizer lr gamma")
    # parser.add_argument('--step_size', type=int, default=25, help="optimizer step size to apply gamma")


    args = parser.parse_args()
    return args