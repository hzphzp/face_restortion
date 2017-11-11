"""
Trains a Pixel-CNN++ generative model on CIFAR-10 or Tiny ImageNet data.
Uses multiple GPUs, indicated by the flag --nr-gpu

Example usage:
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_double_cnn.py --nr_gpu 4
"""

import os
import sys
import time
import json
import argparse

import numpy as np
import tensorflow as tf

import pixel_cnn_pp.nn as nn
import pixel_cnn_pp.plotting as plotting
from model_test import model_spec_of_my
import data.cifar10_data as cifar10_data
import data.imagenet_data as imagenet_data

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-i', '--data_dir', type=str,
                    default='~/lab/data/cifar', help='Location for the dataset')
parser.add_argument('-o', '--save_dir', type=str, default='~/lab/save/face_restortion',
                    help='Location for parameter checkpoints and samples')
parser.add_argument('-d', '--data_set', type=str,
                    default='cifar', help='Can be either cifar|imagenet')
parser.add_argument('-t', '--save_interval', type=int, default=20,
                    help='Every how many epochs to write checkpoint/samples?')
parser.add_argument('-r', '--load_params', dest='load_params', action='store_true',
                    help='Restore training from previous model checkpoint?')
# model
parser.add_argument('-q', '--nr_resnet', type=int, default=5,
                    help='Number of residual blocks per stage of the model')
parser.add_argument('-n', '--nr_filters', type=int, default=160,
                    help='Number of filters to use across the model. Higher = larger model.')
parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10,
                    help='Number of logistic components in the mixture. Higher = more flexible model')
parser.add_argument('-z', '--resnet_nonlinearity', type=str, default='concat_elu',
                    help='Which nonlinearity to use in the ResNet layers. One of "concat_elu", "elu", "relu" ')
parser.add_argument('-c', '--class_conditional', dest='class_conditional',
                    action='store_true',default=True,  help='Condition generative model on labels?')
# optimization
parser.add_argument('-l', '--learning_rate', type=float,
                    default=0.001, help='Base learning rate')
parser.add_argument('-e', '--lr_decay', type=float, default=0.999995,
                    help='Learning rate decay, applied every step of the optimization')
parser.add_argument('-b', '--batch_size', type=int, default=12,
                    help='Batch size during training per GPU')
parser.add_argument('-a', '--init_batch_size', type=int, default=100,
                    help='How much data to use for data-dependent initialization.')
parser.add_argument('-p', '--dropout_p', type=float, default=0.5,
                    help='Dropout strength (i.e. 1 - keep_prob). 0 = No dropout, higher = more dropout.')
parser.add_argument('-x', '--max_epochs', type=int,
                    default=5000, help='How many epochs to run in total?')
parser.add_argument('-g', '--nr_gpu', type=int, default=1,
                    help='How many GPUs to distribute the training across?')
# evaluation
parser.add_argument('--polyak_decay', type=float, default=0.9995,
                    help='Exponential decay rate of the sum of previous model iterates during Polyak averaging')
# reproducibility
parser.add_argument('-s', '--seed', type=int, default=1,
                    help='Random seed to use')
args = parser.parse_args()
print('input args:\n', json.dumps(vars(args), indent=4,
                                  separators=(',', ':')))  # pretty print args

print(args.class_conditional)

# -----------------------------------------------------------------------------
# fix random seed for reproducibility
rng = np.random.RandomState(args.seed)
tf.set_random_seed(args.seed)

# initialize data loaders for train/test splits
if args.data_set == 'imagenet' and args.class_conditional:
    raise("We currently don't have labels for the small imagenet data set")
DataLoader = {'cifar': cifar10_data.DataLoader,
              'imagenet': imagenet_data.DataLoader}[args.data_set]
train_data = DataLoader(args.data_dir, 'train', args.batch_size * args.nr_gpu,
                        rng=rng, shuffle=True, return_labels=args.class_conditional)
test_data = DataLoader(args.data_dir, 'test', args.batch_size *
                       args.nr_gpu, shuffle=False, return_labels=args.class_conditional)
obs_shape = train_data.get_observation_size()                                         # e.g. a tuple (32,32,3)
assert len(obs_shape) == 3, 'assumed right now'

# data place holders
x_init = tf.placeholder(tf.float32, shape=(args.init_batch_size,) + obs_shape)          #shape = (100, 32, 32, 3)
xs = [tf.placeholder(tf.float32, shape=(args.batch_size, ) + obs_shape)                #[nr_gpu * (12, 32, 32, 3)]
      for i in range(args.nr_gpu)]

# if the model is class-conditional we'll set up label placeholders +
# one-hot encodings 'h' to condition on
if args.class_conditional:
    num_labels = train_data.get_num_labels()                                #10
    y_init = tf.placeholder(tf.int32, shape=(args.init_batch_size,))      #(100,)
    h_init = tf.one_hot(y_init, num_labels)                             #(100, 10)
    y_sample = np.split(
        np.mod(np.arange(args.batch_size * args.nr_gpu), num_labels), args.nr_gpu)
    h_sample = [tf.one_hot(tf.Variable(
        y_sample[i], trainable=False), num_labels) for i in range(args.nr_gpu)]
    ys = [tf.placeholder(tf.int32, shape=(args.batch_size,))
          for i in range(args.nr_gpu)]                            #(12, )
    hs = [tf.one_hot(ys[i], num_labels) for i in range(args.nr_gpu)]            #(12, 10)

else:
    h_init = None
    h_sample = [None] * args.nr_gpu
    hs = h_sample

# create the model
model_opt = {'nr_resnet': args.nr_resnet, 'nr_filters': args.nr_filters,
             'nr_logistic_mix': args.nr_logistic_mix, 'resnet_nonlinearity': args.resnet_nonlinearity}
model = tf.make_template('model', model_spec_of_my)

# run once for data dependent initialization of parameters
gen_par = model(x_init, h_init, init=True,
                dropout_p=args.dropout_p, **model_opt)

print('done')