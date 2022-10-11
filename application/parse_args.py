#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-----------------File Info-----------------------
Name: parse_args.py
Description:
Author: GentleCP
Email: me@gentlecp.com
Create Date: 2022/10/11 
-----------------End-----------------------------
"""
import os
import tensorflow as tf
import argparse
from graphnnSiamese import graphnn

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='0',
                    help='visible gpu device')
parser.add_argument('--fea_dim', type=int, default=7,
                    help='feature dimension')
parser.add_argument('--embed_dim', type=int, default=64,
                    help='embedding dimension')
parser.add_argument('--embed_depth', type=int, default=2,
                    help='embedding network depth')
parser.add_argument('--output_dim', type=int, default=64,
                    help='output layer dimension')
parser.add_argument('--iter_level', type=int, default=5,
                    help='iteration times')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--epoch', type=int, default=100,
                    help='epoch number')
parser.add_argument('--batch_size', type=int, default=10,
                    help='batch size')
parser.add_argument('--load_path', type=str,
                    default='../saved_model/graphnn-model-DFcon_best',
                    help='path for model loading, "#LATEST#" for the latest checkpoint')
parser.add_argument('--log_path', type=str, default=None,
                    help='path for training log')

args = parser.parse_args()
args.dtype = tf.float32
print("=================================")
print(args)
print("=================================")

os.environ["CUDA_VISIBLE_DEVICES"] = args.device

Dtype = args.dtype
NODE_FEATURE_DIM = args.fea_dim
EMBED_DIM = args.embed_dim
EMBED_DEPTH = args.embed_depth
OUTPUT_DIM = args.output_dim
ITERATION_LEVEL = args.iter_level
LEARNING_RATE = args.lr
MAX_EPOCH = args.epoch
BATCH_SIZE = args.batch_size
LOAD_PATH = args.load_path
LOG_PATH = args.log_path

gnn = None


def load_model():
    global gnn
    if gnn is None:
        gnn = graphnn(
            N_x=NODE_FEATURE_DIM,
            Dtype=Dtype,
            N_embed=EMBED_DIM,
            depth_embed=EMBED_DEPTH,
            N_o=OUTPUT_DIM,
            ITER_LEVEL=ITERATION_LEVEL,
            lr=LEARNING_RATE
        )
        gnn.init(LOAD_PATH, LOG_PATH)
    return gnn
