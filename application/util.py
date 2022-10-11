#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-----------------File Info-----------------------
Name: util.py
Description:
Author: GentleCP
Email: me@gentlecp.com
Create Date: 2022/9/30 
-----------------End-----------------------------
"""
import json
import pickle as pkl
from pathlib import Path
from collections import OrderedDict

import numpy as np


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def transform_input_data(batch_data):
    """
    transform original data into network input data
    :return:
    """
    max_node_size = 0
    for data in batch_data:
        max_node_size = max(data['n_num'], max_node_size)
    feature_dim = len(batch_data[0]['features'][0])
    X = np.zeros((len(batch_data), max_node_size, feature_dim))  # (batch, max_node_size, )
    mask = np.zeros((len(batch_data), max_node_size, max_node_size))

    for i, data in enumerate(batch_data):
        succs = data['succs']
        features = data['features']
        for start_node, end_nodes in enumerate(succs):
            X[i, start_node, :] = np.array(features[start_node])
            for node in end_nodes:
                mask[i, start_node, node] = 1
    return X, mask


def transform_data(data):
    """
    :param data: {'n_num': 10, 'features': [[1, 2, ...]], 'succs': [[], ..]}
    :return:
    """
    feature_dim = len(data['features'][0])
    node_size = data['n_num']
    X = np.zeros((1, node_size, feature_dim))
    mask = np.zeros((1, node_size, node_size))
    for start_node, end_nodes in enumerate(data['succs']):
        X[0, start_node, :] = np.array(data['features'][start_node])
        for node in end_nodes:
            mask[0, start_node, node] = 1
    return X, mask


def write_pickle(content, fname):
    with open(fname, 'wb') as f:
        pkl.dump(content, f)


def read_pickle(fname):
    with open(fname, 'rb') as f:
        return pkl.load(f)
