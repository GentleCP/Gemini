# embed feature in advance

import tensorflow as tf
import sys

from application.util import transform_input_data

sys.path.append("..")
from utils import *
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import pickle as pkl
from extract_gemini_feat import load_paths
from collections import defaultdict
from util import write_json


def write_pickle(content, fname):
    with open(fname, 'wb') as f:
        pkl.dump(content, f)


def read_pickle(fname):
    with open(fname, 'rb') as f:
        return pkl.load(f)


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
parser.add_argument('--batch_size', type=int, default=5,
                    help='batch size')
parser.add_argument('--load_path', type=str,
                    default='../saved_model/graphnn-model-DFcon_best',
                    help='path for model loading, "#LATEST#" for the latest checkpoint')
parser.add_argument('--log_path', type=str, default=None,
                    help='path for training log')


def embedding_by_bin(gnn, feature_path, embedding_path):
    # print('[embedding] from {} to {}'.format(feature_path, embedding_path))
    # if embedding_path.exists():
    #     return {
    #         'errcode': 0,
    #         'errmsg': 'embedding exist',
    #         'feat_path': str(feature_path),
    #     }
    if not feat_path.exists():
        return {
            'errcode': 400, 'errmsg': 'feat_path not exist', 'feat_path': str(feature_path)
        }

    with open(feature_path, 'r') as f:
        batch_data = []
        func_names = []
        for line in f:
            try:
                gemini_features = json.loads(line.strip())
            except json.decoder.JSONDecodeError:
                continue
            batch_data.append(gemini_features)
            func_names.append(gemini_features['func_name'])


        if not batch_data:
            return {
                'errcode': 300,
                'errmsg': 'feature is empty'
            }
        X, mask = transform_input_data(batch_data)
        batch_embedding = gnn.get_embed(X, mask)
        func_name2embedding = {func_name: emb for func_name, emb in zip(func_names, batch_embedding)}
        write_pickle(func_name2embedding, embedding_path)
    return {
        'errcode': 0,
        'feat_path': str(feature_path),
    }


if __name__ == '__main__':
    from parse_args import load_model

    gnn = load_model()

    base_path = Path('/home/cp/dataset/buildroot-elf-5arch')

    bar = tqdm(load_paths())
    embedding_results = defaultdict(list)
    for path in bar:
        bin_path = base_path.joinpath(path)
        feat_path = bin_path.parent.joinpath(f"{bin_path.name}_Gemini_features.json")
        embed_path = bin_path.parent.joinpath(f"{bin_path.name}_Gemini_embeddings.pkl")
        res = embedding_by_bin(gnn, feat_path, embed_path)
        if res['errcode'] == 0:
            embedding_results['success'].append(res)
        else:
            embedding_results['fail'].append(res)
        bar.set_description(f'success: {len(embedding_results["success"])}, fail:{len(embedding_results["fail"])}')
    write_json(embedding_results, 'embedding_results.json')
