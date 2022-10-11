# embed feature in advance

import sys

sys.path.append("..")
import json
from tqdm import tqdm
from .util import transform_input_data, transform_data
from .parse_args import BATCH_SIZE, load_model


def embed_by_feat(feat, gnn):
    X, mask = transform_data(feat)
    return gnn.get_embed(X, mask)


def embedding_by_text(feature_path, embedding_path):
    gnn = load_model()

    def embed_by_batch(batch, f_embed):
        X, mask = transform_input_data(batch)
        embeddings = gnn.get_embed(X, mask)
        for fid, embedding in zip(fids, embeddings):
            f_embed.write(json.dumps({'fid': fid, 'embedding': embedding.tolist()}) + '\n')

    f_embed = open(embedding_path, 'w')
    with open(feature_path, 'r') as f:
        batch_data = []
        fids = []
        for line in tqdm(f, desc='embedding...'):
            data = json.loads(line.strip())
            batch_data.append(data)
            fids.append(data['fid'])
            if len(batch_data) == BATCH_SIZE:
                embed_by_batch(batch_data, f_embed)
                batch_data = []
                fids = []
        if batch_data:
            embed_by_batch(batch_data, f)
    f_embed.close()
