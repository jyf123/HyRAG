import torch
import os.path as osp
import numpy as np


def get_raw_text_wikics(use_text=False, seed=0, with_abs_con=False):
    if with_abs_con:
        path = f"./processed_data/wikics-ac.pt"
    else:
        path = f"./processed_data/wikics.pt"
    print("load data from:",path)
    if osp.exists(path):
        data = torch.load(path, map_location='cpu')
        # data.train_mask = data.train_mask[:,seed]
        # data.val_mask = data.val_mask[:,seed]
        # data.test_mask = data.test_masks[seed]
        data.num_nodes = data.y.shape[0]

        # split data
        node_id = np.arange(data.num_nodes)
        np.random.shuffle(node_id)

        data.train_id = np.sort(node_id[:int(data.num_nodes * 0.6)])
        data.val_id = np.sort(
            node_id[int(data.num_nodes * 0.6):int(data.num_nodes * 0.8)])
        data.test_id = np.sort(node_id[int(data.num_nodes * 0.8):])

        data.train_mask = torch.tensor(
            [x in data.train_id for x in range(data.num_nodes)])
        data.val_mask = torch.tensor(
            [x in data.val_id for x in range(data.num_nodes)])
        data.test_mask = torch.tensor(
            [x in data.test_id for x in range(data.num_nodes)])
        return data, data.raw_texts
    else:
        raise NotImplementedError('No existing wikics dataset!')