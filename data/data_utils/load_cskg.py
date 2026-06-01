import torch
import os.path as osp
import pandas as pd
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, to_undirected
import numpy as np


def get_cskg(path):
    # path = f"./datasets/{path}"
    print("load data from:",path)
    if osp.exists(path):
        data = torch.load(path, map_location='cpu')
        data.num_nodes = data.ori_x.shape[0]
        return data
    else:
        raise NotImplementedError('No existing cskg dataset!')