import torch
import json
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader,DataListLoader
from data.dataloader import NestedGraphLoader
import torch_geometric.transforms as T
from tqdm import tqdm
from operator import itemgetter
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def parse_source_data(name, data):
    transform = T.AddRandomWalkPE(walk_length=32, attr_name='pe')
    json_data = []

    with open(f'./summary/summary-{name}.json', 'r') as fcc_file: # subgraph-summary pair
        fcc_data = json.load(fcc_file)
        json_data = fcc_data

    collected_graph_data = []
    # collected_text_data = []
    print("process", name)
    for id, jd in enumerate(tqdm(json_data)):
        assert id == jd['id']
        edges = torch.tensor(jd['graph'])
        summary = jd['summary']
        # reindex
        node_idx = torch.unique(edges)
        node_idx_map = {j : i for i, j in enumerate(node_idx.numpy().tolist())}
        sources_idx = list(map(node_idx_map.get, edges[0].numpy().tolist()))
        target_idx = list(map(node_idx_map.get, edges[1].numpy().tolist()))
        edge_index = torch.IntTensor([sources_idx, target_idx]).long()
        graph = Data(edge_index=edge_index, x=data.x[node_idx], y=data.y[jd['id']], root_n_index=node_idx_map[jd['id']], summary=summary)
        graph=transform(graph) # add PE
        collected_graph_data.append(graph)
    return collected_graph_data

def parse_target_data2(name, data, need_rawtext=False,add_noise_rate=0.0,source=''):
    if add_noise_rate==0.0:
        json_path = f"./target_data/{name}.json"
    else:
        json_path = f"./target_data/{name}_noise{add_noise_rate}.json"
    if source!='':
        json_path = f"./target_data/{name}-{source}.json"
    # path = 'target_data/' + name
    print("load subgraph from:",json_path)
    transform = T.AddRandomWalkPE(walk_length=32, attr_name='pe')
    
    json_data = []
    with open(json_path, 'r') as fcc_file:
        fcc_data = json.load(fcc_file)
        json_data = fcc_data

    collected_graph_data = []
    
    for id, jd in enumerate(json_data):
        assert id == jd['id']
        center_node = jd['id']
        edges = torch.tensor(jd['graph'])
        if edges.shape[1] == 0:
            edges = torch.tensor([[id],[id]])
            
        # reindex
        node_idx = torch.unique(edges)
        node_idx_list = node_idx.numpy().tolist()
        node_idx_map = {j : i for i, j in enumerate(node_idx.numpy().tolist())}
        sources_idx = list(map(node_idx_map.get, edges[0].numpy().tolist()))
        target_idx = list(map(node_idx_map.get, edges[1].numpy().tolist()))
        edge_index = torch.IntTensor([sources_idx, target_idx]).long()

        if name=='cskg':
            graph = Data(edge_index=edge_index, x=data.x[node_idx], root_n_index=node_idx_map[center_node],raw_texts = data.raw_texts[center_node],root_n_index_ori=str(center_node))
        else:
            if need_rawtext:
                graph = Data(edge_index=edge_index, x=data.x[node_idx], y=data.y[jd['id']], raw_texts = data.raw_texts[jd['id']], root_n_index=node_idx_map[jd['id']])
            else:
                graph = Data(edge_index=edge_index, x=data.x[node_idx], y=data.y[jd['id']], root_n_index=node_idx_map[jd['id']])
        
        graph=transform(graph) # add PE 为每个节点生成位置编码

        collected_graph_data.append(graph)
    # if flag==1:
    #     print("!!!!!!!!!!!!!!!!!!!!!!!!!!!find 71244")
    # else:
    #     print("!!!!!!!!!!!!!!!!!!!!!!!!!!!no 71244")
        # collected_text_data.append(summary)
    return collected_graph_data

def parse_target_data(name, data, need_rawtext=False,source=''):
    if source!='':
        json_path = f"./target_data/{name}-{source}.json"
    # path = 'target_data/' + name
    json_path = f"./target_data/{name}.json"
    print("load subgraph from:",json_path)
    transform = T.AddRandomWalkPE(walk_length=32, attr_name='pe')
    
    json_data = []
    with open(json_path, 'r') as fcc_file:
        fcc_data = json.load(fcc_file)
        json_data = fcc_data

    collected_graph_data = []
    
    for id, jd in enumerate(json_data):
        assert id == jd['id']
        center_node = jd['id']
        edges = torch.tensor(jd['graph'])
        if edges.shape[1] == 0:
            edges = torch.tensor([[id],[id]])
            
        # reindex
        node_idx = torch.unique(edges)
        node_idx_list = node_idx.numpy().tolist()
        node_idx_map = {j : i for i, j in enumerate(node_idx.numpy().tolist())}
        sources_idx = list(map(node_idx_map.get, edges[0].numpy().tolist()))
        target_idx = list(map(node_idx_map.get, edges[1].numpy().tolist()))
        edge_index = torch.IntTensor([sources_idx, target_idx]).long()

        if name=='cskg':
            graph = Data(edge_index=edge_index, x=data.x[node_idx], root_n_index=node_idx_map[center_node],raw_texts = data.raw_texts[center_node],root_n_index_ori=str(center_node))
        # elif name=='children':
        #     graph = Data(edge_index=edge_index, x=data.x[node_idx], y=data.y[jd['id']], raw_texts = data.raw_texts[jd['id']], root_n_index=node_idx_map[jd['id']])
        else:
            if need_rawtext:
                graph = Data(edge_index=edge_index, x=data.x[node_idx], y=data.y[jd['id']], raw_texts = data.raw_texts[jd['id']], root_n_index=node_idx_map[jd['id']],abs_texts=data.abs_texts[jd['id']],con_texts=data.con_texts[jd['id']])
                # graph = Data(edge_index=edge_index, x=data.x[node_idx], y=data.y[jd['id']], raw_texts = data.raw_texts[jd['id']], root_n_index=node_idx_map[jd['id']])
            else:
                graph = Data(edge_index=edge_index, x=data.x[node_idx], y=data.y[jd['id']], raw_texts = data.raw_texts[jd['id']], root_n_index=node_idx_map[jd['id']],abs_texts=data.abs_texts[jd['id']],con_texts=data.con_texts[jd['id']])
                # graph = Data(edge_index=edge_index, x=data.x[node_idx], y=data.y[jd['id']], root_n_index=node_idx_map[jd['id']])
        
        graph=transform(graph) # add PE 为每个节点生成位置编码

        collected_graph_data.append(graph)
    # if flag==1:
    #     print("!!!!!!!!!!!!!!!!!!!!!!!!!!!find 71244")
    # else:
    #     print("!!!!!!!!!!!!!!!!!!!!!!!!!!!no 71244")
        # collected_text_data.append(summary)
    return collected_graph_data

# def custom_collate(batch):
#     print("Custom collate function is called!")
#     # 1. 使用默认方式合并图数据（处理张量属性，如 x, edge_index）
#     batch = Batch.from_data_list(batch)
    
#     # 2. 手动合并所有子图的 raw_texts 为一个一维列表
#     raw_texts = []
#     for data in batch.to_data_list():  # 遍历批次中的每个子图
#         raw_texts.extend(data.raw_texts)  # 将该子图的节点文本合并到总列表
#     batch.raw_texts = raw_texts  # 覆盖原有的 raw_texts 属性
    
#     return batch


def split_dataloader(data, graphs, batch_size, seed=0, name='cora',use_sg=False):
    train_idx = data.train_mask.nonzero().squeeze()
    val_idx = data.val_mask.nonzero().squeeze()
    test_idx = data.test_mask.nonzero().squeeze()
    train_dataset = [graphs[idx] for idx in train_idx]
    val_dataset = [graphs[idx] for idx in val_idx]
    test_dataset = [graphs[idx] for idx in test_idx]
    test_nums = len(test_dataset)

    if use_sg:
        print(111)
        train_loader = NestedGraphLoader(train_dataset, batch_size=batch_size, shuffle=True) # use DataListLoader for DP rather than DataLoader
        val_loader = NestedGraphLoader(val_dataset, batch_size=batch_size)
        # test_loader = DataListLoader(test_dataset, batch_size=batch_size)
        test_loader = NestedGraphLoader(test_dataset, batch_size=batch_size)
    else:
        train_loader = DataListLoader(train_dataset, batch_size=batch_size, shuffle=True) # use DataListLoader for DP rather than DataLoader
        val_loader = DataListLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        # test_loader = DataListLoader(test_dataset, batch_size=batch_size)


    return train_loader, val_loader, test_loader,test_nums