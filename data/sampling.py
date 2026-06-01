import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, to_undirected, remove_isolated_nodes, dropout_adj, remove_self_loops, k_hop_subgraph, to_edge_index, to_dgl
from torch_geometric.utils.num_nodes import maybe_num_nodes
import copy
from torch_sparse import SparseTensor
import dgl
from tqdm import tqdm
from collections import deque
import torch.nn.functional as F
from hyper_codes.hyper_distance import get_instance_topk,get_distances,get_single_distance
from hyper_codes.utils import manifolds

def pyg_random_walk(seeds, graph, length, restart_prob=0.8):
    edge_index = graph.edge_index
    node_num = graph.y.shape[0]
    start_nodes = seeds
    graph_num = start_nodes.shape[0]

    value = torch.arange(edge_index.size(1))

    if type(edge_index) == SparseTensor:
        adj_t = edge_index
    else:
        adj_t = SparseTensor(row=edge_index[0], col=edge_index[1],
                                    value=value,
                                    sparse_sizes=(node_num, node_num)).t()
        
    current_nodes = start_nodes.clone()

    history = start_nodes.clone().unsqueeze(0)
    signs = torch.ones(graph_num, dtype=torch.bool).unsqueeze(0)
    for i in range(length):
        seed = torch.rand([graph_num])
        nei = adj_t.sample(1, current_nodes).squeeze()
        sign = seed < restart_prob
        nei[sign] = start_nodes[sign]
        history = torch.cat((history, nei.unsqueeze(0)), dim=0)
        signs = torch.cat((signs, sign.unsqueeze(0)), dim=0)
        current_nodes = nei
    history = history.T
    signs = signs.T

    node_list = []
    edge_list = []
    for i in range(graph_num):
        path = history[i]
        sign = signs[i]
        node_idx = path.unique()
        node_list.append(node_idx)

        sources = path[:-1].numpy().tolist()
        targets = path[1:].numpy().tolist()
        sub_edges = torch.IntTensor([sources, targets]).long()
        sub_edges = sub_edges.T[~sign[1:]].T
        # undirectional
        if sub_edges.shape[1] != 0:
            sub_edges = to_undirected(sub_edges)
        edge_list.append(sub_edges)
    return node_list, edge_list


def RWR_sampler(selected_ids, graph, walk_steps=256, restart_ratio=0.5):
    graph  = copy.deepcopy(graph) # modified on the copy
    edge_index = graph.edge_index
    node_num = graph.x.shape[0]
    start_nodes = selected_ids # only sampling selected nodes as subgraphs
    graph_num = start_nodes.shape[0]
    
    value = torch.arange(edge_index.size(1))

    if type(edge_index) == SparseTensor:
        adj_t = edge_index
    else:
        adj_t = SparseTensor(row=edge_index[0], col=edge_index[1],
                                    value=value,
                                    sparse_sizes=(node_num, node_num)).t()
        
    current_nodes = start_nodes.clone()
    history = start_nodes.clone().unsqueeze(0)
    signs = torch.ones(graph_num, dtype=torch.bool).unsqueeze(0)
    for i in range(walk_steps):
        seed = torch.rand([graph_num])
        nei = adj_t.sample(1, current_nodes).squeeze()
        sign = seed < restart_ratio
        nei[sign] = start_nodes[sign]
        history = torch.cat((history, nei.unsqueeze(0)), dim=0)
        signs = torch.cat((signs, sign.unsqueeze(0)), dim=0)
        current_nodes = nei
    history = history.T
    signs = signs.T

    graph_list = []
    for i in range(graph_num):
        path = history[i]
        sign = signs[i]
        node_idx = path.unique()
        # place the targe index in the first place
        target_idx = path[0].item()
        pos = torch.where(node_idx==target_idx)[0].item()
        if pos != 0:
            tmp = node_idx[0].item()
            node_idx[0] = target_idx
            node_idx[pos] = tmp
        sources = path[:-1].numpy().tolist()
        targets = path[1:].numpy().tolist()
        sub_edges = torch.IntTensor([sources, targets]).long()
        sub_edges = sub_edges.T[~sign[1:]].T
        # undirectional
        if sub_edges.shape[1] != 0:
            sub_edges = to_undirected(sub_edges)
        view = adjust_idx(sub_edges, node_idx, graph, path[0].item())
        view['center_idx'] = target_idx
        view['neig_idx'] = node_idx
        # variables with 'index' will be automatically increased in data loader
        # view = Data(edge_index=sub_edges, x=graph.x[node_idx], center_index=target_idx, center_idx=target_idx, neig_idx=node_idx, y=graph.y[target_idx])

        graph_list.append(view)
    return graph_list

def add_remaining_selfloop_for_isolated_nodes(edge_index, num_nodes):
    num_nodes = max(maybe_num_nodes(edge_index), num_nodes)
    # only add self-loop on isolated nodes
    # edge_index, _ = remove_self_loops(edge_index)
    loop_index = torch.arange(0, num_nodes, dtype=torch.long, device=edge_index.device)
    connected_nodes_indices = torch.cat([edge_index[0], edge_index[1]]).unique()
    mask = torch.ones(num_nodes, dtype=torch.bool)
    mask[connected_nodes_indices] = False
    loops_for_isolatd_nodes = loop_index[mask]
    loops_for_isolatd_nodes = loops_for_isolatd_nodes.unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index, loops_for_isolatd_nodes], dim=1)
    return edge_index

    
    
def collect_subgraphs(selected_id, graph, walk_steps=20, restart_ratio=0.5):
    graph  = copy.deepcopy(graph) # modified on the copy
    edge_index = graph.edge_index
    node_num = graph.x.shape[0]
    start_nodes = selected_id # only sampling selected nodes as subgraphs
    graph_num = start_nodes.shape[0]
    
    value = torch.arange(edge_index.size(1))

    if type(edge_index) == SparseTensor:
        adj_t = edge_index
    else:
        adj_t = SparseTensor(row=edge_index[0], col=edge_index[1],
                                    value=value,
                                    sparse_sizes=(node_num, node_num)).t()
    
    current_nodes = start_nodes.clone()
    history = start_nodes.clone().unsqueeze(0)
    signs = torch.ones(graph_num, dtype=torch.bool).unsqueeze(0)
    for i in range(walk_steps):
        seed = torch.rand([graph_num])
        nei = adj_t.sample(1, current_nodes).squeeze()
        sign = seed < restart_ratio
        nei[sign] = start_nodes[sign]
        history = torch.cat((history, nei.unsqueeze(0)), dim=0)
        signs = torch.cat((signs, sign.unsqueeze(0)), dim=0)
        current_nodes = nei
    history = history.T
    signs = signs.T
    
    graph_list = []
    for i in range(graph_num):
        path = history[i]
        sign = signs[i]
        node_idx = path.unique()
        sources = path[:-1].numpy().tolist()
        targets = path[1:].numpy().tolist()
        sub_edges = torch.IntTensor([sources, targets]).long()
        sub_edges = sub_edges.T[~sign[1:]].T
        # undirectional
        if sub_edges.shape[1] != 0:
            sub_edges = to_undirected(sub_edges)
        view = adjust_idx(sub_edges, node_idx, graph, path[0].item())

        graph_list.append(view)
    return graph_list
        
def adjust_idx(edge_index, node_idx, full_g, center_idx):
    '''re-index the nodes and edge index

    In the subgraphs, some nodes are droppped. We need to change the node index in edge_index in order to corresponds 
    nodes' index to edge index
    '''
    # # put center node in the first place
    # pos = torch.where(node_idx==center_idx)[0].item()
    # if pos != 0:
    #     tmp = node_idx[0]
    #     node_idx[0] = center_idx
    #     node_idx[pos] = tmp
    node_idx_map = {j : i for i, j in enumerate(node_idx.numpy().tolist())}
    sources_idx = list(map(node_idx_map.get, edge_index[0].numpy().tolist()))
    target_idx = list(map(node_idx_map.get, edge_index[1].numpy().tolist()))
    edge_index = torch.IntTensor([sources_idx, target_idx]).long()
    x_view = Data(edge_index=edge_index, x=full_g.x[node_idx], y=full_g.y[center_idx], root_n_index=node_idx_map[center_idx])
    return x_view
def get_full_adj(data,target_rels):
    # 为了效率，我们先构建一个全局的邻接字典，包含所有关系
    # 结构: {起点: {关系类型: [终点1, 终点2]}}
    # 这样只需要遍历一次 Tensor，后面查字典就行
    full_adj = {}
    
    # 筛选出所有我们感兴趣的关系（为了减少构建字典的数据量）
    # map relation-name -> id (assume `data_type` is a list mapping id -> name)
    edge_type_unique = set(data.edge_type)
    relation2id = {name: i for i, name in enumerate(edge_type_unique)}
    # keep only known names and convert to id list
    target_rels = [relation2id[r] for r in target_rels if r in relation2id]
    edge_type = torch.tensor([relation2id[r] for r in data.edge_type if r in relation2id])
    rels_tensor = torch.tensor(target_rels, device=data.ori_x.device)
    mask = torch.isin(edge_type, rels_tensor)
    filtered_edge_index = data.edge_index[:, mask]
    indices = torch.where(mask)[0].tolist()  # [0, 2, 4]
    assert len(indices) == filtered_edge_index.size(1)
    filtered_edge_type = [data.edge_type[i] for i in indices]
    # filtered_edge_type = data.edge_type[mask]
    
    src_list = filtered_edge_index[0].tolist()
    dst_list = filtered_edge_index[1].tolist()
    type_list = filtered_edge_type
    
    # --- 步骤 1: 构建分层的邻接字典 ---
    for s, d, t in zip(src_list, dst_list, type_list):
        if s not in full_adj:
            full_adj[s] = {}
        if t not in full_adj[s]:
            full_adj[s][t] = []
        full_adj[s][t].append(d)
    return full_adj
def one_hop_subgraph(node_idx,edge_index,edge_type):
    src_nodes = edge_index[0]
    dst_nodes = edge_index[1]
    # 找到所有直接邻居（无论方向）
    neighbor_mask = (src_nodes == node_idx) | (dst_nodes == node_idx)
    sub_edge_index = edge_index[:, neighbor_mask]

    indices = torch.nonzero(neighbor_mask).squeeze().cpu().tolist()
    if isinstance(indices,list):
        sub_edge_type = [edge_type[i] for i in indices]
    else:
        sub_edge_type = edge_type[indices]
    
    # 提取所有相关节点（去重）
    all_nodes = torch.cat([sub_edge_index[0], sub_edge_index[1]])
    subset = torch.unique(all_nodes)
    return subset,sub_edge_index,sub_edge_type
def select_topk_indices(subset,topk_indice,edge_index):
    new_subset = []
    for i in range(subset.size(0)):
        if subset[i].item() in topk_indice:
            new_subset.append(subset[i].item())
    new_subset = torch.tensor(new_subset)
    keep_mask = torch.isin(edge_index[0], new_subset) & torch.isin(edge_index[1], new_subset)
    sub_edge_index = edge_index[:, keep_mask]
    # edge_indices = torch.nonzero(keep_mask).squeeze()
    # sub_edge_type = [data.edge_type[i] for i in indices.cpu().tolist()]
    return new_subset,sub_edge_index


def mmr_select(query_emb, cand_emb, k, lam=0.7, mode=None):
    # print("gamma",lam)
    # query_emb: [1500] / [1,1500] / [1,1,1500]; cand_emb: [N,1500] / [N,1,1500]
    if query_emb.dim() == 1: query_emb = query_emb.unsqueeze(0).unsqueeze(0)   # [1,1,1500]
    elif query_emb.dim() == 2: query_emb = query_emb.unsqueeze(1)              # [1,1,1500]
    if cand_emb.dim() == 2: cand_emb = cand_emb.unsqueeze(1)                   # [N,1,1500]
    N = cand_emb.size(0); k = min(k, N)
    d_q = get_distances(query_emb, cand_emb).squeeze()             # [N] (smaller=better)
    # d_q = -d_q  # convert to similarity score (larger=better)
    chosen = torch.zeros(N, dtype=torch.bool, device=cand_emb.device)
    min_d = torch.full((N,), float("inf"), device=cand_emb.device)             # min dist to selected set
    selected = torch.empty(k, dtype=torch.long, device=cand_emb.device)
    for t in range(k):
        score = (-lam * d_q + (1 - lam) * min_d) if t > 0 else (-d_q)
        best = torch.argmax(score.masked_fill(chosen, -1e30))
        selected[t] = best; chosen[best] = True
        d_best = get_distances(cand_emb[best].unsqueeze(0), cand_emb, mode=None).squeeze()  # [N]
        # d_best = -d_best  # convert to similarity score (larger=better)
        min_d = torch.minimum(min_d, d_best)
    return selected.tolist()

def mmr_select_cos(query, candidates_emb, k, lam=0.7):
    """
    query: q
    candidates: list of nodes
    sim(a, b): similarity function (larger = more similar)
    """
    selected = []

    # 预先算 query 相似度
    sim_q = F.cosine_similarity(query, candidates_emb, dim=1)
    # print(sim_q.shape)

    candidates = list(range(candidates_emb.size(0)))

    for _ in range(k):
        best, best_score = None, -float("inf")
        for c in candidates:
            if c in selected:
                continue
            if not selected:
                score = sim_q[c]
            else:
                # print(candidates_emb[c].shape)
                # print(candidates_emb[selected[0]].shape)
                redundancy = max(F.cosine_similarity(candidates_emb[c].unsqueeze(0), candidates_emb[s].unsqueeze(0), dim=1) for s in selected)
                score = lam * sim_q[c] - (1 - lam) * redundancy
            if score > best_score:
                best, best_score = c, score

        selected.append(best)

    return selected


def select_by_angle(center_emb, cand_emb,k=5):
    # print("angel_k",k)
    center_emb, bce = center_emb.split([int(2*1500/3), int(1500/3)], dim=2)#[1024,1,1500]->[1024,1,1000],[1024,1,500]
    cand_emb, bca = cand_emb.split([int(2*1500/3), int(1500/3)], dim=2)#[1024,50,1500]->[1024,50,1000],[1024,50,500]
    
    batch_size = center_emb.size()[0]
    dim = int(center_emb.size()[2] / 2)
    center_emb = center_emb.view(batch_size, -1, 2, dim).transpose(2, 3) 
    cand_emb = cand_emb.view(cand_emb.size()[0], -1, 2, dim).transpose(2, 3) 
    manifold = manifolds.PoincareManifold(K=0.1)
    energy = manifold.angle_at_u(center_emb, cand_emb) - manifold.half_aperture(center_emb)
    energy = energy.clamp(min = 0)
    avg_energy = energy.mean(dim=-1).squeeze()  #[1024,50,1]
    # keep only avg_energy < 0
    # mask = avg_energy < 0
    # if mask.any():
    #     avg_energy_zero = avg_energy.clone()
    #     avg_energy[~mask] = float('inf')
    # print(cand_emb.size(),energy.size(),avg_energy.size())
    k = min(k, avg_energy.size(0))
    selected = torch.argsort(avg_energy, dim=0)[:k]  # indices of smallest avg_energy per batch
    
    return selected
def filter_edges_fast_gpu_corrected(edge_index, start_node, end_nodes, max_hops=3):
    """
    修正版：修复了矩阵乘法方向，能够正确进行双向BFS筛选。
    """
    device = edge_index.device
    
    # 1. 动态确定节点数量 (防止 edge_index 索引越界)
    # 如果你知道具体的节点总数，最好手动传入 num_nodes，否则这里自动推断
    num_nodes = int(edge_index.max()) + 1
    num_edges = edge_index.shape[1]
    
    # 2. 构建稀疏邻接矩阵 (Adjacency Matrix) A
    # A[u, v] = 1 表示 u -> v
    # 必须调用 coalesce() 整理索引，否则稀疏计算可能出错
    values = torch.ones(num_edges, device=device, dtype=torch.float32)
    adj_mat = torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes)).coalesce()
    
    # 3. 定义并行 BFS 函数
    def sparse_bfs(start_indices, matrix, k_limit, direction_name="BFS"):
        # 初始化距离向量: 默认为 max_hops + 1 (不可达)
        # 使用 int16 节省显存
        dists = torch.full((num_nodes,), k_limit + 1, device=device, dtype=torch.int16)
        
        # 当前层活跃节点 (Dense Vector, N x 1)
        current_layer = torch.zeros((num_nodes, 1), device=device, dtype=torch.float32)
        
        if isinstance(start_indices, int):
            start_indices = [start_indices]
            
        # 设置起点
        starts = torch.tensor(start_indices, device=device)
        current_layer[starts] = 1.0
        dists[starts] = 0
        
        # print(f"[{direction_name}] Start nodes: {len(starts)}")
        
        for k in range(1, k_limit + 1):
            if current_layer.sum() == 0:
                break
            
            # 矩阵乘法扩散: matrix @ vector
            # 结果 > 0 的位置即为下一跳邻居
            next_layer_val = torch.sparse.mm(matrix, current_layer)
            
            # 生成 Mask
            next_mask = (next_layer_val > 0).squeeze()
            
            # 只保留【未访问】的节点
            unvisited = (dists == (k_limit + 1))
            new_nodes = next_mask & unvisited
            
            # 更新距离
            dists[new_nodes] = k
            
            # 准备下一层 (重置为 1.0)
            current_layer.zero_()
            current_layer[new_nodes] = 1.0
            
            # count = new_nodes.sum().item()
            # print(f"[{direction_name}] Hop {k}: found {count} new nodes")
            
        return dists.long()

    # 4. 执行双向 BFS (关键修正点！)
    
    # 前向 (Start -> Out): 找后代
    # 邻接矩阵 A 的行是源，列是目。要找 u 的出边，需要列向量 x 在 u 处为 1，
    # 我们想要结果 y 在 v 处为 1 (如果 u->v)。
    # 公式应为 y = A.t() @ x
    # 解释: (A.t() @ x)[v] = sum(A.t()[v,k] * x[k]) = sum(A[k,v] * x[k])
    # 当 x[u]=1, 结果为 A[u,v]，即 u->v 的边存在。正确。
    dist_from_start = sparse_bfs(start_node, adj_mat.t(), max_hops, "Forward")
    
    # 后向 (In -> End): 找祖先
    # 我们想要找谁指向 v。即求 u 使得 u->v。
    # 公式应为 y = A @ x
    # 解释: (A @ x)[u] = sum(A[u,k] * x[k])
    # 当 x[v]=1, 结果为 A[u,v]，即 u->v 的边存在。正确。
    dist_to_end = sparse_bfs(end_nodes, adj_mat, max_hops, "Backward")
    
    # 5. 向量化筛选边
    u_vec = edge_index[0]
    v_vec = edge_index[1]
    
    # 路径判定: dist(start->u) + 1 + dist(v->end) <= max_hops
    # 注意处理溢出: 如果 dist 是 max_hops+1 (inf)，相加会很大，mask 自然为 False
    path_len = dist_from_start[u_vec] + 1 + dist_to_end[v_vec]
    
    mask = path_len <= max_hops
    
    filtered_edges = edge_index[:, mask]
    # print(f"筛选前边数: {num_edges}, 筛选后边数: {filtered_edges.shape[1]}")
    
    return filtered_edges
def filter_edges_with_hops(edge_index, start_node, end_nodes, max_hops=3):
    """
    筛选从 start_node 到 end_nodes 的路径，且路径总长度 <= max_hops
    """
    num_nodes = edge_index.max().item() + 1
    device = edge_index.device
    
    # 转为 CPU 列表进行 BFS (Python 的 dict/list 在图遍历逻辑上通常比 Tensor 循环快)
    src_list = edge_index[0].tolist()
    dst_list = edge_index[1].tolist()
    num_edges = edge_index.shape[1]

    # 1. 构建邻接表 (Adjacency List) 和 反向邻接表 (Reverse Adjacency List)
    adj = {}      # u -> [v1, v2...]
    rev_adj = {}  # v -> [u1, u2...]
    
    for i in range(num_edges):
        u, v = src_list[i], dst_list[i]
        if u not in adj: adj[u] = []
        adj[u].append(v)
        if v not in rev_adj: rev_adj[v] = []
        rev_adj[v].append(u)

    # 2. 定义带距离限制的 BFS 函数
    def bfs_distance(roots, graph_dict, limit):
        # 记录节点到根的距离: {node_id: distance}
        distances = {node: 0 for node in roots}
        queue = deque(roots)
        
        while queue:
            u = queue.popleft()
            d = distances[u]
            
            if d >= limit: # 达到深度限制，不再扩散
                continue
            
            if u in graph_dict:
                for v in graph_dict[u]:
                    if v not in distances: # 只记录第一次到达的最短距离
                        distances[v] = d + 1
                        queue.append(v)
        return distances

    # 3. 执行双向 BFS
    # 前向：计算 start_node 到所有点的距离
    dist_from_start = bfs_distance([start_node], adj, max_hops)
    
    # 后向：计算所有点到 end_nodes 的距离 (使用反向图 rev_adj)
    # 注意：end_nodes 即使是 Tensor 也要转 list
    targets = end_nodes if isinstance(end_nodes, list) else end_nodes.tolist()
    dist_to_end = bfs_distance(targets, rev_adj, max_hops)

    # 4. 向量化筛选 (Vectorized Filtering)
    # 我们利用 Tensor 操作来一次性判断 12万条边，而不是用 for 循环
    
    # 创建距离映射张量 (初始化为无穷大)
    # 用 max_hops + 1 代表不可达/超出限制
    inf_val = max_hops + 999
    d_start_tensor = torch.full((num_nodes,), inf_val, device=device, dtype=torch.long)
    d_end_tensor = torch.full((num_nodes,), inf_val, device=device, dtype=torch.long)
    
    # 填充 BFS 计算出的距离
    if not dist_from_start or not dist_to_end:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    # 将 dict 的键值对转为 tensor 索引填充
    nodes_s = torch.tensor(list(dist_from_start.keys()), device=device)
    vals_s = torch.tensor(list(dist_from_start.values()), device=device)
    d_start_tensor[nodes_s] = vals_s
    
    nodes_e = torch.tensor(list(dist_to_end.keys()), device=device)
    vals_e = torch.tensor(list(dist_to_end.values()), device=device)
    d_end_tensor[nodes_e] = vals_e

    # 5. 核心判断公式
    # 边的起点 u 的 d_start + 1 + 边的终点 v 的 d_end <= max_hops
    # edge_index[0] 是所有边的 u，edge_index[1] 是所有边的 v
    
    u_vec = edge_index[0]
    v_vec = edge_index[1]
    
    # 计算每条边如果作为路径一部分，其构成的总路径长度
    total_path_len = d_start_tensor[u_vec] + 1 + d_end_tensor[v_vec]
    
    # 生成掩码
    mask = total_path_len <= max_hops
    
    return edge_index[:, mask]



def get_embedding(idx,sel_subset,data,query_embedding,device,mode='abs'):
    # 从 sel_subset 中去掉中心节点 idx，得到 neighbors，并计算 data.ori_x[idx] + w * data.ori_x[neighbors]
    neighbors = sel_subset[sel_subset != idx]

    if neighbors.numel() == 0:
        # 没有邻居时直接使用中心向量
        # node_weight = torch.zeros(1, device=device)
        # tau = 5
        # w = torch.tensor([1.], device=device)
        z_kb = data.ori_x[idx].to(device).clone()
    else:
        if mode=='el':
            corpus_embeddings = data.ori_x[neighbors].to(device)
            query_embedding = query_embedding.unsqueeze(0)
            query_embeddings_norm = F.normalize(query_embedding, p=2, dim=1)      # (64, 384)
            corpus_embeddings_norm = F.normalize(corpus_embeddings, p=2, dim=1)      # (2000, 384)

            # 计算余弦相似度：(64, 384) @ (384, 2000) -> (64, 2000)
            node_weight = torch.mm(query_embeddings_norm, corpus_embeddings_norm.t()).squeeze()   # .t() 转置 B_norm
            tau = 5
            w = torch.softmax(node_weight / tau, dim=0)
            z_kb = data.ori_x[idx].to(device) + (w.unsqueeze(-1) * data.ori_x[neighbors].to(device)).sum(dim=0)
        else:
            # 只在 neighbors 上计算距离与权重
            node_weight = get_distances(
                query_embedding.unsqueeze(0).unsqueeze(1),
                data.mlp_x[neighbors].unsqueeze(1).to(device)
            ).squeeze()
            tau = 5
            w = torch.softmax(-node_weight / tau, dim=0)
            z_kb = data.ori_x[idx].to(device) + (w.unsqueeze(-1) * data.ori_x[neighbors].to(device)).sum(dim=0)
    return z_kb

def ego_graphs_sampler_el(node_idx, data,hop=2, sparse=False,mode=None,query_embedding=None,topk_indices=None,gamma=0.7,angle_k=10):
    ego_graphs = []
    graph_embs = []
    if query_embedding!=None:
        device = query_embedding.device
    elif topk_indices!=None:
        device = topk_indices.device
    if sparse:
        edge_index, _ = to_edge_index(data.edge_index)
    else:
        edge_index  = data.edge_index
    if mode=='el':
        q_i = 0
        for idx in node_idx.numpy().tolist():
            subset,sub_edge_index,sub_edge_type = one_hop_subgraph(idx,edge_index,data.edge_type) 
            sel_subset = subset
            sel_sub_edge_index = sub_edge_index
           
            z_kb = get_embedding(idx,sel_subset,data,query_embedding[q_i],device,mode='el')
            graph_embs.append(z_kb.unsqueeze(dim=0))
            q_i +=1
    if mode=='abs' or mode==None:
        q_i = 0
        for idx in node_idx.numpy().tolist():
            subset,sub_edge_index,sub_edge_type = one_hop_subgraph(idx,edge_index,data.edge_type)
            if subset.size(0)>10:
                #用余弦相似度选择topk节点
                ori_resouce_embedding = data.ori_x[subset].to(query_embedding.device)
                # print(ori_resouce_embedding.shape)
                selected_cos = mmr_select_cos(query_embedding[q_i], ori_resouce_embedding, k=min(10,subset.size(0)), lam=0.7)
                sel_subset = subset[selected_cos]
                keep_mask = torch.isin(sub_edge_index[0], sel_subset) & torch.isin(sub_edge_index[1], sel_subset)
                sel_sub_edge_index = sub_edge_index[:, keep_mask]
            else:
                sel_subset = subset
                sel_sub_edge_index = sub_edge_index
            z_kb = get_embedding(idx,sel_subset,data,query_embedding[q_i],device,mode='el')
            graph_embs.append(z_kb.unsqueeze(dim=0))
            q_i +=1
    elif mode=='con':
        topk_indices = topk_indices.cpu()
        q_i = 0
        for idx in node_idx.numpy().tolist():
            subset,sub_edge_index,sub_edge_type = one_hop_subgraph(idx,edge_index,data.edge_type)
            sel_subset = subset
            z_kb = get_embedding(idx,sel_subset,data,query_embedding[q_i],device,mode='el')
            graph_embs.append(z_kb.unsqueeze(dim=0))
    result = torch.cat(graph_embs, dim=0)
    return result
def ego_graphs_sampler(node_idx, data,hop=2, sparse=False,mode=None,query_embedding=None,topk_indices=None,gamma=0.7,angle_k=10):
    ego_graphs = []
    graph_embs = []
    if query_embedding!=None:
        device = query_embedding.device
    elif topk_indices!=None:
        device = topk_indices.device
    if sparse:
        edge_index, _ = to_edge_index(data.edge_index)
    else:
        edge_index  = data.edge_index
    if mode=='el':
        q_i = 0
        for idx in node_idx.numpy().tolist():
            subset,sub_edge_index,sub_edge_type = one_hop_subgraph(idx,edge_index,data.edge_type) 
            sel_subset = subset
            sel_sub_edge_index = sub_edge_index
           
            z_kb = get_embedding(idx,sel_subset,data,query_embedding[q_i],device)
            graph_embs.append(z_kb.unsqueeze(dim=0))
            q_i +=1
    if mode=='abs' or mode==None:
        q_i = 0
        for idx in node_idx.numpy().tolist():
            subset,sub_edge_index,sub_edge_type = one_hop_subgraph(idx,edge_index,data.edge_type)
            if subset.size(0)>10:
                #用双曲距离选择topk节点
                resouce_embedding = data.mlp_x[subset].unsqueeze(1).to(device)
                selected = mmr_select(query_embedding[q_i].unsqueeze(0).unsqueeze(1), resouce_embedding, k=min(10,subset.size(0)), lam=gamma)
                sel_subset = subset[selected]
                keep_mask = torch.isin(sub_edge_index[0], sel_subset) & torch.isin(sub_edge_index[1], sel_subset)
                sel_sub_edge_index = sub_edge_index[:, keep_mask]
                #用余弦相似度选择topk节点
                # ori_resouce_embedding = data.ori_x[subset].to(query_embedding.device)
                # # print(ori_resouce_embedding.shape)
                # selected_cos = mmr_select_cos(query_embedding[q_i], ori_resouce_embedding, k=min(10,subset.size(0)), lam=0.7)
                # sel_subset = subset[selected_cos]
                # keep_mask = torch.isin(sub_edge_index[0], sel_subset) & torch.isin(sub_edge_index[1], sel_subset)
                # sel_sub_edge_index = sub_edge_index[:, keep_mask]
            else:
                sel_subset = subset
                sel_sub_edge_index = sub_edge_index
            z_kb = get_embedding(idx,sel_subset,data,query_embedding[q_i],device)
            graph_embs.append(z_kb.unsqueeze(dim=0))
            q_i +=1
    elif mode=='con':
        topk_indices = topk_indices.cpu()
        q_i = 0
        for idx in node_idx.numpy().tolist():
            subset,sub_edge_index,sub_edge_type = one_hop_subgraph(idx,edge_index,data.edge_type)
            if subset.size(0)>10:
                intersection = torch.tensor([x for x in topk_indices[q_i] if x in subset])
                one_hop_subset = torch.unique(intersection)  # 去重
            else:
                one_hop_subset = subset
            keep_mask = torch.isin(sub_edge_index[0], one_hop_subset) & torch.isin(sub_edge_index[1], one_hop_subset)
            sel_sub_edge_index = sub_edge_index[:, keep_mask]
            # print("one_hop_subset:",one_hop_subset.shape)
            #对topk_indices进行筛选
            # print(one_hop_subset.device,topk_indices.device)
            candidate_topk = topk_indices[q_i][~torch.isin(topk_indices[q_i], one_hop_subset)]
            center_emb = data.mlp_x[idx].unsqueeze(0).unsqueeze(1).cuda()
            candid_emb = data.mlp_x[candidate_topk].unsqueeze(1).cuda()
            selected=select_by_angle(center_emb,candid_emb,k=angle_k).cpu()

            # print("selected:",selected.shape)
            # print(one_hop_subset.device, candidate_topk[selected].device)
            sel_subset = torch.cat((one_hop_subset, candidate_topk[selected]), dim=0).to(torch.int64)
            # 构建从中心节点到 candidate_topk[selected] 的边，并加入到全图 edge_index 中（双向）
            candidate_nodes = candidate_topk[selected]
            if candidate_nodes.numel() > 0:
                src = torch.full((candidate_nodes.size(0),), idx, dtype=torch.long, device=data.edge_index.device)
                tgt = candidate_nodes.to(data.edge_index.device)
                extra_edges = torch.stack([src, tgt], dim=0)
                extra_edges = to_undirected(extra_edges)
                # 将额外边拼接到原始 edge_index，使后续的 filter 能识别这些边
                sel_sub_edge_index = torch.cat([sel_sub_edge_index, extra_edges], dim=1)
            # if sel_subset.dtype=="torch.float32":

            #     print(sel_subset.dtype)
    
            # node_weight = get_distances(query_embedding[q_i].unsqueeze(0).unsqueeze(1), data.mlp_x[sel_subset].unsqueeze(1).to(device)).squeeze()
            # tau = 5
            # w = torch.softmax(-node_weight / tau, dim=0)
            # z_kb = (w.unsqueeze(-1) * data.ori_x[sel_subset].to(device)).sum(dim=0)
            z_kb = get_embedding(idx,sel_subset,data,query_embedding[q_i],device)
            graph_embs.append(z_kb.unsqueeze(dim=0))
            # sub_x = data.ori_x[sel_subset]
            # # raw_texts = [data.raw_texts[i.item()] for i in sel_subset]
            # g = Data(x=sub_x, edge_index=sel_sub_edge_index,node_weight=node_weight)
            # ego_graphs.append(g)
            # q_i +=1
    result = torch.cat(graph_embs, dim=0)
    return result

def find_homogeneous_edges_dict(data, start_node, target_rels,full_adj,subset,topk_indice):
    """
    分别查找从 start_node 开始，沿着 target_rels 中 **单一关系** 延伸的所有边。
    不许在路径中间切换关系类型。
    
    Returns:
        dict: { relation_id: [(u, v), (v, w)...] }
    """
    results = {}
    subset = subset.tolist()
            
    # --- 步骤 2: 对每种目标关系分别进行 BFS ---
    for rel_id in target_rels:
        # 检查起点是否有这种关系，如果没有，直接跳过（这就是你要求的“不算”）
        if start_node not in full_adj or rel_id not in full_adj[start_node]:
            # print(f"起点 {start_node} 没有关系 {rel_id}，跳过。")
            continue
        if full_adj[start_node][rel_id]==[]:
            continue

        one_hop_neighbors = full_adj[start_node][rel_id]
        selected_one_hop_neighbors = []
        for n in one_hop_neighbors:
            if n in subset:
                selected_one_hop_neighbors.append(n)
        tmp_full_adj = full_adj
        tmp_full_adj[start_node][rel_id] = selected_one_hop_neighbors

        if tmp_full_adj[start_node][rel_id]==[]:
            continue
        
        # 开始该关系的独立 BFS
        collected_edges = []
        queue = deque([start_node])
        visited_nodes = {start_node}
        
        while queue:
            curr_node = queue.popleft()
            
            # 这一步很关键：只获取当前 rel_id 下的邻居
            # 如果当前节点没有这种关系的下级，这一支就断了
            if curr_node not in tmp_full_adj or rel_id not in tmp_full_adj[curr_node]:
                continue
            if tmp_full_adj[curr_node][rel_id]==[]:
                continue
            #筛选距离小于threshold的节点
            
            neighbors = tmp_full_adj[curr_node][rel_id]
            # print("before:",neighbors)
            seleced_neighbors = []
            for n in neighbors:
                # print(distance[i].item())
                if n in topk_indice:
                    seleced_neighbors.append(n)
            # print("after:",seleced_neighbors)

            
            for next_node in seleced_neighbors:
                if next_node not in visited_nodes:
                    # 记录边
                    collected_edges.append((curr_node, next_node))
                    # 标记并入队
                    visited_nodes.add(next_node)
                    queue.append(next_node)
        
        # 存入结果字典
        if collected_edges:
            results[rel_id] = collected_edges

    return results
def ego_graphs_sampler_cksg(node_idx, data, hop=2, sparse=False):
    ego_graphs = []
    if sparse:
        edge_index, _ = to_edge_index(data.edge_index)
    else:
        edge_index  = data.edge_index
    for idx in tqdm(node_idx.numpy().tolist()):
        subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph([idx], hop, edge_index, relabel_nodes=False)
        # sub_edge_index = to_undirected(sub_edge_index)
        # print(idx,mapping)
        # assert idx==mapping.item(), f"{idx} vs {mapping.item()}"

        # center_idx = subset[mapping].item() # node idx in the original graph, use idx instead
        # g = Data(x=sub_x, edge_index=sub_edge_index, root_n_index=mapping, y=data.y[idx], original_idx=subset)
        g = Data(edge_index=sub_edge_index, root_n_index=idx) 
         # note: there we use root_n_index to record the index of target node, because `PyG` increments attributes by the number of nodes whenever their attribute names contain the substring :obj:`index`
        ego_graphs.append(g)
    return ego_graphs
# def ego_graphs_sampler(node_idx, data, hop=2, sparse=False):
#     ego_graphs = []
#     if sparse:
#         edge_index, _ = to_edge_index(data.edge_index)
#     else:
#         edge_index  = data.edge_index
#     row, col = edge_index
#     num_nodes = data.x.shape[0]
#     for idx in node_idx.numpy().tolist():
#         subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph([idx], hop, edge_index, relabel_nodes=False)
#         # sub_edge_index = to_undirected(sub_edge_index)
#         pos = torch.where(idx==subset)[0].item()
#         if pos != 0:
#             tmp = subset[0].item()
#             subset[0] = idx
#             subset[pos] = tmp
#         sub_x = data.x[subset]
#         mapping = row.new_full((num_nodes, ), -1)
#         mapping[subset] = torch.arange(subset.size(0), device=row.device)
#         sub_edge_index = mapping[sub_edge_index]

#         # center_idx = subset[mapping].item() # node idx in the original graph, use idx instead
#         g = Data(x=sub_x, edge_index=sub_edge_index, root_n_index=mapping, y=data.y[idx], original_idx=subset) # note: there we use root_n_index to record the index of target node, because `PyG` increments attributes by the number of nodes whenever their attribute names contain the substring :obj:`index`
#         g['center_idx'] = idx
#         g['neig_idx'] = subset
#         ego_graphs.append(g)
#     return ego_graphs