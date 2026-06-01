import torch
import numpy as np
from pcst_fast import pcst_fast
from torch_geometric.data.data import Data
import torch_geometric.transforms as T

def retrieval_via_pcst_woedge(cskg, q_emb, topk=10, cost_e=0.5, center_node=None):
    c = 0.01
    x = cskg.x
    compare_x = cskg.compare_x
    edge_index = cskg.edge_index
    root = -1  # unrooted
    num_clusters = 1
    pruning = 'gw'
    verbosity_level = 0

    num_queries = q_emb.size(0)  # 32
    subgraphs = []  # 存储每个查询的子图

    for i in range(num_queries):
        current_q = q_emb[i]  # shape: [1, graph_input_dim]
        if topk > 0:
            # print(current_q.size(),x.size())
            sims = torch.nn.CosineSimilarity(dim=-1)(current_q, compare_x)
        # n_prizes = (n_prizes - n_prizes.min()) / (n_prizes.max() - n_prizes.min() + 1e-8)  # 归一化到 [0, 1]
            topk = min(topk, compare_x.size(0))
            _, topk_n_indices = torch.topk(sims, topk, largest=True)
            
            n_prizes = torch.zeros_like(sims)
            n_prizes[topk_n_indices] = torch.arange(topk, 0, -1).float()

            # # 计算一次完整的相似度以便选出最不相关的节点（避免与已选 topk 重叠）
            # sims = torch.nn.CosineSimilarity(dim=-1)(current_q, x)
            # # 已有 topk 的数量（在上面已做 min）
            # topk_count = min(topk, x.size(0))
            # 最不相关的数量，避免与 topk 重叠
            # bottomk_count = min(topk, x.size(0) - topk)

           
            # _, bottomk_n_indices = torch.topk(sims, topk, largest=False)
            # # 最不相关的第 i 个节点分配 prize = topk - i（i 从 0 开始，最不相关的为 topk）
            # bottom_prizes = torch.zeros_like(sims)
            # bottom_prizes[bottomk_n_indices] = torch.arange(topk, 0, -1, device=sims.device, dtype=n_prizes.dtype)
        else:
            n_prizes = torch.zeros(x.size(0))
    
        edges = []
    
        costs = [cost_e for _ in range(edge_index.size(1))]
        edges = edge_index.T.cpu().numpy()

    
        costs = np.array(costs)
        edges = np.array(edges)
        n_prizes = n_prizes.cpu().numpy()
        vertices, edges = pcst_fast(edges, n_prizes, costs, root, num_clusters, pruning, verbosity_level)
        # bottom_vertices, bottom_edges = pcst_fast(edges, bottom_prizes.cpu().numpy(), costs, root, num_clusters, pruning, verbosity_level)
        # print(vertices, edges)
        
        cos = torch.nn.CosineSimilarity(dim=-1)(current_q, compare_x[vertices])
        max_sim_idx = vertices[np.argmax(cos.cpu().numpy())]

        selected_nodes = vertices
        selected_edges = edges

        new_edge_index = edge_index[:, selected_edges]
        selected_nodes = np.unique(np.concatenate([selected_nodes, new_edge_index[0].cpu().numpy(), new_edge_index[1].cpu().numpy()]))

        mapping = {n: i for i, n in enumerate(selected_nodes.tolist())}

        new_x = x[selected_nodes]
        new_rawtext = [cskg.raw_texts[i] for i in selected_nodes]
        src = [mapping[i] for i in new_edge_index[0].cpu().tolist()]
        dst = [mapping[i] for i in new_edge_index[1].cpu().tolist()]
        new_edge_index = torch.LongTensor([src, dst])
        data = Data(x=new_x, edge_index=new_edge_index, raw_texts=new_rawtext,root_n_index=mapping[max_sim_idx])
        # print(data.x.size(),data.edge_index.size())
        transform = T.AddRandomWalkPE(walk_length=32, attr_name='pe')
        data = transform(data)
        subgraphs.append(data)

    # return data,desc
    return subgraphs

def retrieval_via_pcst_woedge_fft(cskg, q_emb, topk=10, cost_e=0.5, center_node=None):
    c = 0.01
    x = cskg.x
    compare_x = cskg.compare_x
    compare_x = torch.fft.fft(compare_x, dim=1)
    edge_index = cskg.edge_index
    root = -1  # unrooted
    num_clusters = 1
    pruning = 'gw'
    verbosity_level = 0
    
    num_queries = q_emb.size(0)  # 32
    subgraphs = []  # 存储每个查询的子图

    for i in range(num_queries):
        current_q = q_emb[i]  # shape: [1, graph_input_dim]
        if topk > 0:
            # print(current_q.size(),x.size())
            print(current_q.dtype,compare_x.dtype)
            sims = torch.nn.CosineSimilarity(dim=-1)(current_q, compare_x)
        # n_prizes = (n_prizes - n_prizes.min()) / (n_prizes.max() - n_prizes.min() + 1e-8)  # 归一化到 [0, 1]
            topk = min(topk, compare_x.size(0))
            _, topk_n_indices = torch.topk(sims, topk, largest=True)
            
            n_prizes = torch.zeros_like(sims)
            n_prizes[topk_n_indices] = torch.arange(topk, 0, -1).float()

            # # 计算一次完整的相似度以便选出最不相关的节点（避免与已选 topk 重叠）
            # sims = torch.nn.CosineSimilarity(dim=-1)(current_q, x)
            # # 已有 topk 的数量（在上面已做 min）
            # topk_count = min(topk, x.size(0))
            # 最不相关的数量，避免与 topk 重叠
            # bottomk_count = min(topk, x.size(0) - topk)

           
            # _, bottomk_n_indices = torch.topk(sims, topk, largest=False)
            # # 最不相关的第 i 个节点分配 prize = topk - i（i 从 0 开始，最不相关的为 topk）
            # bottom_prizes = torch.zeros_like(sims)
            # bottom_prizes[bottomk_n_indices] = torch.arange(topk, 0, -1, device=sims.device, dtype=n_prizes.dtype)
        else:
            n_prizes = torch.zeros(x.size(0))
    
        edges = []
    
        costs = [cost_e for _ in range(edge_index.size(1))]
        edges = edge_index.T.cpu().numpy()

    
        costs = np.array(costs)
        edges = np.array(edges)
        n_prizes = n_prizes.cpu().numpy()
        vertices, edges = pcst_fast(edges, n_prizes, costs, root, num_clusters, pruning, verbosity_level)
        # bottom_vertices, bottom_edges = pcst_fast(edges, bottom_prizes.cpu().numpy(), costs, root, num_clusters, pruning, verbosity_level)
        # print(vertices, edges)
        
        cos = torch.nn.CosineSimilarity(dim=-1)(current_q, compare_x[vertices])
        max_sim_idx = vertices[np.argmax(cos.cpu().numpy())]

        selected_nodes = vertices
        selected_edges = edges

        new_edge_index = edge_index[:, selected_edges]
        selected_nodes = np.unique(np.concatenate([selected_nodes, new_edge_index[0].cpu().numpy(), new_edge_index[1].cpu().numpy()]))

        mapping = {n: i for i, n in enumerate(selected_nodes.tolist())}

        new_x = x[selected_nodes]
        new_rawtext = [cskg.raw_texts[i] for i in selected_nodes]
        src = [mapping[i] for i in new_edge_index[0].cpu().tolist()]
        dst = [mapping[i] for i in new_edge_index[1].cpu().tolist()]
        new_edge_index = torch.LongTensor([src, dst])
        data = Data(x=new_x, edge_index=new_edge_index, raw_texts=new_rawtext,root_n_index=mapping[max_sim_idx])
        # print(data.x.size(),data.edge_index.size())
        transform = T.AddRandomWalkPE(walk_length=32, attr_name='pe')
        data = transform(data)
        subgraphs.append(data)

    # return data,desc
    return subgraphs

def retrieval_via_pcst_withedge(graph, q_emb, topk=3, topk_e=3, cost_e=0.5, center_node=None):
    c = 0.01
    # if len(textual_nodes) == 0 or len(textual_edges) == 0:
    #     desc = textual_nodes.to_csv(index=False) + '\n' + textual_edges.to_csv(index=False, columns=['src', 'edge_attr', 'dst'])
    #     graph = Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr, num_nodes=graph.num_nodes)
    #     return graph, desc

    root = -1  # unrooted
    num_clusters = 1
    pruning = 'gw'
    verbosity_level = 0

    num_queries = q_emb.size(0)  # 32
    subgraphs = []  # 存储每个查询的子图

    for i in range(num_queries):
        current_q = q_emb[i]
        if topk > 0:
            n_prizes = torch.nn.CosineSimilarity(dim=-1)(current_q, graph.x)
            topk = min(topk, graph.num_nodes)
            _, topk_n_indices = torch.topk(n_prizes, topk, largest=True)

            n_prizes = torch.zeros_like(n_prizes)
            n_prizes[topk_n_indices] = torch.arange(topk, 0, -1).float()
        else:
            n_prizes = torch.zeros(graph.num_nodes)

        if topk_e > 0:
            e_prizes = torch.nn.CosineSimilarity(dim=-1)(current_q, graph.edge_feature)
            topk_e = min(topk_e, e_prizes.unique().size(0))

            topk_e_values, _ = torch.topk(e_prizes.unique(), topk_e, largest=True)
            e_prizes[e_prizes < topk_e_values[-1]] = 0.0
            last_topk_e_value = topk_e
            for k in range(topk_e):
                indices = e_prizes == topk_e_values[k]
                value = min((topk_e-k)/sum(indices), last_topk_e_value)
                e_prizes[indices] = value
                last_topk_e_value = value*(1-c)
            # reduce the cost of the edges such that at least one edge is selected
            cost_e = min(cost_e, e_prizes.max().item()*(1-c/2))
        else:
            e_prizes = torch.zeros(graph.num_edges)

        costs = []
        edges = []
        vritual_n_prizes = []
        virtual_edges = []
        virtual_costs = []
        mapping_n = {}
        mapping_e = {}
        for i, (src, dst) in enumerate(graph.edge_index.T.numpy()):
            prize_e = e_prizes[i]
            if prize_e <= cost_e:
                mapping_e[len(edges)] = i
                edges.append((src, dst))
                costs.append(cost_e - prize_e)
            else:
                virtual_node_id = graph.num_nodes + len(vritual_n_prizes)
                mapping_n[virtual_node_id] = i
                virtual_edges.append((src, virtual_node_id))
                virtual_edges.append((virtual_node_id, dst))
                virtual_costs.append(0)
                virtual_costs.append(0)
                vritual_n_prizes.append(prize_e - cost_e)

        prizes = np.concatenate([n_prizes, np.array(vritual_n_prizes)])
        num_edges = len(edges)
        if len(virtual_costs) > 0:
            costs = np.array(costs+virtual_costs)
            edges = np.array(edges+virtual_edges)

        vertices, edges = pcst_fast(edges, prizes, costs, root, num_clusters, pruning, verbosity_level)

        selected_nodes = vertices[vertices < graph.num_nodes]
        selected_edges = [mapping_e[e] for e in edges if e < num_edges]
        virtual_vertices = vertices[vertices >= graph.num_nodes]
        if len(virtual_vertices) > 0:
            virtual_vertices = vertices[vertices >= graph.num_nodes]
            virtual_edges = [mapping_n[i] for i in virtual_vertices]
            selected_edges = np.array(selected_edges+virtual_edges)

        edge_index = graph.edge_index[:, selected_edges]
        selected_nodes = np.unique(np.concatenate([selected_nodes, edge_index[0].numpy(), edge_index[1].numpy()]))

        # n = textual_nodes.iloc[selected_nodes]
        # e = textual_edges.iloc[selected_edges]
        # desc = n.to_csv(index=False)+'\n'+e.to_csv(index=False, columns=['src', 'edge_attr', 'dst'])

        mapping = {n: i for i, n in enumerate(selected_nodes.tolist())}

        x = graph.x[selected_nodes]
        edge_attr = graph.edge_attr[selected_edges]
        src = [mapping[i] for i in edge_index[0].tolist()]
        dst = [mapping[i] for i in edge_index[1].tolist()]
        edge_index = torch.LongTensor([src, dst])
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(selected_nodes),root_n_index=center_node[i])
        transform = T.AddRandomWalkPE(walk_length=32, attr_name='pe')
        data=transform(data)
        subgraphs.append(data)

    return subgraphs


    c = 0.01
    if len(textual_nodes) == 0 or len(textual_edges) == 0:
        desc = textual_nodes.to_csv(index=False) + '\n' + textual_edges.to_csv(index=False, columns=['src', 'edge_attr', 'dst'])
        graph = Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr, num_nodes=graph.num_nodes)
        return graph, desc

    root = -1  # unrooted
    num_clusters = 1
    pruning = 'gw'
    verbosity_level = 0
    if topk > 0:
        n_prizes = torch.nn.CosineSimilarity(dim=-1)(q_emb, graph.x)
        topk = min(topk, graph.num_nodes)
        _, topk_n_indices = torch.topk(n_prizes, topk, largest=True)

        n_prizes = torch.zeros_like(n_prizes)
        n_prizes[topk_n_indices] = torch.arange(topk, 0, -1).float()
    else:
        n_prizes = torch.zeros(graph.num_nodes)

    if topk_e > 0:
        e_prizes = torch.nn.CosineSimilarity(dim=-1)(q_emb, graph.edge_attr)
        topk_e = min(topk_e, e_prizes.unique().size(0))

        topk_e_values, _ = torch.topk(e_prizes.unique(), topk_e, largest=True)
        e_prizes[e_prizes < topk_e_values[-1]] = 0.0
        last_topk_e_value = topk_e
        for k in range(topk_e):
            indices = e_prizes == topk_e_values[k]
            value = min((topk_e-k)/sum(indices), last_topk_e_value)
            e_prizes[indices] = value
            last_topk_e_value = value*(1-c)
        # reduce the cost of the edges such that at least one edge is selected
        cost_e = min(cost_e, e_prizes.max().item()*(1-c/2))
    else:
        e_prizes = torch.zeros(graph.num_edges)

    costs = []
    edges = []
    vritual_n_prizes = []
    virtual_edges = []
    virtual_costs = []
    mapping_n = {}
    mapping_e = {}
    for i, (src, dst) in enumerate(graph.edge_index.T.numpy()):
        prize_e = e_prizes[i]
        if prize_e <= cost_e:
            mapping_e[len(edges)] = i
            edges.append((src, dst))
            costs.append(cost_e - prize_e)
        else:
            virtual_node_id = graph.num_nodes + len(vritual_n_prizes)
            mapping_n[virtual_node_id] = i
            virtual_edges.append((src, virtual_node_id))
            virtual_edges.append((virtual_node_id, dst))
            virtual_costs.append(0)
            virtual_costs.append(0)
            vritual_n_prizes.append(prize_e - cost_e)

    prizes = np.concatenate([n_prizes, np.array(vritual_n_prizes)])
    num_edges = len(edges)
    if len(virtual_costs) > 0:
        costs = np.array(costs+virtual_costs)
        edges = np.array(edges+virtual_edges)

    vertices, edges = pcst_fast(edges, prizes, costs, root, num_clusters, pruning, verbosity_level)

    selected_nodes = vertices[vertices < graph.num_nodes]
    selected_edges = [mapping_e[e] for e in edges if e < num_edges]
    virtual_vertices = vertices[vertices >= graph.num_nodes]
    if len(virtual_vertices) > 0:
        virtual_vertices = vertices[vertices >= graph.num_nodes]
        virtual_edges = [mapping_n[i] for i in virtual_vertices]
        selected_edges = np.array(selected_edges+virtual_edges)

    edge_index = graph.edge_index[:, selected_edges]
    selected_nodes = np.unique(np.concatenate([selected_nodes, edge_index[0].numpy(), edge_index[1].numpy()]))

    n = textual_nodes.iloc[selected_nodes]
    e = textual_edges.iloc[selected_edges]
    desc = n.to_csv(index=False)+'\n'+e.to_csv(index=False, columns=['src', 'edge_attr', 'dst'])

    mapping = {n: i for i, n in enumerate(selected_nodes.tolist())}

    x = graph.x[selected_nodes]
    edge_attr = graph.edge_attr[selected_edges]
    src = [mapping[i] for i in edge_index[0].tolist()]
    dst = [mapping[i] for i in edge_index[1].tolist()]
    edge_index = torch.LongTensor([src, dst])
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(selected_nodes))

    return data, desc
