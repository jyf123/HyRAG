import torch
from torch_geometric import seed_everything
from transformers import AutoTokenizer
import torch.nn.functional as F
from data.load import load_data
from models.graphclip import GraphCLIP
from utils.args import Arguments
from utils.process import parse_target_data, split_dataloader
from data.retrieval import retrieval_via_pcst_woedge
from torch_geometric.data import Batch, Data
import torch_geometric.transforms as T
from hyper_codes.model import KGEModel
# from utils.gen_abs_con import generate_representations,load_model_and_tokenizer
import hyper_codes.utils.hyperbolic_utils as hyperbolic
import os
from data.sampling import ego_graphs_sampler
from data.data_utils.load_cskg import get_cskg
from tqdm import tqdm
from models.lm_modeling import load_model, load_text2embedding
import json
from hyper_codes.hyper_distance import get_instance_topk,get_instance_topk_optimized
import gc
import exp
# from models.Propagation import Propagation
from torch_scatter import scatter_add,scatter_max

eval_template={
    'cora': "this paper has a topic on {c}", 
    'citeseer': "good paper of {c} ", 
    'pubmed': "it belongs to {c} research area",
    'arxiv_2023': "it belongs to {c} research area",
    'wikics': "it belongs to {c} research area",
    'photo':  "this product belongs to {c}",
    'computer':  "is {c} category", 
    'history': "this book belongs to {c}",
    'instagram': "{c}",
    'reddit': "{c}",
    'children': "this book belongs to {c}"
}



def get_center_node(batch_g):
    bs = int(batch_g.num_graphs) if hasattr(batch_g, "num_graphs") else int(batch_g.batch.max().item() + 1)
    ptr = batch_g.ptr if hasattr(batch_g, "ptr") else None
    if ptr is None:
        counts = torch.tensor([(batch_g.batch == i).sum().item() for i in range(bs)], device=device)
        ptr = torch.zeros(bs + 1, dtype=torch.long, device=device)
        ptr[1:] = counts.cumsum(0)
    else:
        ptr = ptr.to(device)
    counts = (ptr[1:] - ptr[:-1]).to(device)

    root_idx = batch_g.root_n_index.clone().to(device)
    if root_idx.max().item() < counts.max().item():
        abs_root = root_idx + ptr[:-1]
    else:
        abs_root = root_idx

    center_feat = batch_g.x[abs_root].to(device)
    return center_feat,abs_root

   

def rag(raw_text_embs, query_embeddings, cskg_data, kge_model, k=3, mode="abs"):
    device = query_embeddings.device
    query_embeddings1 = query_embeddings
    re_hyper = [[] for _ in range(query_embeddings.size(0))]
    re_os = [[] for _ in range(query_embeddings.size(0))]
    # print(query_embeddings.device)
    query_embeddings = kge_model.get_entity_embedding_dir(query_embeddings)
    corpus_embeddings = cskg_data.mlp_x.to(device)
  
    graph_bs = 16
    if mode=="abs":
       
        topk_values, topk_indices, threshold = get_instance_topk(query_embeddings.unsqueeze(1), corpus_embeddings.unsqueeze(1),k=k,mode=mode)
        del corpus_embeddings
        gc.collect()
      
    

        # flatten for retrieval
        flat_indices = topk_indices.reshape(-1)  # [batch_size * k]
        expanded_embeddings = query_embeddings.repeat_interleave(k, dim=0).to(device)

        for j in range(4):
            print("abs:",[cskg_data.raw_texts[topk_indices[j][i].item()] for i in range(k)])
        # retrieve ego graphs for all selected entities (flattened)
        graph_embedding = ego_graphs_sampler(flat_indices.cpu(), cskg_data, hop=1,query_embedding=expanded_embeddings)  #这里graphs的初始向量是用sbert之后的
        batch_size = topk_indices.size(0)
        k_sel = topk_indices.size(1)
        graph_embedding = graph_embedding.view(batch_size, k_sel, -1)
        re_weight = topk_values
    elif mode=="con":
       
        topk_values, topk_indices, threshold = get_instance_topk(query_embeddings.unsqueeze(1), corpus_embeddings.unsqueeze(1),k=2000,mode=mode)
     
        selected_topk_values = topk_values[:,:k]
        selected_topk_indices = topk_indices[:,:k]
        topk_indices = topk_indices.repeat_interleave(k, dim=0).to(device)
        # topk_values = topk_values.repeat_interleave(k, dim=0).to(device)
        del corpus_embeddings
        gc.collect()
        torch.cuda.empty_cache()

        # flatten for retrieval
        flat_indices = selected_topk_indices.reshape(-1)  # [batch_size * k]
        expanded_embeddings = query_embeddings.repeat_interleave(k, dim=0).to(device)
        
        
        for j in range(4):
            print("con:",[cskg_data.raw_texts[selected_topk_indices[j][i].item()] for i in range(k)])
        graph_embedding = ego_graphs_sampler(flat_indices.cpu(), cskg_data, hop=1,mode='con',query_embedding=expanded_embeddings,topk_indices=topk_indices)  #Todo这里graphs的初始向量是用sbert之后的,还是sbert+mlp的
        batch_size = selected_topk_indices.size(0)
        k_sel = selected_topk_indices.size(1)
        graph_embedding = graph_embedding.view(batch_size, k_sel, -1)
        re_weight = selected_topk_values
    return graph_embedding,re_weight
    # return result

def attribute_graph(batch_g,center_feat,graph_embedding,k=3):
    batch_size = batch_g.y.size(0)
    k_sel = k
    transform = T.AddRandomWalkPE(walk_length=32, attr_name='pe')
      
    print(graph_embedding.shape)
    
    
    # center_feat,_ = get_center_node(batch_g)

    d = center_feat.size(1)
    
    assert graph_embedding.size(-1) == d 

    new_graph_list = []

    neighbors_idx = torch.arange(1, k + 1, dtype=torch.long)
    center_idx = torch.zeros(k, dtype=torch.long)
    edge_index_local = torch.stack([
        torch.cat([center_idx, neighbors_idx]),
        torch.cat([neighbors_idx, center_idx])
    ], dim=0).to(device)

    for i in range(batch_size):
        center_node_feat = center_feat[i].unsqueeze(0)
        # center_node_feat = raw_text_embs[i].unsqueeze(0)
        neighbor_feats = graph_embedding[i] 
        
        x_local = torch.cat([center_node_feat, neighbor_feats], dim=0).to(device)
        
        new_graph = Data(x=x_local, edge_index=edge_index_local.clone().to(device),
                            root_n_index=torch.tensor(0, dtype=torch.long))

        new_graph = transform(new_graph)
        new_graph_list.append(new_graph)

    batch_attr_g = Batch.from_data_list(new_graph_list).to(device)
    return batch_attr_g
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


@torch.no_grad()
def test_with_node_text(loader, classes, c_descs, dataset_name,cskg_data=None,hyper_k=3):
    model.eval()
    kge_model.eval()
   
    text_inputs = [eval_template[dataset_name].format(c=c) for c in classes]
    text_inputs = [ti+desc for ti, desc in zip(text_inputs, c_descs)]
    correct = 0
    correct_abs = 0
    correct_con = 0
    correct_rag =0
    jc = 0
    t_model, t_tokenizer, t_device = load_model["sbert"]()
    text2embedding = load_text2embedding["sbert"]
    
    for i, batch in tqdm(enumerate(loader),total=len(loader),desc='Evaling'):
        batch = batch.to(device)
        # print(batch.x.shape)
        re_results = []
        batch_t = tokenizer(text_inputs, truncation=True, padding=True, return_tensors="pt", max_length=512).to(device)
        abs_texts = batch.abs_texts
        con_texts = batch.con_texts
        raw_texts = batch.raw_texts
       
       
        with torch.no_grad():

           
          
            raw_text_embs = text2embedding(t_model, t_tokenizer, t_device, raw_texts).to(device)
            abs_text_embs = text2embedding(t_model, t_tokenizer, t_device, abs_texts).to(device)
            con_text_embs = text2embedding(t_model, t_tokenizer, t_device, con_texts).to(device)
           
            absembedding,abs_value = rag(raw_text_embs,abs_text_embs,cskg_data,kge_model,k=hyper_k,mode='abs')
           
            conembedding,con_value = rag(raw_text_embs,con_text_embs,cskg_data,kge_model,k=hyper_k,mode='con')

            
            #feature-level fusion
            center_feat,center_idx = get_center_node(batch)
            know_embedding = torch.cat([absembedding,conembedding],dim=1)
            know_value = torch.cat([abs_value,con_value],dim=1)
            tau = 5
            w = torch.softmax(-know_value / tau, dim=1)
            print(w.unsqueeze(-1).shape)
            print(know_embedding.shape)
            z_kb = (w.unsqueeze(-1) * know_embedding).sum(dim=1)
            

            print(config.beta)
            fuse_initial = center_feat+config.beta*z_kb
            batch.x[center_idx] = fuse_initial

            abs_graph = attribute_graph(batch,fuse_initial,absembedding,k=hyper_k)
            con_graph = attribute_graph(batch,fuse_initial,conembedding,k=hyper_k)
            
            graph_embs, _ = model.encode_graph(batch)
            abs_graph_embs,_ = model.encode_graph(abs_graph)
            con_graph_embs,_ = model.encode_graph(con_graph)


            text_embs = model.encode_text(batch_t["input_ids"], batch_t['token_type_ids'], batch_t["attention_mask"])  #

            
            text_embs /= text_embs.norm(dim=-1, keepdim=True)
            # embs = (graph_embs+raw_text_embs)/2
            # embs /= embs.norm(dim=-1, keepdim=True)
            # final_similarity = (100 * embs @ text_embs.T).softmax(dim=-1)

            
            # fuse_embs_norm = fuse_embs / fuse_embs.norm(dim=-1, keepdim=True)
            # fuse_logit = (100 * fuse_embs_norm @ text_embs.T)
            # # fuse_entropy = softmax_entropy(fuse_logit)
            # final_similarity = fuse_logit.softmax(dim=-1)

         
            graph_embs_norm = graph_embs / graph_embs.norm(dim=-1, keepdim=True)
            logit = (100 * graph_embs_norm @ text_embs.T)
            entropy = softmax_entropy(logit)
            similarity = logit.softmax(dim=-1)
            
            abs_graph_embs_norm = abs_graph_embs / abs_graph_embs.norm(dim=-1, keepdim=True)
            abs_logit = (100 * abs_graph_embs_norm @ text_embs.T)
            abs_entropy = softmax_entropy(abs_logit)
            abs_similarity = abs_logit.softmax(dim=-1)

            con_graph_embs_norm = con_graph_embs / con_graph_embs.norm(dim=-1, keepdim=True)
            con_logit = (100 * con_graph_embs_norm @ text_embs.T)
            con_entropy = softmax_entropy(con_logit)
            con_similarity = con_logit.softmax(dim=-1)

            group_size = w.size(1) // 2
            abs_con_w = torch.stack([w[:, :group_size].sum(dim=1), w[:, group_size:].sum(dim=1)], dim=1)
            abs_similarity = abs_similarity * abs_con_w[:, 0].unsqueeze(1)
            con_similarity = con_similarity * abs_con_w[:, 1].unsqueeze(1)
            final_similarity = similarity+config.alpha*(abs_similarity+con_similarity)

          
            
            y = batch.y
            
            correct += torch.sum(final_similarity.argmax(dim=1) == y).item()
          
            print(torch.sum(final_similarity.argmax(dim=1) == y).item()/y.shape[0])
           
    print("fuse correct:", correct/len(loader.dataset))
   
    return correct / len(loader.dataset)

if __name__ == "__main__":
    config = Arguments().parse_args()
    print(config)
  
    exp_name = f"{config.target_data}_{config.hyper_k}_alpha{config.alpha}_beta{config.beta}"
    
    log_root = "./eval_logs"
    exp_dir = exp.init_experiment_dir(log_root, exp_name)
    print(f"Experiment started. Results saved to: {exp_dir}")

    logger = exp.redirect_output_to_log(os.path.join(exp_dir, 'output.log'))
    
    exp.save_commit_hash(exp_dir)
    
  
    # current_files_to_backup = [
    #     'eval_withhyper.py', 
    #     'models/graphclip.py', 
    #     'hyper_codes/hyper_distance.py'
    # ]
    exp.backup_files(exp.get_backup_code_path(exp_dir))
    
    exp.save_args(config, exp_dir)
    seed_everything(config.seed) 
    attn_kwargs = {'dropout': 0.0}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained('/data/jyf/code/graphtta/all-MiniLM-L6-v2')
    model = GraphCLIP(384, 1024, 12, attn_kwargs, tokenizer, text_model=config.lm_type,tta=config.tta,step_size=config.step_size)
    pr = torch.load(f"./checkpoints/{config.ckpt}.pt")
    model.load_state_dict(torch.load(f"./checkpoints/{config.ckpt}.pt"), strict=False)
    model.to(device)
    print("model is loaded")

    kge_model = KGEModel(
        model_name='ConE',
        hidden_dim=500,
        gamma=10,
        double_entity_embedding=True,
        double_relation_embedding=False,
        args=config
    ).cuda()
    model_path = os.path.join(config.cskg_pretrained,"checkpoint/ckpt_39999")
    print('Loading checkpoint %s...' % model_path)
    checkpoint = torch.load(model_path)
    # ignore unexpected keys in checkpoint (e.g., relation_mask_embedding, entity_embedding, relation_embedding)
    kge_model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    
    cskg_data = get_cskg(config.cskg_path)
  
    
    target_data = config.target_data.split("+") # testing citeseer dataset, you can add more datasets here
    target_datasets = target_data
    target_classes_list = []
    target_c_desc_list = []
    target_test_loaders = []
    for d in target_data:
        data, text, classes, c_descs = load_data(d, seed=config.seed,with_abs_con=True)
        target_classes_list.append(classes)
        target_c_desc_list.append(c_descs)
        target_graph = parse_target_data(d, data, need_rawtext=True)
        _, _, target_test_loader,_ = split_dataloader(data, target_graph, config.batch_size, seed=config.seed,name=d,use_sg=False)
        
        target_test_loaders.append(target_test_loader)
        print(f"{d} is loaded")
    
    res_str = ""
    all_test_list = []
    res_str2 = ""
    res_str3 = ""
    run_test = []
    for i, classes in enumerate(target_classes_list):
        test_acc2 = test_with_node_text(target_test_loaders[i], classes, target_c_desc_list[i], target_datasets[i], cskg_data=cskg_data,hyper_k=config.hyper_k)
        res_str2 += f" {target_datasets[i]} node text acc: {test_acc2}"
    final_result_str = 'seed:'+str(config.seed) + ', topk: '+str(config.hyper_k)+', alpha:' + str(config.alpha) + ', beta:' + str(config.beta) + ', '+ res_str2 + '\n'
    
    # 写入 result.csv
    with open(exp.get_result_csv_path(exp_dir), 'a') as f:
        f.write(f"Data,seed,topk,alpha,beta,ccuracy\n") # Header
        f.write(f"{config.target_data},{config.seed},{config.hyper_k},{config.alpha},{config.beta},{test_acc2}\n")
    # with open("logs/" + config.target_data + "_output.txt", "a") as f:
    #     f.write()
    print(2, final_result_str)
