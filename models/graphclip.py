from transformers import AutoModel,AutoTokenizer
import numpy as np
import torch
from .gt import GPS
import torch.nn as nn
# import timm
# from timm.models.layers import DropPath
# from timm.models.vision_transformer import Attention, Mlp, Block

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)



text_ids = {
    'tiny': '/data/jyf/code/graphtta/all-MiniLM-L6-v2',
    'sbert':  'sentence-transformers/multi-qa-distilbert-cos-v1', #'sentence-transformers/all-MiniLM-L6-v2', #'sentence-transformers/multi-qa-distilbert-cos-v1',
    'e5': 'intfloat/e5-base-v2',
    'deberta': 'microsoft/deberta-v3-base',
}



class Config:
    def __init__(self):
        self.prefix_projection=True
        self.pre_seq_len=10
        self.num_hidden_layers=6
        self.prefix_hidden_size=384
        self.hidden_size=384

config = Config()


class GraphCLIP(torch.nn.Module):
    def __init__(self, graph_input_dim, graph_hid_dim, graph_num_layer, attn_kwargs, tokenizer, label_dim=7,text_model='tiny',num_heads=8, mlp_ratio=4., prompt = None,tta=False,step_size=None):
        super().__init__()
        self.graph_model = GPS(in_dim=graph_input_dim, channels=graph_hid_dim, out_dim=graph_hid_dim, 
                               pe_dim=8, num_layers=graph_num_layer, attn_type='multihead', attn_kwargs=attn_kwargs)
        print("none")
        self.text_model_type = text_model
        text_id = text_ids[text_model]
        self.tokenizer = tokenizer
        text_model = AutoModel.from_pretrained(text_id)
        self.text_model = text_model
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        if tta:
            if prompt!= None:
                self.prompt_len = len(prompt.split(" "))
                prompt_tokens = tokenizer(prompt, return_tensors="pt")["input_ids"]  # shape: [1, prompt_len]
                prompt_embed = self.text_model.embeddings(prompt_tokens)  # shape: [1, prompt_len, hidden_size]
                self.register_buffer("prompt_init_state", prompt_embed[0, 1 : 1 + self.prompt_len, :].data.clone())
                # self.prompt_init_state = prompt_embed[0, 1 : 1 + self.prompt_len, :].detach().clone()
                # 去掉 batch 维度，并作为 nn.Parameter
                self.le_prompt = torch.nn.Parameter(prompt_embed[0, 1 : 1 + self.prompt_len, :])
            if step_size!= None:
                self.step_size = step_size
                print("step_size:", self.step_size)
                perturb = torch.FloatTensor(graph_input_dim).uniform_(-self.step_size, self.step_size)
                print(perturb.shape)
                self.register_buffer("perturb_init_state", perturb.data.clone())
                self.perturb = torch.nn.Parameter(perturb)
        print("graph_input_dim:", graph_input_dim)
        self.project1 = nn.Linear(graph_input_dim*3, graph_input_dim)
        # self.project2 = nn.Linear(graph_input_dim*2, graph_input_dim)
        # self.project3 = nn.Linear(graph_input_dim*5, graph_input_dim)

    def reset(self): #把可学习的 prompt重置为它们的初始状态
        prompt_emb = self.prompt_init_state
        self.le_prompt.copy_(prompt_emb) # to be optimized
        perturb = self.perturb_init_state
        self.perturb.copy_(perturb) # to be optimized

    def encode_graph(self, batch):
        if batch.edge_weight!=None:
            graph_embs, center_embs = self.graph_model(batch.x, batch.pe, batch.edge_index, batch.batch, batch.root_n_index,batch.edge_weight)
        else:
            # edge_weight = torch.tensor([1]*batch.edge_index.size(1)).to(batch.device)
            graph_embs, center_embs = self.graph_model(batch.x, batch.pe, batch.edge_index, batch.batch, batch.root_n_index)
        return graph_embs, center_embs
    
    # def encode_graph_with_edgeweight(self, batch):
    #     graph_embs, center_embs = self.graph_model.encode_with_edgeweight(batch.x, batch.pe, batch.edge_index, batch.batch, batch.root_n_index,batch.edge_weight)
    #     return graph_embs, center_embs
    
    def encode_whole_graph(self,batch):
        node_embs = self.graph_model.encode_whole_nodes(batch.x,batch.pe,batch.edge_index)
        return node_embs
    
    def get_graph_embedding_withperturb(self,batch):
        perturb = self.perturb.unsqueeze(0).expand(batch.x.size(0), -1)  # shape: [1, graph_input_dim]
        batch.x = batch.x + perturb
        graph_embs = self.graph_model.encode_graphonly(batch.x, batch.pe, batch.edge_index, batch.batch)
        return graph_embs
    
    def get_graph_embedding(self,batch):
        graph_embs = self.graph_model.encode_graphonly(batch.x, batch.pe, batch.edge_index, batch.batch)
        return graph_embs

    def encode_graph_withperturb(self, batch):
        print("use graph perturb")
        perturb = self.perturb.unsqueeze(0).expand(batch.x.size(0), -1)  # shape: [1, graph_input_dim]
        batch.x = batch.x + perturb
        graph_embs, center_embs = self.graph_model(batch.x, batch.pe, batch.edge_index, batch.batch, batch.root_n_index)
        return graph_embs, center_embs

    def encode_text_withleprompt(self, input_ids, token_type_ids, attention_mask):
        print("use le_prompt")
        # 获取嵌入层输出
        input_embeds = self.text_model.embeddings(input_ids)  # shape: [1, seq_len, hidden_size]
        # reset_success = torch.allclose(self.le_prompt,self.prompt_init_state, atol=1e-6)
        # print("✅ Reset 成功" if reset_success else "❌ Reset 失败")
        # 替换embedding（注意不要动 [CLS]）
        le_prompt = self.le_prompt.unsqueeze(0).expand(input_embeds.size(0), -1, -1)  # shape: [1, prompt_len, hidden_size]
        input_embeds[:, 1:self.prompt_len+1, :] = le_prompt
        text_output = self.text_model(inputs_embeds=input_embeds, attention_mask=attention_mask)
        text_embs = mean_pooling(text_output.last_hidden_state, attention_mask)
        return text_embs
    
    def encode_text(self, input_ids, token_type_ids, attention_mask):
        text_output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_embs = mean_pooling(text_output.last_hidden_state, attention_mask)
        return text_embs

    def fuse_embeds(self, graph_embs, low_embs, high_embs):
        x = torch.cat((graph_embs, low_embs, high_embs), dim=1)
        print(x.shape)
        x = self.project1(x)
        return x

    def forward(self, batch_g, batch_t):
        graph_features, c_features = self.encode_graph(batch_g)
        text_features = self.encode_text(**batch_t)

        # normalized features
        graph_features = graph_features / graph_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_graph = logit_scale * graph_features @ text_features.t()
        logits_per_text = logits_per_graph.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_graph, logits_per_text
    
    def freeze_text(self):
        for k, v in self.text_model.named_parameters():
            v.requires_grad = False

    def forward_eval(self, batch_g, batch_t):
        # graph_features, c_features = self.encode_graph(batch_g)
        graph_features, c_features = self.encode_graph_withperturb(batch_g)
        text_features = self.encode_text(batch_t["input_ids"], batch_t['token_type_ids'], batch_t["attention_mask"])
        # normalized features
        # graph_features = graph_features / graph_features.norm(dim=1, keepdim=True)
        # text_features = text_features / text_features.norm(dim=1, keepdim=True)
        # x = self.fuse_graph_text(graph_features, text_features)
        return graph_features, text_features
    
    def fuse_graph_text(self, graph_embs, text_embs):
        # x = torch.cat((graph_embs, text_embs), dim=1)
        # # # x, attn = self.blocks_u(x, ft=False)
        # # # x = self.norm(x)
        # # # x = x.mean(dim=1)
        # x = self.mlp_g(x)
        x = (graph_embs + text_embs)/2

        return x
    
    def loss_cal(self, x, x_aug):
        T = 0.2
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()

        return loss
    
    # def off_diagonal(self,x):
    #     # return a flattened view of the off-diagonal elements of a square matrix
    #     n, m = x.shape
    #     assert n == m
    #     return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    
    # def barlow_twins(self, z1, z2):
    #     # z1 = self.projector(self.backbone(y1))
    #     # z2 = self.projector(self.backbone(y2))

    #     # empirical cross-correlation matrix
    #     c = self.bn(z1).T @ self.bn(z2)

    #     # sum the cross-correlation matrix between all gpus
    #     c.div_(self.args.batch_size)
    #     torch.distributed.all_reduce(c)

    #     on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    #     off_diag = seoff_diagonal(c).pow_(2).sum()
    #     loss = on_diag + self.args.lambd * off_diag
    #     return loss
    
    # used for prompt tuning
    def sup_loss(self, features, labels=None, mask=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    
    