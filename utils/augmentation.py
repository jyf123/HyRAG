import torch
from torch_geometric.utils import dropout_edge
import  copy


def adversarial_aug_train(model_graph, model_text, node_attack, perturb_shapes, step_size, m):
    model_graph.train()
    model_text.train()
    perturbs = []
    for perturb_shape in perturb_shapes:
        perturb = torch.FloatTensor(*perturb_shape).uniform_(-step_size, step_size)  #对节点特征进行扰动，perturb是从(-step_size, step_size)的均匀分布中采样的
        perturb.requires_grad_()
        perturbs.append(perturb)
    
    loss = node_attack(perturbs)
    loss /= m

    for i in range(m-1):#对抗训练
        loss.backward()
        for perturb in perturbs:
            perturb_data = perturb.detach() + step_size * torch.sign(perturb.grad.detach())#希望找到使loss最大化的扰动，因此这里是让pertub_data沿着loss最小化的相反方向（梯度上升）进行更新
            perturb.data = perturb_data.data
            perturb.grad[:] = 0

        loss = node_attack(perturbs)
        loss /=  m

    return loss

def graph_aug(g, f_p, e_p):
    new_g = copy.deepcopy(g)
    drop_mask = torch.empty(
        (g.x.size(1), ),
        dtype=torch.float32,
        device=g.x.device).uniform_(0, 1) < f_p
    
    new_g.x[:, drop_mask] = 0   #对维度进行drop
    e, _ = dropout_edge(new_g.edge_index, p=e_p)
    new_g.edge_index = e
    return new_g