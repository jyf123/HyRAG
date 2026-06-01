import torch
from torch.nn import (
    BatchNorm1d,
    Embedding,
    Linear,
    ModuleList,
    ReLU,
    Sequential,
)
from typing import Any, Dict, Optional
from torch_geometric.nn.attention import PerformerAttention
from torch_geometric.nn import GINEConv, GPSConv, global_add_pool, GCNConv, global_mean_pool, SAGEConv, GATConv, GINConv, SAGPooling,MessagePassing
from torch_geometric.nn import SimpleConv
import gc
from typing import List, Optional, Tuple, Union
from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.typing import Adj, OptPairTensor, Size, SparseTensor, OptTensor
from torch import Tensor

non_MP = SimpleConv(aggr='mean', combine_root='sum')

def check_gpu_memory(top_k=20):
    print(f"\n{'='*20} GPU Memory Snapshot {'='*20}")
    
    # 1. 强制垃圾回收，清理掉那些虽然引用计数为0但还没被释放的显存碎片
    gc.collect()
    torch.cuda.empty_cache()
    
    # 2. 获取当前所有被 Python 追踪的对象
    total_mem = 0
    tensors = []
    
    for obj in gc.get_objects():
        try:
            # 检查对象是否是 Tensor，并且是否在 GPU 上
            if torch.is_tensor(obj) and obj.is_cuda:
                # 记录 Tensor 的信息
                size_mb = obj.element_size() * obj.nelement() / (1024 * 1024)
                tensors.append({
                    'tensor': obj,
                    'shape': tuple(obj.size()),
                    'type': obj.dtype,
                    'size_mb': size_mb,
                    'requires_grad': obj.requires_grad
                })
                total_mem += size_mb
        except Exception as e:
            pass # 有些对象访问可能会报错，跳过
            
    # 3. 按显存占用大小排序 (降序)
    tensors.sort(key=lambda x: x['size_mb'], reverse=True)
    
    print(f"Total GPU Memory Used by Tensors: {total_mem:.2f} MB")
    print(f"Number of Tensors on GPU: {len(tensors)}")
    print(f"\nTop {top_k} Memory Consumers:")
    print(f"{'Size (MB)':<12} | {'Shape':<25} | {'Type':<10} | {'Grad?':<6} | {'Guess'}")
    print("-" * 80)
    
    for t in tensors[:top_k]:
        # 尝试猜测一下这是啥
        guess = "Intermediate/Activation"
        if t['requires_grad']:
            guess = "Parameter/Gradient"
            
        print(f"{t['size_mb']:<12.2f} | {str(t['shape']):<25} | {str(t['type']):<10} | {str(t['requires_grad']):<6} | {guess}")

# from torch_geometric.nn import SAGEConv
# import torch

# class WeightedSAGEConv(SAGEConv):
#      def forward(
#         self,
#         x: Union[Tensor, OptPairTensor],
#         edge_index: Adj,
#         size: Size = None,
    
#     ) -> Tensor:

#         if isinstance(x, Tensor):
#             x = (x, x)

#         if self.project and hasattr(self, 'lin'):
#             x = (self.lin(x[0]).relu(), x[1])

#         # propagate_type: (x: OptPairTensor)
#         out = self.propagate(edge_index, x=x, size=size)
#         out = self.lin_l(out)

#         x_r = x[1]
#         if self.root_weight and x_r is not None:
#             out = out + self.lin_r(x_r)

#         if self.normalize:
#             out = F.normalize(out, p=2., dim=-1)

#         return out
    

class WeightedSAGEConv(MessagePassing):
    r"""
    支持 edge_weight 的 GraphSAGE operator。
    """
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        aggr: Optional[Union[str, List[str], Aggregation]] = "mean",
        normalize: bool = False,
        root_weight: bool = True,
        project: bool = False,
        bias: bool = True,
        **kwargs,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        self.project = project

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        if aggr == 'lstm':
            kwargs.setdefault('aggr_kwargs', {})
            kwargs['aggr_kwargs'].setdefault('in_channels', in_channels[0])
            kwargs['aggr_kwargs'].setdefault('out_channels', in_channels[0])

        super().__init__(aggr, **kwargs)

        if self.project:
            if in_channels[0] <= 0:
                raise ValueError(f"'{self.__class__.__name__}' does not "
                                 f"support lazy initialization with "
                                 f"`project=True`")
            self.lin = Linear(in_channels[0], in_channels[0], bias=True)

        if isinstance(self.aggr_module, MultiAggregation):
            aggr_out_channels = self.aggr_module.get_out_channels(
                in_channels[0])
        else:
            aggr_out_channels = in_channels[0]

        self.lin_l = Linear(aggr_out_channels, out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if self.project:
            self.lin.reset_parameters()
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_weight: OptTensor = None, # <--- 1. 新增 edge_weight 参数
        size: Size = None,
    ) -> Tensor:

        if isinstance(x, Tensor):
            x = (x, x)

        if self.project and hasattr(self, 'lin'):
            x = (self.lin(x[0]).relu(), x[1])

        # propagate_type: (x: OptPairTensor, edge_weight: OptTensor)
        # <--- 2. 将 edge_weight 传给 propagate
        out = self.propagate(edge_index, x=x, size=size, edge_weight=edge_weight)
        
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out = out + self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    # <--- 3. 你的自定义 message 函数
    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        # x_j: [num_edges, channels]
        # edge_weight: [num_edges] or None
        
        if edge_weight is None:
            return x_j
        
        # 将权重维度调整为 [num_edges, 1] 以便广播乘法
        return x_j * edge_weight.view(-1, 1)

    # <--- 4. 重要：删除了 message_and_aggregate
    # 原版 SAGEConv 实现了 message_and_aggregate 来加速计算（绕过 message）。
    # 但那个函数默认不支持 edge_weight。
    # 删除它会强制 PyG 调用上面的 message() 函数，确保权重生效。
    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, aggr={self.aggr})')
    
class GPS(torch.nn.Module):
    def __init__(self, in_dim:int, channels: int, out_dim: int, pe_dim: int, num_layers: int,
                 attn_type: str, attn_kwargs: Dict[str, Any]):
        super().__init__()

        self.node_emb = torch.nn.Linear(in_dim, channels - pe_dim)
        self.pe_lin = Linear(32, pe_dim)
        self.pe_norm = BatchNorm1d(32)

        self.convs = ModuleList()
        for l in range(num_layers):

            conv = GPSConv(channels, WeightedSAGEConv(channels,channels), heads=8,
                           attn_type=attn_type, attn_kwargs=attn_kwargs)
            self.convs.append(conv)

        self.mlp = Sequential(
            Linear(channels*2, 384),
            )
        self.mlp2 = Sequential(
            Linear(channels, 768),
        )
        self.mlp3 = Sequential(
            Linear(channels, 384),
        )
        self.attn_pool = SAGPooling(channels, 0.1)
        self.redraw_projection = RedrawProjection(
            self.convs,
            redraw_interval=1000 if attn_type == 'performer' else None)
        
        self.lora_A_mlp = Linear(channels*2, 16, bias=False)
        self.lora_B_mlp = Linear(16, 384, bias=False)
        self.lora_A_mlp.weight = torch.nn.Parameter(torch.zeros(16,channels*2))
        self.lora_B_mlp.weight = torch.nn.Parameter(torch.zeros(384, 16))
        # self.node_feature_proj = Linear(100, 384)

    def forward(self, x, pe, edge_index, batch, center_idx,edge_weight=None):
        # print(x.shape)
        # print("-------4--------")
        # check_gpu_memory()
        x_pe = self.pe_norm(pe)
        x = torch.cat((self.node_emb(x.squeeze(-1)), self.pe_lin(x_pe)), 1)
        for conv in self.convs:
            if edge_weight!=None:
                x = conv(x, edge_index, batch,edge_weight=edge_weight)
            else:
                x = conv(x, edge_index, batch)
       
        # mean pool
        g_x = global_mean_pool(x, batch)

        c_x = x[center_idx]
        g_x=torch.cat((g_x, c_x), 1) # cat average and center
        
        return self.mlp(g_x), self.mlp2(c_x)
    
    def encode_graphonly(self, x, pe, edge_index, batch):
        x_pe = self.pe_norm(pe)
        x = torch.cat((self.node_emb(x.squeeze(-1)), self.pe_lin(x_pe)), 1)
        for conv in self.convs:
            x = conv(x, edge_index, batch)

        # mean pool
        g_x = global_mean_pool(x, batch)

        return self.mlp3(g_x)
        # return g_x
    
    def encode_whole_nodes(self, x, pe, edge_index):
        # x = self.node_feature_proj(x)
        x_pe = self.pe_norm(pe)
        x = torch.cat((self.node_emb(x.squeeze(-1)), self.pe_lin(x_pe)), 1)
        for conv in self.convs:
            x = conv(x, edge_index)

        # mean pool
        # g_x = global_mean_pool(x)

        return x
    
    # def encode_with_edgeweight(self, x, pe, edge_index, batch, center_idx,edge_weight):
    #     x_pe = self.pe_norm(pe)
    #     x = torch.cat((self.node_emb(x.squeeze(-1)), self.pe_lin(x_pe)), 1)
    #     for conv in self.convs:
    #         x = conv(x, edge_index, batch, edge_attr=edge_weight)

    #     # mean pool
    #     g_x = global_mean_pool(x, batch)

    #     c_x = x[center_idx]
    #     g_x=torch.cat((g_x, c_x), 1) # cat average and center
        
    #     return self.mlp(g_x), self.mlp2(c_x)


class RedrawProjection:
    def __init__(self, model: torch.nn.Module,
                 redraw_interval: Optional[int] = None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            fast_attentions = [
                module for module in self.model.modules()
                if isinstance(module, PerformerAttention)
            ]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        self.num_last_redraw += 1
