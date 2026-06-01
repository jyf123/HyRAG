import torch
import torch.nn.functional as F
import hyper_codes.utils.hyperbolic_utils as hyperbolic

class TopKCollector:
    def __init__(self, k):
        self.k = k
        self.vals = None   # shape [B, k]
        self.idcs = None   # shape [B, k] (global indices in tail)
        self.offset = 0

    def append(self, dist_chunk):
        # normalize dist_chunk to shape [B, C]
        d = dist_chunk

        while d.dim() > 2:
            d = d.squeeze(-1)
        B, C = d.shape
        device = d.device

        chunk_inds = torch.arange(self.offset, self.offset + C, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1)  # [B, C]

        if self.vals is None:
            kk = min(self.k, C)
            vals, inds = torch.topk(d, k=kk, largest=False, dim=1)  # [B, kk], [B, kk]
            global_inds = chunk_inds.gather(1, inds)  # map to global tail indices
            self.vals = vals
            self.idcs = global_inds
        else:
            # combine existing top-k with new chunk values
            combined_vals = torch.cat([self.vals, d], dim=1)        # [B, k + C]
            combined_inds = torch.cat([self.idcs, chunk_inds], dim=1)  # [B, k + C]
            kk = min(self.k, combined_vals.size(1))
            vals, idx = torch.topk(combined_vals, k=kk, largest=False, dim=1)  # [B, kk], [B, kk]
            # gather corresponding global indices
            new_inds = combined_inds.gather(1, idx)
            self.vals = vals
            self.idcs = new_inds

        self.offset += C

    def get_topk(self):
        return self.vals, self.idcs

def get_single_distance(head,tail):
    c = 1
    # print("single:",head.shape)
    head, bh = head.split([int(2*1500/3), int(1500/3)], dim=2)#[1024,1,1500]->[1024,1,1000],[1024,1,500]
    tail, bt = tail.split([int(2*1500/3), int(1500/3)], dim=2)#[1024,50,1500]->[1024,50,1000],[1024,50,500]
    
    batch_size = head.size()[0]
    dim = int(head.size()[2] / 2)
    head = head.view(batch_size, -1, 2, dim).transpose(2, 3) 
    tail = tail.view(tail.size()[0], -1, 2, dim).transpose(2, 3) 
    
    head = hyperbolic.expmap0(head, c)
    tail = hyperbolic.proj(hyperbolic.expmap0(tail, c), c=c)
    
    """
    最简单的分块实现
    """ 
    # 计算距离
    # print(head.shape, tail.shape)
    dist = hyperbolic.sqdist(head, tail, c).squeeze(-1)  # [32, chunk_size, 1, 1]
    # dist_chunk = dist_chunk.view(32, -1)  # [32, chunk_size]

    return dist.mean(dim=-1)

def get_distances(head,tail,mode=None):
    c = 1
    head, bh = head.split([int(2*1500/3), int(1500/3)], dim=2)#[1024,1,1500]->[1024,1,1000],[1024,1,500]
    tail, bt = tail.split([int(2*1500/3), int(1500/3)], dim=2)#[1024,50,1500]->[1024,50,1000],[1024,50,500]
    
    batch_size = head.size()[0]
    dim = int(head.size()[2] / 2)
    head = head.view(batch_size, -1, 2, dim).transpose(2, 3) 
    tail = tail.view(tail.size()[0], -1, 2, dim).transpose(2, 3) 
    
    head = hyperbolic.expmap0(head, c)
    tail = hyperbolic.proj(hyperbolic.expmap0(tail, c), c=c)
    
    #缩放
    if mode == 'abs':
        head_s = hyp_strict_scale(head, s=0.8, c=c)
        head_s = hyperbolic.proj(head_s, c=c)
    elif mode == 'con':
        head_s = hyp_strict_scale(head, s=1.2, c=c)
        head_s = hyperbolic.proj(head_s, c=c)
    else:
        head_s = hyperbolic.proj(head, c=c)
    
    """
    最简单的分块实现
    """
    distance_list = []
    chunk_size = 500  # 调整这个值根据您的GPU内存
    # 将B分块
    for i in range(0, tail.size(0), chunk_size):
        tail_chunk = tail[i:i+chunk_size]  # [chunk_size, 1, 500,2]
        
        # 调整维度
        head_exp = head_s.unsqueeze(1)  # [32, 1, 1, 500,2]
        tail_exp = tail_chunk.unsqueeze(0)  # [1, chunk_size, 1, 500,2]
        
        # 计算距离
        # print(head_exp.shape, tail_exp.shape)
        dist_chunk = hyperbolic.sqdist(head_exp, tail_exp, c).squeeze(-1)  # [32, chunk_size, 1, 1]
        # print("dist_chunk:",dist_chunk.shape)
        # dist_chunk = dist_chunk.view(32, -1)  # [32, chunk_size]
        
        distance_list.append(dist_chunk.mean(dim=-1))
        
        # 清理内存
        del tail_exp, dist_chunk
        torch.cuda.empty_cache()

    # 合并结果
    score = torch.cat(distance_list, dim=1)
    return score


def hyp_strict_scale(x, s: float, c: float):
    """
    Strict hyperbolic scaling in Poincaré ball at origin:
        x' = expmap0( s * logmap0(x, c), c )

    Args:
        x: [..., d] point(s) in Poincaré ball
        s: scaling factor (s<1 shrink, s>1 expand)
        c: positive curvature magnitude (ball radius = 1/sqrt(c))
        proj_fn: optional projection function proj(x, c) to enforce inside-ball constraint

    Returns:
        x_scaled: scaled point(s) in ball
    """
    if s == 1.0:
        return x

    v = hyperbolic.logmap0(x, c)          # map to tangent space at 0 (Euclidean vector)
    v = s * v                  # scale in tangent space (this is the "strict" scaling)
    x2 = hyperbolic.expmap0(v, c)         # map back to ball

    return x2

def radius(tail, eps= 1e-9):
    """
    tail_ball: [N, 500, 2] points in Poincaré ball (c=1 assumed for radius thresholding)
    returns: [N] RMS Euclidean radius across 500 disks
    """
    if tail.dim() == 4 and tail.size(1) == 1:
        tail = tail.squeeze(1)  # [N, 500, 2]
    r = torch.linalg.norm(tail, dim=-1)              # [N, 500]
    r_rms = torch.sqrt(torch.mean(r * r, dim=-1) + eps)   # [N]
    return r_rms

def filter_by_radius(tail_r, r_lo, r_hi):
    """
    tail_rr: [N] precomputed RMS radius of tail nodes
    returns: indices of candidates within [r_lo, r_hi]
    """
    mask = (tail_r >= r_lo) & (tail_r <= r_hi)
    return torch.nonzero(mask, as_tuple=False).squeeze(-1)


def get_instance_topk(head, tail,k=5,mode=None): # a number of $dim$ 2D hyperbolic planes
    c = 1
    head, bh = head.split([int(2*1500/3), int(1500/3)], dim=2)#[1024,1,1500]->[1024,1,1000],[1024,1,500]
    tail, bt = tail.split([int(2*1500/3), int(1500/3)], dim=2)#[1024,50,1500]->[1024,50,1000],[1024,50,500]
    
    batch_size = head.size()[0]
    dim = int(head.size()[2] / 2)
    head = head.view(batch_size, -1, 2, dim).transpose(2, 3) 
    tail = tail.view(tail.size()[0], -1, 2, dim).transpose(2, 3) 
    
    head = hyperbolic.expmap0(head, c)
    tail = hyperbolic.proj(hyperbolic.expmap0(tail, c), c=c)
    head_s = hyperbolic.proj(head, c=c)
    # tail_r = radius(tail)
    # toji = {"0-0.2":0,"0.2-0.4":0,"0.4-0.6":0,"0.6-0.7":0,"0.7-0.9":0,"1":0,"error":0}
    # for n in tail_r:
    #     if n<=0.2:
    #         toji["0-0.2"]+=1
    #     elif n<=0.4:
    #         toji["0.2-0.4"]+=1
    #     elif n<=0.6:
    #         toji["0.4-0.6"]+=1
    #     elif n<=0.7:
    #         toji["0.6-0.7"]+=1
    #     elif n<=0.9:
    #         toji["0.7-0.9"]+=1
    #     elif n<=1:
    #         toji["1"]+=1
    #     else:
    #         toji["error"]+=1
    # print(toji)
    # exit()
    # current_head_norm = torch.norm(head, p=2, dim=-1).cpu()
    # # 2. 设定目标模长 (Target Norm)
    # target_norm = None
    
    #缩放
    # if mode == 'abs':
    #     # 你的 Query 自然模长是 0.82，我们要把它拉回 0.7 (知识库中层分布区)
    #     # head_s = hyp_strict_scale(head, s=0.8, c=c)
    #     # head_s = hyperbolic.proj(head_s, c=c)
    #     tail_idx = filter_by_radius(radius(tail), r_lo=0.0, r_hi=0.8)
    #     tail = tail[tail_idx]
    #     # print(f"Filtered tail size (abs): {tail.size(0)}")
    # elif mode == 'con':
    #     # 你的 Query 自然模长是 0.82，我们要把它推向 0.95 (边缘)
    #     # head_s = hyp_strict_scale(head, s=1.2, c=c)
    #     # head_s = hyperbolic.proj(head_s, c=c)
    #     tail_idx = filter_by_radius(radius(tail), r_lo=0.6, r_hi=1.0)
    #     tail = tail[tail_idx]
    #     print(f"Filtered tail size (con): {tail.size(0)}")
        
    """
    最简单的分块实现
    """
    # A: [32, 1, 1000]
    # B: [200000, 1, 1000]
    
   
    # streaming top-k collector to avoid storing all chunk results in GPU memory
    k_keep = k  # adjust as needed (must match downstream k)
    results = TopKCollector(k_keep)
    chunk_size = 1000  # 调整这个值根据您的GPU内存
    # 将B分块
    for i in range(0, tail.size(0), chunk_size):
        tail_chunk = tail[i:i+chunk_size]  # [chunk_size, 1, 500,2]
        
        # 调整维度
        head_exp = head_s.unsqueeze(1)  # [32, 1, 1, 500,2]
        tail_exp = tail_chunk.unsqueeze(0)  # [1, chunk_size, 1, 500,2]
        
        # 计算距离
        dist_chunk = hyperbolic.sqdist(head_exp, tail_exp, c).squeeze(-1)  # [32, chunk_size, 1, 1]
        
        results.append(dist_chunk.mean(dim=-1))
        
        # 清理内存
        del tail_exp, dist_chunk
        torch.cuda.empty_cache()

  
    topk_values,topk_indices = results.get_topk()
    # if tail_idx is not None:
    #     tail_idx = tail_idx.to(topk_indices.device).long()
    #     topk_indices_global = tail_idx[topk_indices]   # 映射回原始 tail id
    # else:
    topk_indices_global = topk_indices
    distance_threshold = topk_values[:,-1]

    return topk_values,topk_indices_global,distance_threshold

# import torch
# import torch.nn.functional as F

# 假设 TopKCollector 是你自己定义好的类，保持接口不变
# class TopKCollector: ... 

def get_instance_topk_optimized(head, tail, k=5, mode=None):
    """
    Optimized version of get_instance_topk using algebraic decomposition for hyperbolic distance.
    Significantly reduces GPU memory usage and increases speed by avoiding explicit broadcasting.
    """
    c = 1.0
    
    # --- 1. Data Splitting & Preprocessing ---
    # 根据你的逻辑: [2*1500/3] 是前半部分 (1000维)，用来做计算
    split_size = int(2 * 1500 / 3) # 1000
    
    # 切片取前 1000 维
    head = head[..., :split_size] 
    tail = tail[..., :split_size]
    
    batch_size = head.size(0)
    dim = int(head.size(2) / 2) # 500
    
    # Reshape & Transpose: [B, 1, 1000] -> [B, 1, 500, 2]
    # contiguous() is recommended after transpose before view/reshape operations that follow
    head = head.view(batch_size, -1, 2, dim).transpose(2, 3).contiguous() 
    tail = tail.view(tail.size(0), -1, 2, dim).transpose(2, 3).contiguous() 
    
    # Hyperbolic Mapping (Assuming input is in tangent space)
    # Using your hyperbolic library functions
    head = hyperbolic.expmap0(head, c)
    head_s = hyperbolic.proj(head, c=c)
    
    # Tail mapping: usually tail is static, ideally pre-computed outside loop. 
    # But following your structure:
    tail = hyperbolic.expmap0(tail, c)
    tail_s = hyperbolic.proj(tail, c=c) 

    # --- 2. Radius Filtering (Optional / Commented Logic) ---
    # 如果未来需要启用 mode过滤，建议先算 radius，得到 indices 再只对 tail_s 进行切片
    # 这样后续计算量直接减小。
    # indices = torch.arange(tail_s.size(0), device=tail_s.device)
    # if mode == 'abs': ... filter indices ...
    # elif mode == 'con': ... filter indices ...
    # tail_s = tail_s[indices] 

    # --- 3. Optimized Distance Calculation (Algebraic Decomposition) ---
    
    # Pre-compute squared norms: ||x||^2
    # head_s: [B, 1, 500, 2] -> sum last dim -> [B, 1, 500]
    head_sq_norm = (head_s ** 2).sum(dim=-1)
    
    # tail_s: [N, M, 500, 2] -> sum last dim -> [N, M, 500]
    # Assuming M=1 based on your split comment [1024, 1, 1000], or M=50 based on tail comment
    tail_sq_norm = (tail_s ** 2).sum(dim=-1) 

    # Pre-compute Conformal Factors (lambda)
    # lambda = 1 - c * ||x||^2
    head_lambda = 1 - c * head_sq_norm
    tail_lambda = 1 - c * tail_sq_norm
    
    # Collector
    results = TopKCollector(k)
    
    # Chunk size can be much larger now because we don't store (x-y) tensor
    # Try 2000 or even 5000 depending on your GPU (e.g., A100 can handle larger)
    chunk_size = 500
    num_candidates = tail_s.size(0)
    
    # Flatten dimensions for batched matmul: 
    # head [B, 1, 500, 2] -> we treat it as [B, 500, 2] effectively during interaction
    
    for i in range(0, num_candidates, chunk_size):
        # 1. Get Chunk
        # tail_chunk: [Chunk, M, 500, 2]
        tail_chunk = tail_s[i : i + chunk_size]
        curr_tail_sq = tail_sq_norm[i : i + chunk_size] # [Chunk, M, 500]
        curr_tail_lam = tail_lambda[i : i + chunk_size] # [Chunk, M, 500]
        
        # 2. Euclidean Squared Distance ||x-y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
        # Use einsum for efficient inner product <x,y>
        # head: [b, 1, p, d] (b=batch, p=500, d=2)
        # tail: [c, m, p, d] (c=chunk, m=1 or 50)
        # Result dot_prod: [b, c, m, p]
        dot_prod = torch.einsum('bipd, cmpd -> bcmp', head_s, tail_chunk)
        
        # Broadcasting addition
        # [B, 1, 1, 500] + [1, Chunk, M, 500] - 2 * [B, Chunk, M, 500]
        euclid_sq = head_sq_norm.unsqueeze(1).unsqueeze(2) + \
                    curr_tail_sq.unsqueeze(0).unsqueeze(0) - \
                    2 * dot_prod
        
        # Clamp for numerical stability (dist^2 >= 0)
        euclid_sq = euclid_sq.clamp(min=0.0)
        
        # 3. Hyperbolic Distance
        # dist = acosh(1 + 2 * ||x-y||^2 / (lambda_x * lambda_y))
        
        # Denominator: lambda_x * lambda_y
        denom = head_lambda.unsqueeze(1).unsqueeze(2) * curr_tail_lam.unsqueeze(0).unsqueeze(0)
        denom = denom.clamp(min=1e-15)
        
        # Argument for acosh
        arg = 1.0 + 2.0 * c * euclid_sq / denom
        arg = arg.clamp(min=1.0 + 1e-7) # acosh domain is [1, inf)
        
        dist_chunk = torch.acosh(arg)
        
        # 4. Aggregate over the 'dim' (500 planes) dimension
        # dist_chunk: [B, Chunk, M, 500] -> mean -> [B, Chunk, M]
        score = dist_chunk.mean(dim=-1)
        
        # Flatten for collector: [B, Chunk * M]
        # Assuming we just want the top-k nearest instances regardless of M structure
        results.append(score.view(batch_size, -1))
        
        # No explicit del or empty_cache needed! PyTorch allocator handles this efficiently.

    # --- 4. Final Retrieval ---
    topk_values, topk_indices = results.get_topk()
    
    # 如果前面启用了 radius filtering (tail_idx), 这里需要映射回全局索引
    # global_indices = tail_idx[topk_indices] (if filtering enabled)
    # else:
    topk_indices_global = topk_indices
    
    distance_threshold = topk_values[:, -1] # The k-th distance

    return topk_values, topk_indices_global, distance_threshold