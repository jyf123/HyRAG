from hyper_codes.utils.math_utils import tanh, artanh
import torch
import math

MIN_NORM = 1e-15
eps = {torch.float32: 4e-3, torch.float64: 1e-5}


def proj(x, c):
    norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), MIN_NORM)
    maxnorm = (1 - eps[x.dtype]) / (c ** 0.5)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)

    # norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), MIN_NORM)
    # cond = norm >= 1
    # projected = x / (norm - 1e-5)
    # if (norm == 0).any():
    #     print('encounter zero norm in proj')
    #     import pdb
    #     pdb.set_trace()
    # return torch.where(cond, projected, x)


def proj_p(x, p, c):
    x = x - p
    norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), MIN_NORM)
    cond = norm >= 1
    projected = x / (norm - 1e-5)
    x = p + torch.where(cond, projected, x)
    return x


def proj_tan(u, p, c):
    return u


def proj_tan0(u, c):
    return u


def _lambda_x(x, c):
    x_sqnorm = torch.sum(x.data.pow(2), dim=-1, keepdim=True)
    return 2 / (1. - c * x_sqnorm).clamp_min(MIN_NORM)


def expmap(u, p, c):
    sqrt_c = c ** 0.5
    u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    second_term = (
            tanh(sqrt_c / 2 * _lambda_x(p, c) * u_norm)
            * u
            / (sqrt_c * u_norm)
    )
    gamma_1 = mobius_add(p, second_term, c)
    return gamma_1


def expmap0(u, c):
    sqrt_c = c ** 0.5
    u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), MIN_NORM)
    gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)

    if ((sqrt_c * u_norm)== 0).any():
        print('encounter zero norm in expmap0')
        import pdb
        pdb.set_trace()
    return gamma_1


def logmap(p1, p2, c):  #把点 p2 在双曲流形上的位移表示为位于点 p1 的切空间（Euclidean tangent space）中的向量
    sub = mobius_add(-p1, p2, c)
    sub_norm = sub.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    lam = _lambda_x(p1, c)
    sqrt_c = c ** 0.5
    return 2 / sqrt_c / lam * artanh(sqrt_c * sub_norm) * sub / sub_norm


def logmap0(p, c):
    sqrt_c = c ** 0.5
    p_norm = p.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    z = (sqrt_c * p_norm).clamp(max=1.0 - eps[p.dtype])
    scale = 1. / sqrt_c * artanh(z) / p_norm
    return scale * p


def mobius_add(x, y, c, dim=-1):
    x2 = x.pow(2).sum(dim=dim, keepdim=True)
    y2 = y.pow(2).sum(dim=dim, keepdim=True)
    xy = (x * y).sum(dim=dim, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
    # if (denom == 0).any():
    #     print('encounter zero norm in mb')
    #     import pdb
    #     pdb.set_trace()
    return num / denom.clamp_min(MIN_NORM)


def sqdist(p1, p2, c):
    sqrt_c = c ** 0.5
    m = 1e-4
    dist_c = artanh(
        (sqrt_c * mobius_add(-p1, p2, c, dim=-1).norm(dim=-1, p=2, keepdim=True)).clamp(-1+m, 1-m)
    )

    if (dist_c == float('inf')).any():
        print('encountering inf')
        import pdb
        pdb.set_trace()
    # print((sqrt_c * mobius_add(-p1, p2, c, dim=-1).norm(dim=-1, p=2, keepdim=True)).clamp(-1 + MIN_NORM, 1-MIN_NORM) )

    dist = dist_c * 2 / sqrt_c

    return dist ** 2

# def expmap0(v, c):
#     """ Defining all the parameters in the Euclidean tangent space at the origin """
#     v_norm = v.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
#     prod = c ** 0.5 * v_norm
#     y = math.tanh(prod) * v / prod.clamp_min(MIN_NORM)
#     return y
# 
# 
# def mobius_add(x, y, c, dim=-1):
#     x2 = x.pow(2).sum(dim=dim, keepdim=True)
#     y2 = y.pow(2).sum(dim=dim, keepdim=True)
#     xy = (x * y).sum(dim=dim, keepdim=True)
#     num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
#     denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
#     return num / denom.clamp_min(MIN_NORM)
# 
# 
# def dist(x, y, c):
#     xy = mobius_add(-x, y, c)
#     norm = xy.norm(dim=-1, p=2, keepdim=True)
#     c_sqrt = c ** 0.5
#     return 2 * math.artanh(c_sqrt * norm) / c_sqrt
