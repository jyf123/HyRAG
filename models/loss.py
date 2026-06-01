import numpy as np
import torch

import models.train_func as tf
import utils

from itertools import combinations

import torch

def safe_logdet(matrix, eps=1e-6, verbose=False):
    """
    计算 logdet，自动处理数值不稳定问题:
    1. 检查矩阵是否包含 NaN/Inf
    2. 检查最小特征值，如果<=0，则加 eps*I 保证正定
    3. 使用 slogdet 确保稳定性
    """
    # Step 1: 检查 NaN / Inf
    if torch.isnan(matrix).any() or torch.isinf(matrix).any():
        raise ValueError("Matrix contains NaN or Inf")

    # Step 2: 检查特征值
    eigvals = torch.linalg.eigvalsh(matrix)  # 对称矩阵专用
    min_eig = eigvals.min().item()

    if verbose:
        print(f"[safe_logdet] Eigenvalues range: min={min_eig:.4e}, max={eigvals.max().item():.4e}")

    if min_eig <= 0:
        if verbose:
            print(f"[safe_logdet] Matrix not PD, shifting by {eps}")
        matrix = matrix + eps * torch.eye(matrix.size(0), device=matrix.device)

    # Step 3: 用 slogdet 计算
    sign, logabsdet = torch.linalg.slogdet(matrix)
    if sign <= 0:
        # 即便修正过仍然出问题
        raise ValueError("Matrix is still not positive definite after regularization.")

    return logabsdet

def debug_logdet(matrix, tol=1e-8):
    # 检查是否包含 NaN/Inf
    if torch.isnan(matrix).any() or torch.isinf(matrix).any():
        print("⚠️ Matrix contains NaN or Inf directly.")
        return

    # 计算特征值
    try:
        eigvals = torch.linalg.eigvalsh(matrix)  # 适合对称矩阵
        min_eig, max_eig = eigvals.min().item(), eigvals.max().item()
        print(f"Eigenvalues range: min={min_eig:.4e}, max={max_eig:.4e}")
    except Exception as e:
        print("❌ Eigenvalue computation failed:", e)
        eigvals = None

    # 检查是否有负特征值或接近0
    if eigvals is not None:
        if min_eig <= 0:
            print("⚠️ Matrix is not positive definite (has non-positive eigenvalues).")
        elif min_eig < tol:
            print("⚠️ Matrix is near-singular (smallest eigenvalue close to 0).")
        else:
            print("✅ Matrix looks positive definite.")

    # slogdet 更稳健（避免 logdet NaN）
    sign, logabsdet = torch.linalg.slogdet(matrix)
    print(f"slogdet: sign={sign.item()}, log|det|={logabsdet.item():.4e}")

    if sign <= 0:
        print("⚠️ Determinant is non-positive, logdet will be NaN.")
    elif torch.isinf(logabsdet):
        print("⚠️ log|det| overflow (too large magnitude).")
    else:
        print("✅ logdet should be valid.")

    # 实际 logdet 调用
    try:
        logdet = torch.logdet(matrix)
        print(f"logdet = {logdet.item():.4e}")
    except Exception as e:
        print("❌ torch.logdet failed:", e)

class MaximalCodingRateReduction(torch.nn.Module):
    def __init__(self, gam1=1.0, gam2=1.0, eps=0.01):
        super(MaximalCodingRateReduction, self).__init__()
        self.gam1 = gam1
        self.gam2 = gam2
        self.eps = eps

    def compute_discrimn_loss_empirical(self, W):
        """Empirical Discriminative Loss."""
        # print(W.shape)
        p, m = W.shape
        I = torch.eye(p).cuda()
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + self.gam1 * scalar * W.matmul(W.T))
        return logdet / 2.

    def compute_compress_loss_empirical(self, W, Pi):
        """Empirical Compressive Loss."""
        p, m = W.shape
        k, _, _ = Pi.shape
        I = torch.eye(p).cuda()
        compress_loss = 0.
        for j in range(k):
            trPi = torch.trace(Pi[j]) + 1e-8
            scalar = p / (trPi * self.eps)
            # L = torch.linalg.cholesky(Pi[j])

            # print(W.size(),Pi[j].size()
            # temp = scalar * L.T.matmul(W.T).matmul(W).matmul(L)
            # small_matrix = torch.eye(temp.size(-1), device=temp.device) + temp
            log_det = torch.logdet(I + scalar * W.matmul(Pi[j]).matmul(W.T))
            # log_det = safe_logdet(temp, eps=1e-6, verbose=True)
            compress_loss += log_det * trPi / m
            if torch.isnan(log_det):
                print("nan")
                # print(Pi[j])
                # # debug_logdet(temp)
                # print("---------")
                # eigenvalues = torch.linalg.eigvalsh(I + scalar * W.matmul(Pi[j]).matmul(W.T))  # 对对称矩阵更高效，返回实特征值
                # if (eigenvalues <= 0).any():
                #     print("WARNING: Matrix has non-positive eigenvalues!")
                #     num_non_positive = (eigenvalues <= 0).sum().item()
                #     print(f"Number of non-positive eigenvalues: {num_non_positive}")
                #     print(f"Smallest eigenvalue: {eigenvalues.min().item()}")
        return compress_loss / 2.

    def compute_discrimn_loss_theoretical(self, W):
        """Theoretical Discriminative Loss."""
        p, m = W.shape
        I = torch.eye(p).cuda()
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + scalar * W.matmul(W.T))
        return logdet / 2.

    def compute_compress_loss_theoretical(self, W, Pi):
        """Theoretical Compressive Loss."""
        p, m = W.shape
        k, _, _ = Pi.shape
        I = torch.eye(p).cuda()
        compress_loss = 0.
        for j in range(k):
            trPi = torch.trace(Pi[j]) + 1e-8
            scalar = p / (trPi * self.eps)
            log_det = torch.logdet(I + scalar * W.matmul(Pi[j]).matmul(W.T))
            compress_loss += trPi / (2 * m) * log_det
        return compress_loss

    def forward(self, X, Y, num_classes=None):
        if num_classes is None:
            num_classes = Y.max() + 1
        W = X.T
        Pi = tf.label_to_membership(Y.numpy(), num_classes)
        Pi = torch.tensor(Pi, dtype=torch.float16).cuda()

        discrimn_loss_empi = self.compute_discrimn_loss_empirical(W)
        compress_loss_empi = self.compute_compress_loss_empirical(W, Pi)
        # if torch.isnan(compress_loss_empi):
        #     print("nan")
        # if compress_loss_empi==float("nan"):
        #     print("nan")
        discrimn_loss_theo = self.compute_discrimn_loss_theoretical(W)
        compress_loss_theo = self.compute_compress_loss_theoretical(W, Pi)
 
        total_loss_empi = self.gam2 * -discrimn_loss_empi + compress_loss_empi
        return (total_loss_empi,
                [discrimn_loss_empi.item(), compress_loss_empi.item()],
                [discrimn_loss_theo.item(), compress_loss_theo.item()])
