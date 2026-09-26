import math

import torch
import triton
import triton.language as tl


def _accum_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def _flash_attention_forward_pytorch(Q, K, V, is_causal=False, block_size=16):
    batch_size, seq_q, head_dim = Q.shape
    seq_k = K.shape[1]
    sqrt_d = math.sqrt(head_dim)
    acc_dtype = _accum_dtype(Q.dtype)

    O = torch.empty((batch_size, seq_q, V.shape[-1]), device=Q.device, dtype=Q.dtype)
    L = torch.empty((batch_size, seq_q), device=Q.device, dtype=acc_dtype)

    for q_start in range(0, seq_q, block_size):
        q_end = min(q_start + block_size, seq_q)
        q_block = Q[:, q_start:q_end, :].to(acc_dtype)

        m_i = torch.full(
            (batch_size, q_end - q_start),
            float("-inf"),
            device=Q.device,
            dtype=acc_dtype,
        )
        l_i = torch.zeros((batch_size, q_end - q_start), device=Q.device, dtype=acc_dtype)
        o_i = torch.zeros(
            (batch_size, q_end - q_start, V.shape[-1]),
            device=Q.device,
            dtype=acc_dtype,
        )

        for k_start in range(0, seq_k, block_size):
            if is_causal and k_start >= q_end:
                break

            k_end = min(k_start + block_size, seq_k)
            k_block = K[:, k_start:k_end, :].to(acc_dtype)
            v_block = V[:, k_start:k_end, :].to(acc_dtype)

            scores = torch.matmul(q_block, k_block.transpose(-1, -2)) / sqrt_d

            if is_causal:
                q_idx = torch.arange(q_start, q_end, device=Q.device).unsqueeze(-1)
                k_idx = torch.arange(k_start, k_end, device=Q.device).unsqueeze(0)
                causal_mask = q_idx >= k_idx
                scores = scores.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))

            block_max = scores.max(dim=-1).values
            m_new = torch.maximum(m_i, block_max)

            exp_m = torch.where(
                torch.isfinite(m_i),
                torch.exp(m_i - m_new),
                torch.zeros_like(m_new),
            )
            p = torch.exp(scores - m_new.unsqueeze(-1))
            l_new = l_i * exp_m + p.sum(dim=-1)

            alpha = torch.where(l_new > 0, (l_i * exp_m) / l_new, torch.zeros_like(l_new))
            o_i = o_i * alpha.unsqueeze(-1) + torch.matmul(p, v_block) / l_new.unsqueeze(-1)

            m_i = m_new
            l_i = l_new

        O[:, q_start:q_end, :] = o_i.to(Q.dtype)
        L[:, q_start:q_end] = m_i + torch.log(l_i)

    return O, L, sqrt_d


def attention_backward_impl(Q, K, V, L, O, dO, sqrt_d, is_causal):
    """
    Backward pass for Flash Attention using score recomputation.
    """
    acc_dtype = _accum_dtype(Q.dtype)
    Q_ = Q.to(acc_dtype)
    K_ = K.to(acc_dtype)
    V_ = V.to(acc_dtype)
    O_ = O.to(acc_dtype)
    dO_ = dO.to(acc_dtype)
    L_ = L.to(acc_dtype)

    D = (O_ * dO_).sum(dim=-1, keepdim=True)
    S = torch.matmul(Q_, K_.transpose(-1, -2)) / sqrt_d

    if is_causal:
        seq_q, seq_k = Q.shape[1], K.shape[1]
        causal_mask = torch.triu(
            torch.ones(seq_q, seq_k, device=Q.device, dtype=torch.bool),
            diagonal=1,
        )
        S = S.masked_fill(causal_mask.unsqueeze(0), float("-inf"))

    P = torch.exp(S - L_.unsqueeze(-1))
    dV = torch.matmul(P.transpose(-1, -2), dO_)
    dP = torch.matmul(dO_, V_.transpose(-1, -2))
    dS = P * (dP - D)
    dQ = torch.matmul(dS, K_) / sqrt_d
    dK = torch.matmul(dS.transpose(-1, -2), Q_) / sqrt_d

    return dQ.to(Q.dtype), dK.to(K.dtype), dV.to(V.dtype), None


class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        O, L, sqrt_d = _flash_attention_forward_pytorch(Q, K, V, is_causal=is_causal)
        ctx.save_for_backward(Q, K, V, L, O)
        ctx.is_causal = is_causal
        ctx.sqrt_d = sqrt_d
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, L, O = ctx.saved_tensors
        dQ, dK, dV, _ = attention_backward_impl(
            Q, K, V, L, O, dO, ctx.sqrt_d, ctx.is_causal
        )
        return dQ, dK, dV, None


@triton.jit
def flash_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,
    N_QUERIES,
    N_KEYS,
    scale,
    D: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    is_causal: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_b = tl.program_id(1)

    q_start = pid_q * BLOCK_Q
    q_offsets = q_start + tl.arange(0, BLOCK_Q)
    d_offsets = tl.arange(0, D)
    valid_q = q_offsets < N_QUERIES

    Q_batch_ptr = Q_ptr + pid_b * stride_qb
    K_batch_ptr = K_ptr + pid_b * stride_kb
    V_batch_ptr = V_ptr + pid_b * stride_vb
    O_batch_ptr = O_ptr + pid_b * stride_ob
    L_batch_ptr = L_ptr + pid_b * stride_lb

    q_ptrs = Q_batch_ptr + q_offsets[:, None] * stride_qq + d_offsets[None, :] * stride_qd
    Q_i = tl.load(q_ptrs, mask=valid_q[:, None], other=0.0).to(tl.float32)

    m_i = tl.full((BLOCK_Q,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    O_i = tl.zeros((BLOCK_Q, D), dtype=tl.float32)

    for k_start in tl.range(0, N_KEYS, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        valid_k = k_offsets < N_KEYS

        k_ptrs = K_batch_ptr + k_offsets[:, None] * stride_kk + d_offsets[None, :] * stride_kd
        v_ptrs = V_batch_ptr + k_offsets[:, None] * stride_vk + d_offsets[None, :] * stride_vd

        K_j = tl.load(k_ptrs, mask=valid_k[:, None], other=0.0).to(tl.float32)
        V_j = tl.load(v_ptrs, mask=valid_k[:, None], other=0.0)

        S = tl.dot(Q_i, tl.trans(K_j)) * scale
        score_mask = valid_q[:, None] & valid_k[None, :]
        if is_causal:
            causal_mask = q_offsets[:, None] >= k_offsets[None, :]
            score_mask = score_mask & causal_mask
        S = tl.where(score_mask, S, float("-inf"))

        block_max = tl.max(S, axis=1)
        m_new = tl.maximum(m_i, block_max)
        exp_m = tl.where(valid_q & (m_i != float("-inf")), tl.exp(m_i - m_new), 0.0)
        P = tl.where(score_mask, tl.exp(S - m_new[:, None]), 0.0)
        l_new = l_i * exp_m + tl.sum(P, axis=1)

        alpha = tl.where(valid_q & (l_new > 0), (l_i * exp_m) / l_new, 0.0)
        inv_l = tl.where(valid_q & (l_new > 0), 1.0 / l_new, 0.0)
        O_i = O_i * alpha[:, None] + tl.dot(P.to(V_j.dtype), V_j).to(tl.float32) * inv_l[:, None]

        m_i = m_new
        l_i = l_new

    o_ptrs = O_batch_ptr + q_offsets[:, None] * stride_oq + d_offsets[None, :] * stride_od
    tl.store(o_ptrs, O_i.to(tl.float32), mask=valid_q[:, None])

    lse = tl.where(valid_q & (l_i > 0), m_i + tl.log(l_i), float("-inf"))
    l_ptrs = L_batch_ptr + q_offsets * stride_lq
    tl.store(l_ptrs, lse, mask=valid_q)


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        if Q.device.type != "cuda" or K.device.type != "cuda" or V.device.type != "cuda":
            raise ValueError("FlashAttentionTriton requires CUDA tensors")

        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()

        batch_size, n_queries, head_dim = Q.shape
        n_keys = K.shape[1]
        sqrt_d = math.sqrt(head_dim)

        O = torch.empty_like(Q)
        L = torch.empty((batch_size, n_queries), device=Q.device, dtype=torch.float32)

        # Keep shared-memory usage under common Colab GPU limits.
        if head_dim <= 64:
            BLOCK_Q = 64
            BLOCK_K = 32
        else:
            BLOCK_Q = 32
            BLOCK_K = 32
        grid = (triton.cdiv(n_queries, BLOCK_Q), batch_size)

        flash_fwd_kernel[grid](
            Q,
            K,
            V,
            O,
            L,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            O.stride(0),
            O.stride(1),
            O.stride(2),
            L.stride(0),
            L.stride(1),
            n_queries,
            n_keys,
            1.0 / sqrt_d,
            D=head_dim,
            BLOCK_Q=BLOCK_Q,
            BLOCK_K=BLOCK_K,
            is_causal=is_causal,
            num_warps=4,
            num_stages=1,
        )

        ctx.save_for_backward(Q, K, V, L, O)
        ctx.is_causal = is_causal
        ctx.sqrt_d = sqrt_d
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, L, O = ctx.saved_tensors
        dQ, dK, dV, _ = attention_backward_impl(
            Q, K, V, L, O, dO, ctx.sqrt_d, ctx.is_causal
        )
        return dQ, dK, dV, None


__all__ = [
    "FlashAttentionPytorch",
    "FlashAttentionTriton",
    "attention_backward_impl",
    "flash_fwd_kernel",
]
