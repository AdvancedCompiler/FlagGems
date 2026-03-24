import torch
import triton
import triton.language as tl


def cdiv(x: int, y: int) -> int:
    return (x + y - 1) // y


@triton.autotune(
    configs=[
        # ---- BLOCK_D=128, NUM_D_TILES=1 (dim=128) ----
        triton.Config(
            {"BLOCK_KV": 16, "BLOCK_D": 128, "NUM_D_TILES": 1, "HEADS_UNROLL": 4},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_KV": 16, "BLOCK_D": 128, "NUM_D_TILES": 1, "HEADS_UNROLL": 8},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_KV": 16, "BLOCK_D": 128, "NUM_D_TILES": 1, "HEADS_UNROLL": 4},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_KV": 32, "BLOCK_D": 128, "NUM_D_TILES": 1, "HEADS_UNROLL": 4},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_KV": 32, "BLOCK_D": 128, "NUM_D_TILES": 1, "HEADS_UNROLL": 8},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_KV": 32, "BLOCK_D": 128, "NUM_D_TILES": 1, "HEADS_UNROLL": 4},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_KV": 64, "BLOCK_D": 128, "NUM_D_TILES": 1, "HEADS_UNROLL": 4},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_KV": 64, "BLOCK_D": 128, "NUM_D_TILES": 1, "HEADS_UNROLL": 8},
            num_warps=8,
            num_stages=2,
        ),
        # ---- BLOCK_D=64, NUM_D_TILES=2 ----
        triton.Config(
            {"BLOCK_KV": 16, "BLOCK_D": 64, "NUM_D_TILES": 2, "HEADS_UNROLL": 4},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_KV": 32, "BLOCK_D": 64, "NUM_D_TILES": 2, "HEADS_UNROLL": 4},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_KV": 32, "BLOCK_D": 64, "NUM_D_TILES": 2, "HEADS_UNROLL": 8},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_KV": 64, "BLOCK_D": 64, "NUM_D_TILES": 2, "HEADS_UNROLL": 4},
            num_warps=4,
            num_stages=2,
        ),
    ],
    key=["heads", "dim", "block_size"],
)
@triton.jit
def fp8_paged_mqa_logits_kernel(
    q_ptr,
    kv_ptr,
    weights_ptr,
    logits_ptr,
    block_tables_ptr,
    context_lens_ptr,
    stride_qb,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kvblk,
    stride_kvpos,
    stride_kvone,
    stride_kvbyte,
    stride_wrow,
    stride_wh,
    stride_lrow,
    stride_lcol,
    stride_btb,
    stride_bts,
    next_n: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
    block_size: tl.constexpr,
    max_model_len,
    dim_plus_4: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_D_TILES: tl.constexpr,
    HEADS_UNROLL: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_kv_tile = tl.program_id(1)

    batch_idx = pid_row // next_n
    next_n_idx = pid_row % next_n

    context_len = tl.load(context_lens_ptr + batch_idx)
    query_seq_pos = context_len - next_n + next_n_idx

    kv_start = pid_kv_tile * BLOCK_KV
    if kv_start >= context_len:
        return

    offs_kv = tl.arange(0, BLOCK_KV)
    kv_global_pos = kv_start + offs_kv

    context_mask = kv_global_pos < context_len
    causal_mask = kv_global_pos <= query_seq_pos
    valid_mask = context_mask & causal_mask

    phys_block_idx = kv_global_pos // block_size
    intra_block_pos = kv_global_pos % block_size

    phys_block_ids = tl.load(
        block_tables_ptr + batch_idx * stride_btb + phys_block_idx * stride_bts,
        mask=valid_mask,
        other=0,
    )

    kv_base = phys_block_ids * stride_kvblk + intra_block_pos * stride_kvpos

    scale_addr = kv_base + dim * stride_kvbyte
    b0 = tl.load(kv_ptr + scale_addr, mask=valid_mask, other=0).to(tl.uint32)
    b1 = tl.load(kv_ptr + scale_addr + stride_kvbyte, mask=valid_mask, other=0).to(
        tl.uint32
    )
    b2 = tl.load(kv_ptr + scale_addr + 2 * stride_kvbyte, mask=valid_mask, other=0).to(
        tl.uint32
    )
    b3 = tl.load(kv_ptr + scale_addr + 3 * stride_kvbyte, mask=valid_mask, other=0).to(
        tl.uint32
    )
    scale_u32 = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
    scale_f32 = scale_u32.to(tl.float32, bitcast=True)  # [BLOCK_KV]

    logit_accum = tl.zeros([BLOCK_KV], dtype=tl.float32)
    offs_d = tl.arange(0, BLOCK_D)

    for d_tile in tl.static_range(0, NUM_D_TILES):
        d_offs = d_tile * BLOCK_D + offs_d
        d_mask = d_offs < dim

        kv_byte_ptrs = kv_ptr + kv_base[:, None] + d_offs[None, :] * stride_kvbyte
        load_mask = valid_mask[:, None] & d_mask[None, :]
        kv_u8 = tl.load(kv_byte_ptrs, mask=load_mask, other=0)
        kv_fp8 = kv_u8.to(tl.float8e4nv, bitcast=True)
        kv_f32 = kv_fp8.to(tl.float32)

        kv_scaled = kv_f32 * scale_f32[:, None]

        q_base = (
            q_ptr + batch_idx * stride_qb + next_n_idx * stride_qn + d_offs * stride_qd
        )

        for h in range(heads):
            q_vals = tl.load(q_base + h * stride_qh, mask=d_mask, other=0.0).to(
                tl.float32
            )

            w = tl.load(weights_ptr + pid_row * stride_wrow + h * stride_wh)

            partial_dot = tl.sum(kv_scaled * q_vals[None, :], axis=1)

            if NUM_D_TILES == 1:
                dot_relu = tl.maximum(partial_dot, 0.0)
                logit_accum += dot_relu * w

    if NUM_D_TILES > 1:
        logit_accum2 = tl.zeros([BLOCK_KV], dtype=tl.float32)

        for h in range(heads):
            w = tl.load(weights_ptr + pid_row * stride_wrow + h * stride_wh)
            dot = tl.zeros([BLOCK_KV], dtype=tl.float32)

            for d_tile2 in tl.static_range(0, NUM_D_TILES):
                d_offs2 = d_tile2 * BLOCK_D + offs_d
                d_mask2 = d_offs2 < dim

                q_ptrs2 = (
                    q_ptr
                    + batch_idx * stride_qb
                    + next_n_idx * stride_qn
                    + h * stride_qh
                    + d_offs2 * stride_qd
                )
                q_vals2 = tl.load(q_ptrs2, mask=d_mask2, other=0.0).to(tl.float32)

                kv_byte_ptrs2 = (
                    kv_ptr + kv_base[:, None] + d_offs2[None, :] * stride_kvbyte
                )
                load_mask2 = valid_mask[:, None] & d_mask2[None, :]
                kv_u82 = tl.load(kv_byte_ptrs2, mask=load_mask2, other=0)
                kv_fp82 = kv_u82.to(tl.float8e4nv, bitcast=True)
                kv_f322 = kv_fp82.to(tl.float32)

                dot += tl.sum(kv_f322 * q_vals2[None, :], axis=1)

            dot = dot * scale_f32
            dot = tl.maximum(dot, 0.0)
            logit_accum2 += dot * w

        logit_accum = logit_accum2

    out_vals = tl.where(valid_mask, logit_accum, float("-inf"))
    out_ptrs = logits_ptr + pid_row * stride_lrow + kv_global_pos * stride_lcol
    out_mask = valid_mask & (kv_global_pos < max_model_len)
    tl.store(out_ptrs, out_vals, mask=out_mask)


def fp8_paged_mqa_logits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    assert q.is_cuda and kv_cache.is_cuda and weights.is_cuda
    assert context_lens.is_cuda and block_tables.is_cuda

    batch_size, next_n, heads, dim = q.size()
    num_blocks, block_size, one, dim_plus_4 = kv_cache.size()

    assert one == 1, "KV cache must have num_heads=1 (MQA)"
    assert dim_plus_4 == dim + 4, f"KV dim error: {dim_plus_4} != {dim}+4"
    assert weights.shape == (batch_size * next_n, heads), "Weights shape mismatch"
    assert kv_cache.dtype == torch.uint8, "KV cache must be uint8 (packed FP8+scale)"
    assert context_lens.dtype == torch.int32, "Context lens must be int32"
    assert block_tables.dtype == torch.int32, "Block tables must be int32"

    q_contig = q.contiguous()
    kv_contig = kv_cache.contiguous()
    weights_contig = weights.contiguous()
    context_lens_contig = context_lens.contiguous()
    block_tables_contig = block_tables.contiguous()

    logits = torch.full(
        (batch_size * next_n, max_model_len),
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )

    max_context = int(context_lens.max().item())

    def grid(meta):
        BLOCK_KV = meta["BLOCK_KV"]
        num_kv_tiles = cdiv(max_context, BLOCK_KV)
        return (batch_size * next_n, num_kv_tiles)

    fp8_paged_mqa_logits_kernel[grid](
        q_contig,
        kv_contig,
        weights_contig,
        logits,
        block_tables_contig,
        context_lens_contig,
        q_contig.stride(0),
        q_contig.stride(1),
        q_contig.stride(2),
        q_contig.stride(3),
        kv_contig.stride(0),
        kv_contig.stride(1),
        kv_contig.stride(2),
        kv_contig.stride(3),
        weights_contig.stride(0),
        weights_contig.stride(1),
        logits.stride(0),
        logits.stride(1),
        block_tables_contig.stride(0),
        block_tables_contig.stride(1),
        next_n,
        heads,
        dim,
        block_size,
        max_model_len,
        dim_plus_4,
    )

    return logits
