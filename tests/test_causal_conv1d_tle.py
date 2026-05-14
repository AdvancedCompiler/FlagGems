import itertools

import pytest
import torch
import torch.nn.functional as F

import flag_gems
from flag_gems.utils.triton_version_utils import HAS_TLE

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

if flag_gems.vendor_name == "ascend":
    from flag_gems.runtime.backend._ascend.fused import (
        causal_conv1d_fn,
        causal_conv1d_update_npu,
    )
else:
    causal_conv1d_fn = None
    causal_conv1d_update_npu = None

PAD_SLOT_ID = -1

_DECODE_CASES = list(
    itertools.product(
        [3, 64],
        [True, False],
        [2048 + 16, 4096],
        [3, 4],
        [1, 3],
        [False, True],
        [True],
        [torch.bfloat16],
    )
)
_PREFILL_CASES = list(
    itertools.product(
        [4, 10],
        [True, False],
        [64, 4096],
        [8, 249, 4096],
        [4],
        [True],
        [True],
        [torch.bfloat16],
    )
)

if QUICK_MODE:
    DECODE_CASES = [_DECODE_CASES[0], _DECODE_CASES[-1]]
    PREFILL_CASES = [_PREFILL_CASES[0], _PREFILL_CASES[-1]]
else:
    DECODE_CASES = _DECODE_CASES
    PREFILL_CASES = _PREFILL_CASES


def _atol(dtype: torch.dtype) -> float:
    if dtype == torch.bfloat16:
        return 5e-2
    if dtype == torch.float16:
        return 5e-3
    return 1e-3


def _make_query_start_loc(total_tokens: int, padded_batch: int) -> list[int]:
    if padded_batch == 1:
        return [total_tokens]
    eos_pos = torch.randperm(total_tokens - 1)[: padded_batch - 1].sort().values
    boundaries = torch.cat(
        [
            torch.tensor([-1], dtype=torch.int64),
            eos_pos.to(torch.int64),
            torch.tensor([total_tokens - 1], dtype=torch.int64),
        ]
    )
    return torch.diff(boundaries).tolist()


def causal_conv1d_update_ref(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
    cache_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    state_len = conv_state.shape[-1]
    assert conv_state.shape == (batch, dim, state_len)
    assert weight.shape == (dim, width)
    if cache_seqlens is None:
        x_new = torch.cat([conv_state, x], dim=-1).to(weight.dtype)
        conv_state.copy_(x_new[:, :, -state_len:])
    else:
        width_idx = torch.arange(
            -(width - 1), 0, dtype=torch.long, device=x.device
        ).unsqueeze(0) + cache_seqlens.unsqueeze(1)
        width_idx = (
            torch.remainder(width_idx, state_len).unsqueeze(1).expand(-1, dim, -1)
        )
        x_new = torch.cat([conv_state.gather(2, width_idx), x], dim=-1).to(weight.dtype)
        copy_idx = torch.arange(seqlen, dtype=torch.long, device=x.device).unsqueeze(0)
        copy_idx = copy_idx + cache_seqlens.unsqueeze(1)
        copy_idx = torch.remainder(copy_idx, state_len).unsqueeze(1).expand(-1, dim, -1)
        conv_state.scatter_(2, copy_idx, x)
    out = F.conv1d(x_new, weight.unsqueeze(1), bias, padding=0, groups=dim)[
        :, :, -seqlen:
    ]
    if unsqueeze:
        out = out.squeeze(-1)
    return (out if activation is None else F.silu(out)).to(dtype=dtype_in)


def causal_conv1d_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    initial_states: torch.Tensor | None = None,
    return_final_states: bool = False,
    final_states_out: torch.Tensor | None = None,
    activation: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape
    if initial_states is None:
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
    else:
        x = torch.cat([initial_states, x], dim=-1)
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=dim)
    out = out[..., :seqlen]
    if return_final_states:
        final_states = F.pad(x, (width - 1 - x.shape[-1], 0)).to(dtype_in)
        if final_states_out is not None:
            final_states_out.copy_(final_states)
        else:
            final_states_out = final_states
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return (out, None) if not return_final_states else (out, final_states_out)


def causal_conv1d_fn_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    activation: str | None,
    pad_slot_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    state_ref = conv_states.clone()
    seqlens = torch.diff(query_start_loc.to(torch.int64)).tolist()
    outputs = []
    start = 0
    x_3d = x.unsqueeze(0)
    for idx, seqlen in enumerate(seqlens):
        end = start + seqlen
        state_idx = int(cache_indices[idx].item())
        x_chunk = x_3d[:, :, start:end]
        start = end
        if state_idx == pad_slot_id:
            continue
        initial_states = (
            state_ref[state_idx].unsqueeze(0)
            if bool(has_initial_state[idx].item())
            else None
        )
        out, _ = causal_conv1d_ref(
            x_chunk,
            weight,
            bias,
            initial_states=initial_states,
            return_final_states=True,
            final_states_out=state_ref[state_idx].unsqueeze(0),
            activation=activation,
        )
        outputs.append(out.squeeze(0))
    if outputs:
        return torch.cat(outputs, dim=-1), state_ref
    return x.new_empty((x.shape[0], 0)), state_ref


@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend",
    reason="Ascend-only fused causal conv1d decode test",
)
@pytest.mark.skipif(not HAS_TLE, reason="Triton TLE support is unavailable")
@pytest.mark.causal_conv1d_update_npu
@pytest.mark.parametrize(
    "batch_size, with_padding, dim, width, seqlen, has_bias, silu_activation, dtype",
    DECODE_CASES,
)
def test_causal_conv1d_update_npu(
    batch_size,
    with_padding,
    dim,
    width,
    seqlen,
    has_bias,
    silu_activation,
    dtype,
):
    utils.init_seed(1000 + batch_size + dim + width + seqlen + int(with_padding))
    device = flag_gems.device
    padding = 5 if with_padding else 0
    padded_batch_size = batch_size + padding
    total_entries = 10 * batch_size
    activation = "silu" if silu_activation else None

    x = (
        torch.randn((padded_batch_size, seqlen, dim), device=device, dtype=dtype)
        .transpose(1, 2)
        .contiguous()
    )
    conv_state_indices = torch.randperm(total_entries, device=device)[:batch_size].to(
        torch.int32
    )
    padded_state_indices = torch.cat(
        [
            conv_state_indices,
            torch.full((padding,), PAD_SLOT_ID, dtype=torch.int32, device=device),
        ]
    )
    conv_state = (
        torch.randn((total_entries, width - 1, dim), device=device, dtype=dtype)
        .transpose(1, 2)
        .contiguous()
    )
    weight = torch.randn((dim, width), device=device, dtype=dtype)
    bias = torch.randn((dim,), device=device, dtype=dtype) if has_bias else None

    ref_x = utils.to_reference(x, False)
    ref_conv_state_indices = utils.to_reference(conv_state_indices, False).to(
        torch.long
    )
    ref_conv_state = utils.to_reference(conv_state, False)
    ref_weight = utils.to_reference(weight, False)
    ref_bias = utils.to_reference(bias, False)
    ref_state = ref_conv_state.index_select(0, ref_conv_state_indices).clone()

    out = causal_conv1d_update_npu(
        x,
        conv_state,
        weight,
        bias,
        activation=activation,
        conv_state_indices=padded_state_indices,
        pad_slot_id=PAD_SLOT_ID,
    )
    ref_out = causal_conv1d_update_ref(
        ref_x[:batch_size].clone(),
        ref_state,
        ref_weight,
        ref_bias,
        activation=activation,
    )

    utils.gems_assert_close(out[:batch_size], ref_out, dtype, atol=_atol(dtype))
    utils.gems_assert_close(
        conv_state.index_select(0, conv_state_indices.to(torch.long)),
        ref_state,
        dtype,
        atol=_atol(dtype),
    )


@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend",
    reason="Ascend-only fused causal conv1d prefill test",
)
@pytest.mark.skipif(not HAS_TLE, reason="Triton TLE support is unavailable")
@pytest.mark.causal_conv1d_fn
@pytest.mark.parametrize(
    "batch, with_padding, dim, seqlen, width, has_bias, silu_activation, dtype",
    PREFILL_CASES,
)
def test_causal_conv1d_fn(
    batch,
    with_padding,
    dim,
    seqlen,
    width,
    has_bias,
    silu_activation,
    dtype,
):
    utils.init_seed(2000 + batch + dim + seqlen + width + int(with_padding))
    device = flag_gems.device
    activation = "silu" if silu_activation else None
    padding = 3 if with_padding else 0
    padded_batch = batch + padding
    total_entries = batch * 10

    seqlens = _make_query_start_loc(seqlen, padded_batch)
    query_start_loc = torch.tensor(
        [0]
        + list(torch.cumsum(torch.tensor(seqlens, dtype=torch.int32), dim=0).tolist()),
        dtype=torch.int32,
        device=device,
    )
    x = torch.randn((dim, seqlen), device=device, dtype=dtype)
    weight = torch.randn((dim, width), device=device, dtype=dtype)
    bias = torch.randn((dim,), device=device, dtype=dtype) if has_bias else None
    conv_states = (
        torch.randn((total_entries, width - 1, dim), device=device, dtype=dtype)
        .transpose(1, 2)
        .contiguous()
    )
    has_initial_state = torch.randint(
        0, 2, (padded_batch,), dtype=torch.bool, device=device
    )
    state_indices = torch.randperm(total_entries, device=device)[:batch].to(torch.int32)
    padded_state_indices = torch.cat(
        [
            state_indices,
            torch.full((padding,), PAD_SLOT_ID, dtype=torch.int32, device=device),
        ],
        dim=0,
    )

    ref_x = utils.to_reference(x, False)
    ref_weight = utils.to_reference(weight, False)
    ref_bias = utils.to_reference(bias, False)
    ref_conv_states = utils.to_reference(conv_states, False)
    ref_query_start_loc = utils.to_reference(query_start_loc, False)
    ref_padded_state_indices = utils.to_reference(padded_state_indices, False)
    ref_has_initial_state = utils.to_reference(has_initial_state, False)
    ref_state_indices = utils.to_reference(state_indices, False).to(torch.long)

    out = causal_conv1d_fn(
        x,
        weight,
        bias=bias,
        conv_states=conv_states,
        query_start_loc=query_start_loc,
        cache_indices=padded_state_indices,
        has_initial_state=has_initial_state,
        activation=activation,
        pad_slot_id=PAD_SLOT_ID,
    )
    ref_out, ref_states_after = causal_conv1d_fn_ref(
        ref_x,
        ref_weight,
        ref_bias,
        ref_conv_states,
        ref_query_start_loc,
        ref_padded_state_indices,
        ref_has_initial_state,
        activation,
        PAD_SLOT_ID,
    )

    utils.gems_assert_close(
        out[:, : ref_out.shape[-1]], ref_out, dtype, atol=_atol(dtype)
    )
    utils.gems_assert_close(
        conv_states.index_select(0, state_indices.to(torch.long)),
        ref_states_after.index_select(0, ref_state_indices),
        dtype,
        atol=_atol(dtype),
    )
