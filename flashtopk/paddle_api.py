# Copyright (c) 2026 the FlashTopK authors. Licensed under the Apache License, Version 2.0.
"""Fleet's indexer top-k on Paddle tensors: `cudnn_indexer_topk_fwd` without the dense score matrix.

Fleet gets the DSA indexer's selection from a dense cuDNN score matrix [B, S, S_k] fp32 followed by a radix top-k
(`paddlefleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn.cudnn_indexer_topk_fwd`, backends "paddle"/"deep_select").
`indexer_topk` is a drop-in for that pair: same candidates, same ids, same [-1, -inf] padding, from FlashTopK's exact
selector, which never materializes the scores of the keys it does not select.

For query row (b, t) of `index_q` [B, S, H, 128] and the keys of `index_k` [B, S_k, 128]:

    start      = valid_range[b, t, 0]                          (0 without valid_range)
    causal     = (seq_offset + t + 1) // ratio                 clipped to S_k
    end        = min(valid_range[b, t, 1], causal)             (causal without valid_range)
    candidates = the keys [start, end)                         (empty when start >= end)
    score(u)   = softmax_scale * sum_h weights[b, t, h] * relu(index_q[b, t, h] . index_k[b, u])

i.e. exactly the columns Fleet's cuDNN forward leaves finite (past `end` it writes -inf) and exactly the prefix its
radix kernel is allowed to see (`end - start` of the left-aligned window). The outputs are

    topk_indices [B, S, top_k] int32   global compressed-buffer ids, in canonical order (score descending, id ascending)
    topk_length  [B, S] int32          non-negative entries of topk_indices, i.e. min(top_k, end - start)
    topk_scores  [B, S, top_k] float32 the selected scores, -inf in the padding slots; None unless return_scores

Batch: with B > 1 the batches' key buffers are treated as one concatenated buffer, so one selection call covers every
row (row (b, t) gets window [b * S_k + start, b * S_k + end), and the ids of the batches past the first are folded back
into [0, S_k) afterwards -- two [B-1, S, top_k]-wide elementwise kernels). Nothing is concatenated on the device: the
[B, S_k, 128] operand is a view.

Streaming: everything -- FlashTopK's CuTe launches, the torch ops of this module and Paddle's own allocations -- runs
on Paddle's current CUDA stream for the tensors' device. Torch is pointed at that stream with a `torch.cuda.ExternalStream`
and the kernels' TVM-FFI environment stream is set to the same raw handle (they read it with `TVMFFIEnvGetStream`, which
defaults to the legacy default stream, not torch's), so no stream ever waits on another and no host synchronization is
issued.

Copies: `index_q`, `index_k`, `valid_range` and the outputs cross the paddle/torch boundary as DLPack views of the
same memory (this module writes the outputs Paddle allocated). The call's one copy of an input is `weights`, which
FlashTopK takes as float32: a [B, S, H] bf16 -> float32 cast (2 * B * S * H bytes read, 4 * B * S * H written).
Besides the selection, a call issues only [B * S]-wide kernels -- the lengths (a copy of a kept buffer on the causal
path, two kernels with a document mask), the score scale, and, with a document mask, the window's starts and ends --
plus the [B-1, S, top_k]-wide id fold of a batch. Its own cost, measured on an exclusive B30Z with the selection stubbed
out, is host time: ~95 us a call on the causal path (~115 us with return_scores, ~148 us with a document mask), made of
four to six DLPack views (~2 us each), two Paddle allocations (~4 us each), the weights cast, two or three [B * S]-wide
kernels and the stream setup -- per-op latency on this box, not bytes. Against a hand-wired selector call at S = 1024,
S_k = 1024, H = 32, top_k = 64 (262 us) the full call costs 420 us.
"""

from __future__ import annotations

import contextlib
import functools
from typing import Optional, Tuple

import paddle
import torch
import tvm_ffi

from .topk import MAX_TOPK, Selector

__all__ = ["indexer_topk"]

HEAD_DIM = 128
HEADS = (32, 64)
# One selector per (device, dtype, heads, top_k, pruning, return_scores), its workspace grown to the largest call seen.
# Selector.plan() compiles every CuTe kernel variant the configuration can take, so a configuration's second call is
# the first it does not wait on a compiler for.
_SELECTORS: dict = {}
# Paddle's stream as torch and TVM-FFI take it (torch.cuda.ExternalStream wants an object, the FFI a raw handle).
_STREAMS: dict = {}
_TVM_DEVICES: dict = {}


def _check_inputs(index_q, index_k, weights, valid_range):
    """The shapes, dtypes and device Fleet's validator requires, plus the valid_range contract."""
    for name, tensor in (("index_q", index_q), ("index_k", index_k), ("weights", weights)):
        if not isinstance(tensor, paddle.Tensor):
            raise TypeError(f"{name} must be a paddle.Tensor, got {type(tensor)!r}")
        if not tensor.place.is_gpu_place():
            raise ValueError(f"{name} must be on a GPU, got place {tensor.place}; the kernels are CUDA ones")
    if index_q.dim() != 4:
        raise ValueError(f"index_q must be [B, S, H, D], got {index_q.shape}")
    if index_k.dim() != 3:
        raise ValueError(f"index_k must be [B, S_k, D], got {index_k.shape}")
    if weights.dim() != 3:
        raise ValueError(f"weights must be [B, S, H], got {weights.shape}")
    batch, seq_len, heads, dim = (int(v) for v in index_q.shape)
    batch_k, seq_kv, dim_k = (int(v) for v in index_k.shape)
    batch_w, seq_len_w, heads_w = (int(v) for v in weights.shape)
    if batch_k != batch or batch_w != batch:
        raise ValueError(f"batch mismatch: index_q={index_q.shape}, index_k={index_k.shape}, weights={weights.shape}")
    if dim_k != dim or seq_kv < 1:
        raise ValueError(f"index_k must be [B, S_k >= 1, {dim}], got {index_k.shape}")
    if (seq_len_w, heads_w) != (seq_len, heads):
        raise ValueError(f"weights must be [{batch}, {seq_len}, {heads}], got {weights.shape}")
    if heads not in HEADS:
        raise ValueError(f"H must be one of {HEADS}, got {heads}")
    if dim != HEAD_DIM:
        raise ValueError(f"D must be {HEAD_DIM}, got {dim}")
    if index_q.dtype != paddle.bfloat16 or index_k.dtype != paddle.bfloat16:
        raise ValueError(f"index_q and index_k must be bfloat16, got {index_q.dtype} and {index_k.dtype}")
    if weights.dtype not in (paddle.bfloat16, paddle.float32):
        raise ValueError(f"weights must be bfloat16 or float32, got {weights.dtype}")
    place = index_q.place
    if index_k.place != place or weights.place != place:
        raise ValueError(f"index_q, index_k and weights must share a place, got {place}, {index_k.place}, "
                         f"{weights.place}")
    if valid_range is not None:
        if not isinstance(valid_range, paddle.Tensor):
            raise TypeError(f"valid_range must be a paddle.Tensor or None, got {type(valid_range)!r}")
        if tuple(int(v) for v in valid_range.shape) != (batch, seq_len, 2):
            raise ValueError(f"valid_range must be [{batch}, {seq_len}, 2], got {valid_range.shape}")
        if valid_range.place != place:
            raise ValueError(f"valid_range is on {valid_range.place}, the queries on {place}")
    return batch, seq_len, heads, dim, seq_kv


def _paddle_stream(device_id: int) -> int:
    """The raw cudaStream_t of Paddle's current stream on this device."""
    return int(paddle.device.current_stream(paddle.CUDAPlace(device_id)).__cuda_stream__()[1])


@contextlib.contextmanager
def _on_paddle_stream(device_id: int):
    """Torch and the TVM-FFI kernels on Paddle's current stream of this device, then both back where they were.

    The CuTe kernels take their stream from `TVMFFIEnvGetStream` (see `make_fake_stream(use_tvm_ffi_env_stream=True)`
    in kernels/launch.py), a thread-local that starts out as the legacy default stream and knows nothing about torch,
    so it is set explicitly -- pointing only torch at Paddle's stream would leave the selection on stream 0."""
    raw = _paddle_stream(device_id)
    stream = _STREAMS.get((device_id, raw))
    if stream is None:
        stream = _STREAMS[(device_id, raw)] = torch.cuda.ExternalStream(raw, device=torch.device("cuda", device_id))
    ffi_device = _TVM_DEVICES.get(device_id)
    if ffi_device is None:
        ffi_device = _TVM_DEVICES[device_id] = tvm_ffi.device(f"cuda:{device_id}")
    # set_stream/set_stream back rather than `with torch.cuda.stream(...)`: 2 us against 7, and `set_stream` moves the
    # current device to the stream's own, so a caller on another device is put back where it was too.
    previous = torch.cuda.current_stream(device_id)
    torch.cuda.set_stream(stream)
    try:
        with tvm_ffi.use_raw_stream(ffi_device, raw):
            yield
    finally:
        torch.cuda.set_stream(previous)


def _torch_view(tensor: paddle.Tensor) -> torch.Tensor:
    """The same memory as a torch tensor: a DLPack capsule, ~2 us, against ~15 us for `torch.from_dlpack(tensor)`.

    The capsule owns the Paddle tensor, so the torch view keeps the memory alive, and `to_dlpack` neither copies nor
    synchronizes -- the two libraries are on the same stream, so there is nothing to wait for."""
    return torch.utils.dlpack.from_dlpack(paddle.utils.dlpack.to_dlpack(tensor))


@functools.lru_cache(maxsize=8)
def _causal_windows(batch: int, seq_len: int, seq_kv: int, ratio: int, seq_offset: int, device_id: int):
    """Causal prefix windows over the batches' concatenated key buffers: (batch * seq_len,) int32, or (None, ke) for
    B == 1, where every start is 0."""
    device = torch.device("cuda", device_id)
    query = torch.arange(seq_len, dtype=torch.int32, device=device)
    causal = torch.div(seq_offset + 1 + query, ratio, rounding_mode="floor").clamp_(max=seq_kv)
    if batch == 1:
        return None, causal
    offset = torch.arange(batch, dtype=torch.int32, device=device) * seq_kv
    ends = (causal.unsqueeze(0) + offset.unsqueeze(1)).reshape(batch * seq_len)
    starts = offset.unsqueeze(1).expand(batch, seq_len).reshape(batch * seq_len)
    return starts, ends


@functools.lru_cache(maxsize=8)
def _tail_offsets(batch: int, seq_kv: int, device_id: int) -> torch.Tensor:
    """b * S_k for the batches 1 .. B-1, [B-1, 1, 1] int32, to fold the concatenated buffer's ids back per batch."""
    device = torch.device("cuda", device_id)
    return (torch.arange(1, batch, dtype=torch.int32, device=device) * seq_kv).reshape(batch - 1, 1, 1)


@functools.lru_cache(maxsize=8)
def _causal_length(batch: int, seq_len: int, seq_kv: int, ratio: int, seq_offset: int, top_k: int,
                   device_id: int) -> torch.Tensor:
    """The causal path's length [batch * seq_len] int32: min(top_k, causal), the same for every batch."""
    _, ends = _causal_windows(1, seq_len, seq_kv, ratio, seq_offset, device_id)
    return ends.clamp(max=top_k).repeat(batch)


@functools.lru_cache(maxsize=8)
def _zeros(rows: int, device_id: int) -> torch.Tensor:
    return torch.zeros(rows, dtype=torch.int32, device=torch.device("cuda", device_id))


def _windows(device_id, valid_range, batch, seq_len, seq_kv, ratio, seq_offset):
    """(ks, ke) int32 [batch * seq_len] over the concatenated key buffers, in the FlashTopK convention: row r's keys
    are [ks[r], ke[r]) of the flattened KV.

    The causal windows depend on nothing per call, so they are kept -- a row's start is 0 with B == 1 and its batch's
    first key id otherwise. A document mask's starts and ends do depend on the call, and cost a handful of elementwise
    kernels over [batch * seq_len] (nothing next to a selection)."""
    starts, ends = _causal_windows(batch, seq_len, seq_kv, ratio, seq_offset, device_id)
    if valid_range is None:
        return (_zeros(batch * seq_len, device_id) if starts is None else starts), ends
    if valid_range.dtype != paddle.int32:
        valid_range = valid_range.cast("int32")
    window = _torch_view(valid_range).reshape(batch * seq_len, 2)
    start, end = window[:, 0], window[:, 1]
    if starts is not None:
        start, end = start + starts, end + starts
    # A start past the last key leaves the window empty (ks >= ke), as Fleet's radix kernel sees a count of 0.
    return start.clamp(min=0), torch.minimum(ends, end)


def _selector(device_id: int, heads: int, top_k: int, pruning: Optional[str], return_scores: bool, rows: int,
              seq_kv: int) -> Selector:
    """The cached selector of this configuration, planned (workspace allocated, kernels compiled) for this call."""
    key = (device_id, torch.bfloat16, heads, top_k, pruning, return_scores)
    selector = _SELECTORS.get(key)
    if selector is None:
        selector = _SELECTORS[key] = Selector(top_k, num_heads=heads, dtype=torch.bfloat16,
                                              device=torch.device("cuda", device_id), pruning=pruning,
                                              return_scores=return_scores)
    if selector.max_rows < min(rows, selector.chunk_size) or selector.max_kv < seq_kv:
        selector.plan(rows, seq_kv)
    return selector


def indexer_topk(index_q: paddle.Tensor, index_k: paddle.Tensor, weights: paddle.Tensor, top_k: int, *,
                 softmax_scale: float, ratio: int = 1, seq_offset: int = 0,
                 valid_range: Optional[paddle.Tensor] = None, return_scores: bool = False,
                 pruning: Optional[str] = None
                 ) -> Tuple[paddle.Tensor, paddle.Tensor, Optional[paddle.Tensor]]:
    """Fleet's `cudnn_indexer_forward` + `_select_indexer_top_k` as one FlashTopK call.

    Args:
        index_q: [B, S, H, 128] bfloat16 indexer queries (H 32 or 64, on SM100 for 64).
        index_k: [B, S_k, 128] bfloat16 compressed indexer keys.
        weights: [B, S, H] bfloat16 (or float32) per-head weights.
        top_k: entries selected per query row, 1 .. 4096.
        softmax_scale: score scale, D ** -0.5 * indexer_softmax_scale in Fleet's terms; applied to the scores after the
            head sum, as the cuDNN path does.
        ratio: compression ratio of the causal limit, `(seq_offset + t + 1) // ratio`.
        seq_offset: global query position of the first row (context-parallel causal mode).
        valid_range: optional [B, S, 2] int32 document-mask window [start, end) per row; None for a causal prefix.
        return_scores: also return the selected scores (fp32), -inf in the padding slots.
        pruning: FlashTopK pruning mode ("off", "always", "adaptive"); None takes FlashTopK's default.

    Returns:
        (topk_indices [B, S, top_k] int32 global ids with -1 padding, topk_length [B, S] int32, topk_scores or None).
    """
    batch, seq_len, heads, dim, seq_kv = _check_inputs(index_q, index_k, weights, valid_range)
    top_k, ratio, seq_offset = int(top_k), int(ratio), int(seq_offset)
    if not 1 <= top_k <= MAX_TOPK:
        raise ValueError(f"top_k must be in [1, {MAX_TOPK}], got {top_k}")
    if ratio < 1:
        raise ValueError(f"ratio must be positive, got {ratio}")
    if seq_offset < 0:
        raise ValueError(f"seq_offset must be >= 0, got {seq_offset}")
    scale = float(softmax_scale)
    rows = batch * seq_len
    device_id = index_q.place.gpu_device_id()

    # Paddle allocates the results and torch writes them in place: no copy of an output either.
    indices = paddle.empty([batch, seq_len, top_k], dtype="int32")
    length = paddle.empty([batch, seq_len], dtype="int32")
    scores = paddle.empty([batch, seq_len, top_k], dtype="float32") if return_scores else None

    with paddle.no_grad(), _on_paddle_stream(device_id):
        selector = _selector(device_id, heads, top_k, pruning, return_scores, rows, seq_kv)
        query = _torch_view(index_q).reshape(rows, heads, dim)
        key = _torch_view(index_k).reshape(batch * seq_kv, dim)
        weight = _torch_view(weights).reshape(rows, heads)
        if weight.dtype != torch.float32:
            weight = weight.to(torch.float32)  # the score contract's one input copy
        starts, ends = _windows(device_id, valid_range, batch, seq_len, seq_kv, ratio, seq_offset)
        index_out = _torch_view(indices).reshape(rows, top_k)
        score_out = _torch_view(scores).reshape(rows, top_k) if return_scores else None
        selector.run(query, key, weight, starts, ends, None,
                     (score_out, index_out) if return_scores else index_out)
        if batch > 1:
            # The kernels see the batches' key buffers as one, so batch b's ids come back b * S_k too high. Fold the
            # offset back on the rows of the batches past the first; a -1 there goes to -1 - b * S_k < -1 and clamps
            # back to -1, while a real id (>= b * S_k) lands in [0, S_k).
            shifted = index_out.view(batch, seq_len, top_k)[1:]
            shifted.sub_(_tail_offsets(batch, seq_kv, device_id)).clamp_min_(-1)
        if score_out is not None and scale != 1.0:
            score_out.mul_(scale)
        # A row's slots are non-negative exactly for its min(top_k, window) best keys (every key in a window scores
        # finite), which is the value of Fleet's (topk_indices >= 0).sum(-1). The causal path's lengths are the same on
        # every call, so they are copied rather than recomputed.
        length_out = _torch_view(length).reshape(rows)
        if valid_range is None:
            length_out.copy_(_causal_length(batch, seq_len, seq_kv, ratio, seq_offset, top_k, device_id))
        else:
            torch.sub(ends, starts, out=length_out)
            length_out.clamp_(min=0, max=top_k)
    return indices, length, scores
