"""Paged KV cache pool for the 16 full-attention layers + 1 MTP layer.

Design:

* layout ``[n_layers, n_pages, 2, page_size, num_kv_heads, head_dim]`` -- this
  *is* FlashInfer's "NHD" per-page layout (page_size, num_heads, head_dim)
  with a leading ``n_layers`` axis stacked on top and a ``2`` for K/V. It is
  the same 5-D-per-layer shape ``kernels/microbench/attn_decode_bench.py``
  already builds (``_build_paged_kv``), just with the layer axis added so one
  pool serves all 17 attention/MTP layers. We picked NHD (not HND) because
  it is FlashInfer's default (``kv_layout="NHD"``) and is what
  ``BatchDecodeWithPagedKVCacheWrapper``/``BatchPrefillWithPagedKVCacheWrapper``
  consume directly with zero transposition -- see ``flashinfer_attn.py``.
* ``page_size`` is a constructor parameter (16 or 64 are typical; either
  works, nothing here is hardcoded to 16). Unlike vLLM's mamba-page-matching
  trick (784-token blocks), this pool is deliberately independent of the SSM
  slot pool -- KV pages and SSM slots are allocated by
  unrelated allocators with unrelated granularities.
* bf16 and fp8 (``float8_e4m3fn``) storage modes. fp8 carries a
  ``[n_layers, n_pages, 2, num_kv_heads]`` fp32 scale tensor (a superset of
  "per-layer/per-head": it is per-page *and*
  per-layer/per-head). See ``PagedKVPool`` docstring for the scale policy.
* a free-list page allocator (``PageAllocator``) and a host-side per-sequence
  page table (``[max_seqs, max_pages_per_seq]`` int32, -1 = unmapped).
* ``append_kv`` is a pure tensor scatter (no Python-side branching that
  depends on tensor *values*), so it is safe to call inside a captured CUDA
  graph once pages are pre-allocated. Page allocation (``ensure_capacity``)
  is host-side bookkeeping and must happen *outside* the graph, exactly like
  FlashInfer's plan()-outside/run()-inside split.
* ``truncate`` is the O(1) rollback pointer move MTP verify/commit needs:
  the 16 attention layers roll back trivially (move the KV write pointer;
  free pages) -- no data movement, no re-quantization.

The admission cost contract
---------------------------
``ensure_capacity`` is called once per decode row per step (256 times at
concurrency 256).  Two naive implementations turn that into a 26-49 ms host
phase:

1. Computing ``pages_allocated`` as ``int((page_table_host[slot] >= 0).sum())``
   is a CPU reduction over ``max_pages_per_seq`` per call, and the scheduler
   calls it twice per row.  It is instead an O(1) read of ``_n_alloc``, a
   plain Python list kept exact by the four writers (``alloc_slot``,
   ``ensure_capacity``, ``free_pages``, ``reclaim_trailing_pages``, plus
   ``map_page``).
2. ``page_table[slot, have:need] = new_pages.to(device)`` is a **pageable**
   H2D per row that needs one.  PyTorch implements that as
   ``cudaMemcpyAsync`` + ``cudaStreamSynchronize`` (``memcpy_and_sync``), so
   under ``--async-scheduling`` -- where the previous step's 44 ms of kernels
   are queued -- the *first* row of a step that crosses a page boundary
   blocks the host for the rest of that step (admission measured going from
   26 to 49 ms).

The implementation here is a **staged, batched** write: allocation updates the host
mirror and the O(1) counter and records a dirty ``(slot, start, end)`` range;
``flush_page_table`` turns every dirty range into **one** ``index_copy_`` fed
by two pinned, ``non_blocking=True`` H2D copies.  A step in which no row
crosses a page boundary -- the overwhelmingly common case at page_size 16 and
one token per row -- does **zero** device work and never touches CUDA at all.
Use ``defer_page_table_writes()`` to hold the flush until the end of a whole
admission pass; outside that scope ``ensure_capacity`` flushes immediately, so
every pre-existing caller sees byte-identical behaviour.
"""

from __future__ import annotations

import heapq
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

#: Runtime kill switch for the staged admission path.  ``QWENFAST_WS_S3=0``
#: restores the unstaged admission path exactly -- the ``pages_allocated``
#: reduction, the per-row pageable H2D, and the device round trip in the
#: FlashInfer plan -- **in the same binary**.  That makes a before/after
#: measurement an A/B of one change rather than of two builds, and it is the
#: escape hatch if the staged write ever has to be turned off in production
#: without a redeploy.
WS_S3 = os.environ.get("QWENFAST_WS_S3", "1") not in ("0", "false", "False")

import torch

# float8_e4m3fn was added in torch 2.1; guard anyway since this module must
# import cleanly on any torch build (CPU-only dev machines included).
FP8_DTYPE = getattr(torch, "float8_e4m3fn", None)
FP8_MAX = 448.0  # e4m3 max-representable magnitude (2**8 * 1.75)

_DTYPE_NAMES = ("bf16", "fp8")


@dataclass
class KVPoolConfig:
    """Shapes match the Qwen3.8-27B attention layers by default."""

    n_layers: int = 17  # 16 full-attention layers + 1 MTP layer
    num_kv_heads: int = 4
    head_dim: int = 256
    page_size: int = 16  # tokens/page; 16 or 64 are typical -- parameterized
    n_pages: int = 4096
    max_seqs: int = 512
    max_pages_per_seq: int = 256
    dtype: str = "bf16"  # "bf16" | "fp8"
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.dtype not in _DTYPE_NAMES:
            raise ValueError(f"KVPoolConfig.dtype must be one of {_DTYPE_NAMES}, got {self.dtype!r}")
        if self.page_size <= 0:
            raise ValueError("page_size must be positive")


class PageAllocator:
    """Free-list allocator over a flat ``[0, n_pages)`` id space.

    A "page" is a ``page_size``-token slot that exists at the same index in
    *every* layer's KV tensor (the layout is page-major so one page index
    addresses all 17 layers at once). Bookkeeping is host-side
    Python -- only the resulting page-index tensors (``kv_indices``,
    ``page_table`` rows) ever need to reach the device, which is why this
    class holds no CUDA state at all.
    """

    def __init__(self, n_pages: int):
        if n_pages <= 0:
            raise ValueError("n_pages must be positive")
        self.n_pages = n_pages
        # A **min-heap**, not a FIFO free list, so ``alloc(n)`` always returns
        # the n lowest free page ids -- i.e. a fresh sequence gets a
        # *contiguous* run whenever the pool is not badly fragmented.
        #
        # This is a measured effect, not tidiness. A FIFO free list
        # (`self._free[:n]` / `self._free.extend(freed)`) permutes the id
        # space after one alloc/free cycle, so the next sequence's 129 pages
        # are scattered across the pool and FlashInfer's paged decode reads
        # them as 129 unrelated 1 MiB strides. With FIFO, a B=32
        # ctx-2048 step ran 18.45 ms on a pristine pool and 21.8-30.1 ms on
        # a recycled one (+18% to +63%), while B=8 and B=64 (whose page runs
        # happened to stay contiguous) were stable to +/-0.1%. Allocating
        # lowest-first restores locality after an arbitrary free pattern and
        # makes the number reproducible.
        #
        # `list(range(n_pages))` is already a valid min-heap.
        self._free: List[int] = list(range(n_pages))

    @property
    def num_free(self) -> int:
        return len(self._free)

    def alloc(self, n: int) -> List[int]:
        if n < 0:
            raise ValueError("cannot allocate a negative number of pages")
        if n > len(self._free):
            raise RuntimeError(
                f"page pool exhausted: requested {n} pages, {len(self._free)} free of {self.n_pages}"
            )
        if n == 0:
            return []
        return [heapq.heappop(self._free) for _ in range(n)]

    def free(self, pages: Iterable[int]) -> None:
        for p in pages:
            heapq.heappush(self._free, int(p))


class PagedKVPool:
    """Paged bf16/fp8 KV cache for the 16 attention layers + 1 MTP layer.

    fp8 scale policy (deliberately simple): the scale
    tensor is ``[n_layers, n_pages, 2, num_kv_heads]`` fp32, one value per
    (layer, page, K-or-V, head). It defaults to 1.0 and is **not**
    recomputed automatically on every ``append_kv`` -- recomputing an
    amax-based scale after a page already holds some tokens would require
    re-quantizing the earlier tokens too (their old int8 codes were divided
    by the old scale), which is neither an O(1) op nor CUDA-graph-safe.
    Instead:

    * ``calibrate_page_scale`` lets a caller set a page's scale from a
      representative sample (e.g. the first chunk written into it, or an
      offline calibration pass) *before* steady-state appends land on it.
    * subsequent ``append_kv`` calls quantize against whatever scale is
      currently stored for that page, and clamp to the fp8 dynamic range.

    This provides fp8 e4m3 with per-layer/per-head scales as the common
    case -- one calibration per page, mostly stable within a layer/head --
    while keeping the finer ``[n_layers, n_pages, 2, num_kv_heads]`` tensor
    shape.
    """

    def __init__(self, cfg: KVPoolConfig):
        self.cfg = cfg
        if cfg.dtype == "bf16":
            storage_dtype = torch.bfloat16
        else:
            if FP8_DTYPE is None:
                raise RuntimeError("this torch build has no torch.float8_e4m3fn dtype")
            storage_dtype = FP8_DTYPE
        self.storage_dtype = storage_dtype

        shape = (cfg.n_layers, cfg.n_pages, 2, cfg.page_size, cfg.num_kv_heads, cfg.head_dim)
        self.kv = torch.zeros(shape, dtype=storage_dtype, device=cfg.device)
        self.scale: Optional[torch.Tensor] = None
        if cfg.dtype == "fp8":
            self.scale = torch.ones(
                (cfg.n_layers, cfg.n_pages, 2, cfg.num_kv_heads), dtype=torch.float32, device=cfg.device
            )

        self.allocator = PageAllocator(cfg.n_pages)
        self.page_table = torch.full(
            (cfg.max_seqs, cfg.max_pages_per_seq), -1, dtype=torch.int32, device=cfg.device
        )
        # Host mirror of ``page_table``.  Page (de)allocation is *entirely*
        # host-side bookkeeping (``ensure_capacity``/``free_pages``/
        # ``reclaim_trailing_pages``/``free_slot`` are the only writers), so a
        # CPU copy can be kept exactly in sync for free -- and then
        # ``build_flashinfer_indices`` never has to read the device table at
        # all.  Reading ``int(self.page_table[slot, i])`` per page per
        # sequence per step instead is a D2H sync *per page*: at
        # B=32/ctx=2048/page_size=16 that is 32 x 129 = 4,128 device-to-host
        # round trips inside every decode step's plan phase (measured as the
        # dominant cost of a 91 ms B=32 step).
        self.page_table_host = torch.full(
            (cfg.max_seqs, cfg.max_pages_per_seq), -1, dtype=torch.int32, device="cpu"
        )
        # committed length per slot (tokens actually written / visible to attention)
        self.seq_len = torch.zeros(cfg.max_seqs, dtype=torch.int32, device=cfg.device)
        self._free_slots: List[int] = list(range(cfg.max_seqs))
        self._used_slots: set = set()

        # -- O(1) admission ------------------------------------------------ #
        # Exact page count per slot.  Invariant: ``_n_alloc[s]`` == the number
        # of leading ``>= 0`` entries of ``page_table_host[s]`` == the number
        # of leading ``>= 0`` entries of ``page_table[s]`` *after a flush*.
        # ``verify_page_accounting()`` asserts it; the CPU suite calls that.
        self._n_alloc: List[int] = [0] * cfg.max_seqs
        # Ranges of ``page_table_host`` whose device twin is stale.
        self._pt_dirty: List[Tuple[int, int, int]] = []
        self._pt_defer = 0
        # Ring of pinned staging pairs for the batched write.  Four deep so a
        # buffer is never rewritten while its own ``non_blocking`` H2D is
        # still in flight (the engine pipeline is one step deep; four is
        # belt-and-braces, and the event check below makes it exact).
        self._pt_ring: Optional[List[List]] = None
        self._pt_ring_i = 0
        # Ring of pinned staging for ``build_flashinfer_indices(staged=True)``.
        self._fi_ring: Optional[List[List]] = None
        self._fi_ring_i = 0

    # -- introspection -------------------------------------------------- #
    def nbytes(self) -> int:
        n = self.kv.numel() * self.kv.element_size()
        if self.scale is not None:
            n += self.scale.numel() * self.scale.element_size()
        return n

    @property
    def num_free_pages(self) -> int:
        return self.allocator.num_free

    def pages_needed(self, num_tokens: int) -> int:
        return (num_tokens + self.cfg.page_size - 1) // self.cfg.page_size

    def pages_allocated(self, slot: int) -> int:
        """How many pages ``slot`` owns.  **O(1)**, no tensor op, no sync.

        The unstaged form, ``int((self.page_table_host[slot] >= 0).sum())``,
        is correct and free of device syncs, but still a CPU reduction over
        ``max_pages_per_seq`` (256-512 int32) plus a ``.item()``, on every
        call.  ``Scheduler._ensure_capacity_with_preemption`` calls it once
        and then calls ``ensure_capacity``, which calls it again: two
        reductions per decode row, 512 per step at concurrency 256.  The
        counter below is maintained by every writer of the page table and
        checked by :meth:`verify_page_accounting`.
        """
        if not WS_S3:
            return int((self.page_table_host[slot] >= 0).sum())
        return self._n_alloc[slot]

    def verify_page_accounting(self, slot: Optional[int] = None) -> None:
        """Assert the O(1) counter still agrees with the host mirror.

        Test-only (and cheap enough for a CPU test to call after every step);
        nothing on the serving path calls it.  Raises ``AssertionError`` with
        the offending slot, which is the whole point: ``_n_alloc`` is a cache
        of a derived quantity and a missed update would silently mis-plan
        FlashInfer rather than crash.
        """
        slots = range(self.cfg.max_seqs) if slot is None else (slot,)
        for s in slots:
            row = self.page_table_host[s]
            n = int((row >= 0).sum())
            if n != self._n_alloc[s]:
                raise AssertionError(
                    f"page accounting desync at slot {s}: _n_alloc={self._n_alloc[s]}, "
                    f"host mirror has {n} mapped pages"
                )
            # the mapped pages must be a *prefix* -- every allocator here
            # appends, so a hole would mean a write went to the wrong index.
            if n and int((row[:n] < 0).sum()):
                raise AssertionError(f"slot {s}: mapped pages are not a prefix of the row")

    # -- the staged, batched page-table write ------------------------------ #
    @contextmanager
    def defer_page_table_writes(self) -> Iterator["PagedKVPool"]:
        """Hold every device page-table write until the scope exits.

        One admission pass over B decode rows becomes **one** device
        operation instead of up to B, and **zero** when no row crossed a page
        boundary.  Re-entrant.  The host mirror, the allocator and
        ``_n_alloc`` are updated eagerly inside the scope exactly as they are
        outside it, so ``pages_allocated``/``build_flashinfer_indices`` (which
        read only host state) are correct at every point *within* the scope --
        it is only the device twin that lags, and nothing reads that until
        ``append_kv``/``block_table_for``, both of which run after the flush.
        """
        self._pt_defer += 1
        try:
            yield self
        finally:
            self._pt_defer -= 1
            if self._pt_defer == 0:
                self.flush_page_table()

    def _ensure_pt_ring(self, n: int) -> None:
        ring = self._pt_ring
        if ring is not None and ring[0][0].numel() >= n:
            return
        cap = max(int(n) * 2, 512)
        dev = self.page_table.device
        cuda = dev.type == "cuda"
        if ring is not None:
            # a growth drops the old buffers; make sure no copy is still
            # reading them.
            for s in ring:
                if s[5]:
                    s[4].synchronize()
        new: List[List] = []
        for _ in range(4):
            try:
                idx_h = torch.zeros(cap, dtype=torch.int64, pin_memory=cuda)
                val_h = torch.zeros(cap, dtype=torch.int32, pin_memory=cuda)
            except (RuntimeError, NotImplementedError):  # pragma: no cover
                idx_h = torch.zeros(cap, dtype=torch.int64)
                val_h = torch.zeros(cap, dtype=torch.int32)
            new.append([
                idx_h,
                val_h,
                torch.zeros(cap, dtype=torch.int64, device=dev),
                torch.zeros(cap, dtype=torch.int32, device=dev),
                torch.cuda.Event() if cuda else None,
                False,
            ])
        self._pt_ring = new
        self._pt_ring_i = 0

    def flush_page_table(self) -> int:
        """Push every pending host-mirror change to the device.  Idempotent.

        Returns the number of page-table entries written (0 when there was
        nothing to do, which is the common decode step).  On CUDA this is
        exactly three device operations regardless of how many rows changed:
        two ``non_blocking`` H2D copies out of pinned staging and one
        ``index_copy_`` -- **no** ``cudaStreamSynchronize``, which is the
        whole point (see the module docstring).
        """
        dirty = self._pt_dirty
        if not dirty:
            return 0
        self._pt_dirty = []
        n = 0
        for _, a, b in dirty:
            n += b - a
        if n <= 0:
            return 0
        pt = self.page_table
        if pt.device.type != "cuda":
            for slot, a, b in dirty:
                pt[slot, a:b] = self.page_table_host[slot, a:b]
            return n
        capturing = getattr(torch.cuda, "is_current_stream_capturing", None)
        if capturing is not None and capturing():  # pragma: no cover
            raise RuntimeError(
                "flush_page_table() reached during CUDA graph capture: page allocation "
                "is host-side bookkeeping and must be flushed before capture/replay "
                "(see PagedKVPool.defer_page_table_writes)"
            )
        self._ensure_pt_ring(n)
        ring = self._pt_ring
        s = ring[self._pt_ring_i]
        self._pt_ring_i = (self._pt_ring_i + 1) % len(ring)
        if s[5]:
            # This staging pair's previous H2D may still be in flight; the
            # ring is four deep so in practice the event is long past.
            s[4].synchronize()
        idx_h, val_h, idx_d, val_d, ev = s[0], s[1], s[2], s[3], s[4]
        mpps = self.cfg.max_pages_per_seq
        flat: List[int] = []
        for slot, a, b in dirty:
            base = slot * mpps
            if b - a == 1:
                flat.append(base + a)
            else:
                flat.extend(range(base + a, base + b))
        idx_h[:n] = torch.tensor(flat, dtype=torch.int64)
        val_h[:n] = self.page_table_host.view(-1)[idx_h[:n]]
        idx_d[:n].copy_(idx_h[:n], non_blocking=True)
        val_d[:n].copy_(val_h[:n], non_blocking=True)
        pt.view(-1).index_copy_(0, idx_d[:n], val_d[:n])
        ev.record()
        s[5] = True
        return n

    def map_page(self, slot: int, index: int, page: int) -> None:
        """Point ``slot``'s ``index``-th page slot at ``page``.

        The escape hatch for callers that own a page from the allocator
        directly (``FusedQwenForCausalLM._init_scratch_slot``) rather than
        through ``ensure_capacity``; it exists so those callers cannot leave
        ``_n_alloc`` behind.

        ``index`` may **re-map** an already-mapped cell (``index <
        pages_allocated(slot)``), which is what ``_init_scratch_slot`` does
        when it is called a second time -- ``SpecDecoder.warmup`` calls it
        again after the model's own constructor did.  That has always leaked
        the previously-mapped page back to nobody; it is preserved here
        rather than fixed, because "the scratch slot's page id changes at
        warmup" is a behaviour some capture path may rely on.  What it may
        not do is leave a hole:
        appending past the end is rejected.
        """
        n = self._n_alloc[slot]
        if index > n:
            raise ValueError(
                f"map_page: slot {slot} has {n} pages, cannot map index {index} "
                "(pages must stay a contiguous prefix)"
            )
        self.page_table_host[slot, index] = page
        self._n_alloc[slot] = max(n, index + 1)
        self._pt_dirty.append((slot, index, index + 1))
        if not self._pt_defer:
            self.flush_page_table()

    # -- sequence-slot lifecycle ----------------------------------------- #
    def alloc_slot(self) -> int:
        """Allocate a page-table row for a new sequence. Returns the slot id
        (what device buffers elsewhere call ``slot_ids[B]``)."""
        if not self._free_slots:
            raise RuntimeError("no free sequence slots in the KV page table")
        slot = self._free_slots.pop()
        self._used_slots.add(slot)
        self.flush_page_table()  # this row's device twin is about to be reset
        self.page_table[slot].fill_(-1)
        self.page_table_host[slot].fill_(-1)
        self._n_alloc[slot] = 0
        self.seq_len[slot] = 0
        return slot

    def free_slot(self, slot: int) -> None:
        """Release a sequence's pages back to the free list *and* the slot."""
        self.free_pages(slot)
        self._used_slots.discard(slot)
        self._free_slots.append(slot)

    def free_pages(self, slot: int) -> int:
        """Return every page mapped to ``slot`` to the allocator; keep the slot.

        ``free_slot`` also hands the *sequence slot id* back to
        ``_free_slots``, which the runtime cannot use at all --
        ``fused_model._claim_all_kv_slots`` deliberately marks every row
        permanently live because slot ids are owned by
        ``scheduler.SlotManager`` (one id indexes the SSM pool *and* the KV
        page table).  This method is how a caller says "this slot is being
        reused for a different sequence, drop its pages";
        ``FusedQwenForCausalLM.reset_slot`` only zeroes the SSM/conv state.
        A caller that recycles a slot without it leaks the whole page set
        and eventually exhausts the pool.

        Returns the number of pages freed.  Host-side bookkeeping, never
        graph-safe -- same contract as ``ensure_capacity``.
        """
        self._check_slot(slot)
        # Only the mapped prefix is scanned (`_n_alloc`), not the whole
        # `max_pages_per_seq`-wide row -- this runs inside `_finish`, i.e.
        # inside the asynchronous commit, once per finishing request.
        n = self._n_alloc[slot]
        pages = [int(p) for p in self.page_table_host[slot, :n].tolist() if p >= 0]
        if pages:
            self.allocator.free(pages)
        self.flush_page_table()  # this row's device twin is about to be reset
        self.page_table[slot].fill_(-1)
        self.page_table_host[slot].fill_(-1)
        self._n_alloc[slot] = 0
        self.seq_len[slot] = 0
        return len(pages)

    def detach_pages(self, slot: int, keep_tokens: int) -> List[int]:
        """Release ``slot`` like :meth:`free_pages`, but hand the pages that
        hold its first ``keep_tokens`` tokens to the caller instead of the
        allocator (the prefix cache keeps them for the next turn of the same
        conversation). Pages past that prefix go back to the allocator.

        The kept pages are owned by the caller from here on: they are in no
        slot's page table and not on the free list, so nothing can write them
        until :meth:`attach_pages` maps them into a slot again (or the caller
        returns them with ``allocator.free``).
        """
        self._check_slot(slot)
        n = self._n_alloc[slot]
        pages = [int(p) for p in self.page_table_host[slot, :n].tolist() if p >= 0]
        keep = min(self.pages_needed(keep_tokens), len(pages))
        kept, dropped = pages[:keep], pages[keep:]
        if dropped:
            self.allocator.free(dropped)
        self.flush_page_table()
        self.page_table[slot].fill_(-1)
        self.page_table_host[slot].fill_(-1)
        self._n_alloc[slot] = 0
        self.seq_len[slot] = 0
        return kept

    def attach_pages(self, slot: int, pages: List[int], length: int) -> None:
        """Map ``pages`` (from :meth:`detach_pages`) as ``slot``'s leading
        pages and mark its first ``length`` tokens committed. ``slot`` must
        hold no pages. Positions past ``length`` inside the last page are stale
        and get overwritten by the next prefill, exactly as after ``truncate``.
        """
        self._check_slot(slot)
        if self._n_alloc[slot]:
            raise ValueError(f"attach_pages: slot {slot} already holds {self._n_alloc[slot]} pages")
        if len(pages) > self.cfg.max_pages_per_seq:
            raise ValueError(f"attach_pages: {len(pages)} pages > max_pages_per_seq")
        if self.pages_needed(length) > len(pages):
            raise ValueError(f"attach_pages: {len(pages)} pages cannot hold {length} tokens")
        if pages:
            self.page_table_host[slot, : len(pages)] = torch.tensor(pages, dtype=torch.int32)
            self._n_alloc[slot] = len(pages)
            self._pt_dirty.append((slot, 0, len(pages)))
            if not self._pt_defer:
                self.flush_page_table()
        self.seq_len[slot] = length

    def _check_slot(self, slot: int) -> None:
        if slot not in self._used_slots:
            raise ValueError(f"slot {slot} is not an allocated sequence slot")

    # -- page (de)allocation: host-side, NOT CUDA-graph-safe -------------- #
    def ensure_capacity(self, slot: int, num_tokens: int) -> None:
        """Grow ``slot``'s page table so it can hold ``num_tokens`` tokens.

        Call this during scheduling/admission, *before* the decode step
        (graph replay) or prefill chunk that will write those tokens --
        mirrors FlashInfer's ``plan()``-outside-graph / ``run()``-inside-graph
        split used in ``flashinfer_attn.py``. Never shrinks or frees pages
        (use ``truncate`` for rollback, ``free_slot`` to release everything).
        """
        self._check_slot(slot)
        need = self.pages_needed(num_tokens)
        have = self.pages_allocated(slot)
        if need <= have:
            return
        if need > self.cfg.max_pages_per_seq:
            raise RuntimeError(
                f"slot {slot} needs {need} pages (> max_pages_per_seq={self.cfg.max_pages_per_seq})"
            )
        new_pages = self.allocator.alloc(need - have)
        # The device write is **staged**, not issued here.  A direct
        #     self.page_table[slot, have:need] = new_pages_t.to(device)
        # is a pageable H2D, which PyTorch implements as cudaMemcpyAsync +
        # cudaStreamSynchronize.  Once per row that crosses a page boundary,
        # inside an admission pass that runs while the *previous* step's 44 ms
        # of kernels are queued, that is a full-step host block (admission
        # measured going from 26.3 to 48.6 ms once the pipeline became
        # asynchronous).  Instead the host mirror and the O(1) counter are
        # updated here and the changed range is recorded; `flush_page_table`
        # writes every range of the whole step in one `index_copy_` fed by
        # pinned, genuinely non-blocking copies.  A step where nothing crosses
        # a page boundary does no device work at all -- it does not even
        # reach this line.
        #
        # The host mirror is written with one vectorized slice assignment,
        # not a Python loop of scalar writes (`for i, p in
        # enumerate(new_pages): self.page_table[slot, have + i] = p`). On a
        # device table each loop iteration is a separate host-to-device copy
        # with full CUDA-API dispatch overhead (~10-50us), so a single B=512,
        # ctx=8192, page_size=16 sequence (512 pages) already costs ~10-25ms,
        # and a benchmark sweep calling this per-sequence, per-cell turns
        # that into minutes of host-loop-bound wall time.
        new_pages_t = torch.tensor(new_pages, dtype=self.page_table.dtype, device="cpu")
        self.page_table_host[slot, have:need] = new_pages_t
        self._n_alloc[slot] = need
        if not WS_S3:
            self.page_table[slot, have:need] = new_pages_t.to(self.page_table.device)
            return
        self._pt_dirty.append((slot, have, need))
        if not self._pt_defer:
            self.flush_page_table()

    def calibrate_page_scale(
        self, layer: int, page: int, k_or_v: int, sample: torch.Tensor, headroom: float = 1.0 / 448.0
    ) -> None:
        """Set the fp8 scale for one (layer, page, K-or-V) from a sample.

        ``sample``: ``[..., num_kv_heads, head_dim]`` in a higher-precision
        dtype (e.g. the bf16 tokens about to fill the page). Scale is
        ``amax / FP8_MAX`` per head, floored so an all-zero sample doesn't
        produce a zero (division-by-zero) scale. ``headroom`` reserves a
        little dynamic range for tokens written later into the same page
        whose magnitude wasn't seen in ``sample``: the stored code at
        exactly ``amax`` lands at ``(1 - headroom)`` of ``FP8_MAX``, not at
        ``FP8_MAX`` itself, leaving that fraction of headroom above it.

        The expression for "``headroom`` fraction of range reserved above
        ``amax``" is ``amax / (FP8_MAX * (1 - headroom))``. Note that
        ``amax / (FP8_MAX * headroom)`` would be wrong in a way no round-trip
        test notices: with a small ``headroom`` it makes the scale ~447x too
        large, so stored codes sit around magnitude ~1 instead of using
        e4m3's [-448, 448] range (self-consistent, just wasted precision).
        """
        if self.scale is None:
            raise RuntimeError("calibrate_page_scale is only valid for fp8 pools")
        # reduce every dim except the head axis (second-to-last) -> [num_kv_heads]
        head_dim_axis = sample.dim() - 2
        reduce_dims = tuple(d for d in range(sample.dim()) if d != head_dim_axis)
        amax = sample.float().abs().amax(dim=reduce_dims)
        amax = torch.clamp(amax, min=1e-6)
        self.scale[layer, page, k_or_v] = amax / (FP8_MAX * (1.0 - headroom))

    def calibrate_uniform_scale(
        self, layer: int, k_or_v: int, sample: torch.Tensor, headroom: float = 1.0 / 448.0
    ) -> float:
        """Calibrate ONE Python ``float`` scale for an entire layer's K (or
        V), shared by *every* page and *every* head, and return it.

        ``calibrate_page_scale``'s ``[n_layers, n_pages, 2, num_kv_heads]``
        per-page/per-head resolution is real and is what the torch-fallback
        path (``gather_dense``) reads. But FlashInfer's fp8 paged attention
        kernels (``BatchDecodeWithPagedKVCacheWrapper``/
        ``BatchPrefillWithPagedKVCacheWrapper``, both via this package's
        wrapper classes' ``run(..., k_scale=..., v_scale=...)``) accept only
        a single Python ``float`` per call (flashinfer 0.6.16 source:
        ``k_scale: Optional[float] = None``, not a tensor, not per-head).
        A per-page or
        per-head scale is therefore *not representable* through FlashInfer's
        kernel in one call: whatever scale you pass applies uniformly to
        every page and every head that call reads. Any caller that will read
        this layer's K/V through ``FlashInferDecodeAttention``/
        ``FlashInferPrefillAttention`` must calibrate with this method
        instead of (or as well as) ``calibrate_page_scale`` -- or otherwise
        guarantee every page/head touched by a single kernel call shares the
        same scale -- and pass the returned float back in as
        ``run(k_scale=..., v_scale=...)``. FlashInfer's ``run()`` defaults
        unset ``k_scale``/``v_scale`` to ``1.0``, which is silently wrong
        whenever the pool's calibrated scale isn't 1.0
        (``test_decode_matches_torch_fallback_fp8_kv`` guards this).
        """
        if self.scale is None:
            raise RuntimeError("calibrate_uniform_scale is only valid for fp8 pools")
        amax = torch.clamp(sample.float().abs().amax(), min=1e-6)
        scale = float(amax / (FP8_MAX * (1.0 - headroom)))
        self.scale[layer, :, k_or_v, :] = scale
        return scale

    # -- kv write: pure scatter, CUDA-graph-safe -------------------------- #
    def append_kv(
        self,
        layer: int,
        slot_ids: torch.Tensor,
        positions: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """Write ``T`` tokens' K/V into layer ``layer``'s pages.

        ``slot_ids``: ``[T]`` int -- which sequence slot each token belongs
        to (decode: ``T == B``, one token per live sequence; prefill: ``T``
        is a whole packed varlen chunk, many tokens per slot).
        ``positions``: ``[T]`` int -- absolute position within that
        sequence.  ``k``, ``v``: ``[T, num_kv_heads, head_dim]``.

        Destination pages must already exist (``ensure_capacity`` first) --
        this method never allocates, so it contains no data-dependent control
        flow and is safe to call from inside a captured CUDA graph with
        ``slot_ids``/``positions``/``k``/``v`` as fixed-shape device buffers.
        """
        if not (slot_ids.shape[0] == positions.shape[0] == k.shape[0] == v.shape[0]):
            raise ValueError("slot_ids, positions, k, v must share their leading (token) dimension")
        if self._pt_dirty:
            # Safety net: this method reads the *device* page table, so
            # a staged allocation has to land first.  Every real caller runs
            # after the scheduler's own flush, so this is a no-op truthiness
            # test on the hot path -- and it raises rather than silently
            # capturing an `index_copy_` if a graph capture ever got here with
            # work outstanding (see `flush_page_table`).
            self.flush_page_table()
        page_size = self.cfg.page_size
        slot_ids_l = slot_ids.long()
        positions_l = positions.long()
        page_idx_in_seq = positions_l // page_size
        offset = positions_l % page_size

        pages = self.page_table[slot_ids_l, page_idx_in_seq]
        if bool((pages < 0).any()):
            raise RuntimeError(
                "append_kv: destination page not allocated for one or more tokens -- "
                "call ensure_capacity(slot, num_tokens) first"
            )
        pages = pages.long()

        if self.cfg.dtype == "fp8":
            k_scale = self.scale[layer, pages, 0].unsqueeze(-1)  # [T, H, 1]
            v_scale = self.scale[layer, pages, 1].unsqueeze(-1)
            k_store = (k.float() / k_scale).clamp(-FP8_MAX, FP8_MAX).to(self.storage_dtype)
            v_store = (v.float() / v_scale).clamp(-FP8_MAX, FP8_MAX).to(self.storage_dtype)
        else:
            k_store = k.to(self.storage_dtype)
            v_store = v.to(self.storage_dtype)

        self.kv[layer, pages, 0, offset] = k_store
        self.kv[layer, pages, 1, offset] = v_store

        # seq_len is a *max* over layers/calls (so a multi-layer forward
        # converges to the right length regardless of layer call order) AND,
        # within a single call, a max over every token routed to the same
        # slot -- a prefill chunk writes many tokens (possibly all of one
        # sequence's slot) in a single append_kv call, so slot_ids_l can
        # contain duplicates. Plain fancy-index assignment
        # (`self.seq_len[slot_ids_l] = ...`) does NOT perform that reduction
        # when the destination index has duplicates: PyTorch documents the
        # result of writing the same location more than once via basic
        # advanced-indexing assignment as unspecified on CUDA (the scatter is
        # parallel, no ordering guarantee) -- and empirically, on H200 it can
        # keep the *first* token's value, not the max, leaving a multi-token
        # prefill sequence's seq_len at 1 so FlashInfer masks its KV down to
        # a single valid token (`test_prefill_matches_torch_fallback_bf16`
        # guards this). CPU's *deterministic* "last write wins" semantics
        # happen to give the right answer for our specific access pattern
        # (positions are written in increasing order within one call, so the
        # last duplicate is also the true max), so a CPU-only test cannot
        # detect the hazard. `scatter_reduce_(..., reduce="amax")` is
        # the correct, write-order-independent way to fold duplicate
        # destination indices, on both CPU and CUDA.
        new_len = (positions_l + 1).to(self.seq_len.dtype)
        self.seq_len.scatter_reduce_(0, slot_ids_l, new_len, reduce="amax", include_self=True)

    # -- rollback: O(1) pointer move, CUDA-graph-safe ---------------------- #
    def truncate(self, slot: int, new_len: int) -> None:
        """Roll back ``slot`` to ``new_len`` tokens.

        The 16 attention layers roll back trivially (move the KV write
        pointer; free pages). This is the pointer-move half -- no
        data is touched, no pages are freed, so a later token written past
        ``new_len`` (e.g. a re-accepted draft on the next verify round) simply
        overwrites stale bytes. Call ``reclaim_trailing_pages`` separately if
        you want the now-unused pages back on the free list (that part *is*
        host-side bookkeeping, not graph-safe, so it is not automatic here).
        """
        self._check_slot(slot)
        if new_len < 0:
            raise ValueError("new_len must be >= 0")
        if new_len > self.pages_allocated(slot) * self.cfg.page_size:
            raise ValueError("new_len exceeds the pages currently allocated to this slot")
        self.seq_len[slot] = new_len

    def reclaim_trailing_pages(self, slot: int) -> int:
        """Free pages strictly beyond ``seq_len[slot]`` back to the allocator.

        Optional memory-reclaim step after a ``truncate`` (e.g. once a
        speculative-decode round has settled and the rejected tail's pages
        are known to be dead weight). Returns the number of pages freed.
        """
        self._check_slot(slot)
        keep = self.pages_needed(int(self.seq_len[slot]))
        have = self.pages_allocated(slot)
        if keep >= have:
            return 0
        freed = [int(p) for p in self.page_table_host[slot, keep:have].tolist() if p >= 0]
        self.allocator.free(freed)
        self.flush_page_table()  # this range's device twin is about to be reset
        self.page_table[slot, keep:have] = -1
        self.page_table_host[slot, keep:have] = -1
        self._n_alloc[slot] = keep
        return len(freed)

    # -- reads for the torch-fallback attention path ----------------------- #
    def gather_dense(self, layer: int, slot: int, length: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dequantized, contiguous ``(k, v)`` of shape ``[length, H, D]`` for
        one sequence -- used by the SDPA fallback path, never in the hot
        FlashInfer path (which reads pages directly)."""
        self._check_slot(slot)
        n = int(self.seq_len[slot]) if length is None else length
        if n == 0:
            return (
                torch.empty(0, self.cfg.num_kv_heads, self.cfg.head_dim, dtype=torch.bfloat16, device=self.kv.device),
                torch.empty(0, self.cfg.num_kv_heads, self.cfg.head_dim, dtype=torch.bfloat16, device=self.kv.device),
            )
        n_pages = self.pages_needed(n)
        if self._pt_dirty:
            self.flush_page_table()
        page_ids = self.page_table[slot, :n_pages].long()
        if bool((page_ids < 0).any()):
            raise RuntimeError(f"slot {slot} has fewer than {n_pages} pages allocated for length {n}")
        k_pages = self.kv[layer, page_ids, 0]  # [n_pages, page_size, H, D]
        v_pages = self.kv[layer, page_ids, 1]
        if self.cfg.dtype == "fp8":
            k_scale = self.scale[layer, page_ids, 0].unsqueeze(1).unsqueeze(-1)  # [n_pages,1,H,1]
            v_scale = self.scale[layer, page_ids, 1].unsqueeze(1).unsqueeze(-1)
            k_pages = k_pages.float() * k_scale
            v_pages = v_pages.float() * v_scale
        k = k_pages.reshape(-1, self.cfg.num_kv_heads, self.cfg.head_dim)[:n].to(torch.bfloat16)
        v = v_pages.reshape(-1, self.cfg.num_kv_heads, self.cfg.head_dim)[:n].to(torch.bfloat16)
        return k, v

    # -- FlashAttention-3 paged-decode input ------------------------------- #
    def block_table_for(self, slot_ids: torch.Tensor) -> torch.Tensor:
        """``[len(slot_ids), max_pages_per_seq]`` int32 page-index table for
        the given slots, clamped to ``>= 0`` (unmapped ``-1`` rows become
        ``0``). This *is* FlashAttention-3's ``block_table`` argument
        (``flash_attn_varlen_func(..., block_table=..., seqused_k=...)``, see
        ``flashinfer_attn.fa3_decode_with_kvcache``) -- the clamp exists only
        so an unmapped page never produces an out-of-range address; the
        kernel never reads past ``seqused_k`` so the clamped value itself is
        never actually used for real attention math.
        """
        if self._pt_dirty:
            self.flush_page_table()
        table = self.page_table[slot_ids.long()]
        return table.clamp(min=0).to(torch.int32)

    # -- FlashInfer planning inputs ----------------------------------------- #
    def build_flashinfer_indices(
        self,
        slot_ids: Sequence[int],
        seq_lens: Optional[Sequence[int]] = None,
        *,
        staged: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the ``(kv_indptr, kv_indices, kv_last_page_len, seq_lens)``
        quadruple FlashInfer's ``plan()`` wants, for the given batch of slots
        (in order), as **device** int32 tensors.

        Cost contract.  A naive Python loop doing ``int(self.seq_len[slot])``
        per sequence and ``int(self.page_table[slot, i])`` per *page* costs
        one device-to-host sync each: a B=32, ctx=2048, page_size=16 decode
        step would do 32 x (1 + 129) = 4,160 D2H round trips *before every
        graph replay*, at roughly 15 us apiece, which dominated a measured
        91 ms B=32 step.  This version does:

        * page ids: **zero** device reads -- they come from
          ``page_table_host``, which is exact by construction (only host-side
          bookkeeping ever allocates or frees a page).
        * lengths: **one** vectorised D2H (``self.seq_len[idx].cpu()``), or
          zero if the caller passes ``seq_lens`` (the scheduler always knows
          them; ``append_kv`` is the only writer that advances the device
          copy without the host seeing it).
        * three small H2D copies for the returned tensors.

        i.e. O(1) syncs per step instead of O(B x pages_per_seq).

        ``staged=True`` closes the last hole in that
        contract.  "Three small H2D copies" above is only true if they are
        *asynchronous*, and ``t.to(device, non_blocking=True)`` out of a
        freshly-allocated (therefore **pageable**) CPU tensor is not: PyTorch
        silently degrades it to ``cudaMemcpyAsync`` + ``cudaStreamSynchronize``.
        Under ``--async-scheduling`` that is a host block on the previous
        step's whole kernel time.  With ``staged=True`` the arrays are built
        into a pinned ring buffer and the return is
        ``(indptr_host, indices_device, last_page_host, seq_lens_host)`` --
        FlashInfer's ``plan()`` wants the first and third **on the host**
        anyway (it does ``indptr.to("cpu")`` internally), so handing it host
        tensors removes a full H2D-then-D2H round trip per step as well.
        The returned host tensors are views into the ring and are only valid
        until the fourth subsequent staged call.
        """
        page_size = self.cfg.page_size
        idx = torch.as_tensor(list(slot_ids), dtype=torch.long, device="cpu")

        if seq_lens is None:
            lens = self.seq_len[idx.to(self.seq_len.device)].to("cpu", torch.int64)
        else:
            lens = torch.as_tensor(list(seq_lens), dtype=torch.int64, device="cpu")

        n_pages = (lens + (page_size - 1)) // page_size            # [B]
        indptr = torch.zeros(idx.numel() + 1, dtype=torch.int32)
        indptr[1:] = n_pages.cumsum(0).to(torch.int32)

        max_np = int(n_pages.max()) if n_pages.numel() else 0
        if max_np > 0:
            rows = self.page_table_host[idx, :max_np]              # [B, max_np]
            keep = torch.arange(max_np).unsqueeze(0) < n_pages.unsqueeze(1)
            # row-major masked select == the per-sequence page runs
            # concatenated in order, which is exactly FlashInfer's kv_indices.
            indices = rows[keep].to(torch.int32)
        else:
            indices = torch.zeros(0, dtype=torch.int32)

        rem = lens - (n_pages - 1).clamp(min=0) * page_size
        last = torch.where(
            n_pages > 0,
            torch.where(rem > 0, rem, torch.full_like(rem, page_size)),
            torch.zeros_like(rem),
        ).to(torch.int32)

        device = self.kv.device
        if indices.numel() == 0:
            indices = torch.zeros(1, dtype=torch.int32)
        if last.numel() == 0:
            last = torch.zeros(1, dtype=torch.int32)
        lens32 = lens.to(torch.int32)
        if staged and WS_S3 and device.type == "cuda":
            return self._stage_flashinfer_indices(indptr, indices, last, lens32)
        return (
            indptr.to(device, non_blocking=True),
            indices.to(device, non_blocking=True),
            last.to(device, non_blocking=True),
            lens32.to(device, non_blocking=True),
        )

    # -- the pinned staging ring behind ``staged=True`` --------------------- #
    def _stage_flashinfer_indices(
        self,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        last: torch.Tensor,
        lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        nb, ni = int(indptr.numel()), int(indices.numel())
        ring = self._fi_ring
        if ring is None or ring[0][0].numel() < nb or ring[0][3].numel() < ni:
            self._grow_fi_ring(nb, ni)
            ring = self._fi_ring
        s = ring[self._fi_ring_i]
        self._fi_ring_i = (self._fi_ring_i + 1) % len(ring)
        if s[6]:
            s[5].synchronize()
        indptr_h, last_h, lens_h, idx_h, idx_d, ev = s[0], s[1], s[2], s[3], s[4], s[5]
        nl = int(last.numel())
        indptr_h[:nb].copy_(indptr)
        last_h[:nl].copy_(last)
        lens_h[: lens.numel()].copy_(lens)
        idx_h[:ni].copy_(indices)
        out_idx = idx_d[:ni]
        out_idx.copy_(idx_h[:ni], non_blocking=True)
        ev.record()
        s[6] = True
        return indptr_h[:nb], out_idx, last_h[:nl], lens_h[: lens.numel()]

    def _grow_fi_ring(self, nb: int, ni: int) -> None:
        old = self._fi_ring
        if old is not None:
            for s in old:
                if s[6]:
                    s[5].synchronize()
        cap_b = max(nb * 2, 64)
        cap_i = max(ni * 2, 4096)
        dev = self.kv.device
        new: List[List] = []
        for _ in range(4):
            try:
                bufs = [
                    torch.zeros(cap_b, dtype=torch.int32, pin_memory=True),
                    torch.zeros(cap_b, dtype=torch.int32, pin_memory=True),
                    torch.zeros(cap_b, dtype=torch.int32, pin_memory=True),
                    torch.zeros(cap_i, dtype=torch.int32, pin_memory=True),
                ]
            except (RuntimeError, NotImplementedError):  # pragma: no cover
                bufs = [
                    torch.zeros(cap_b, dtype=torch.int32),
                    torch.zeros(cap_b, dtype=torch.int32),
                    torch.zeros(cap_b, dtype=torch.int32),
                    torch.zeros(cap_i, dtype=torch.int32),
                ]
            new.append(bufs + [
                torch.zeros(cap_i, dtype=torch.int32, device=dev),
                torch.cuda.Event(),
                False,
            ])
        self._fi_ring = new
        self._fi_ring_i = 0


__all__ = ["KVPoolConfig", "PageAllocator", "PagedKVPool", "FP8_DTYPE", "FP8_MAX"]
