"""Turn to turn prefix cache for agent conversations on the hybrid model.

An agent loop sends the same conversation again every turn with the previous
assistant message and the tool results appended. Without a cache every turn
re-prefills the whole context (10k tok/s on an h200: 5 s per turn at a 50k
context). A paged KV cache alone cannot fix that for this model, because 48 of
its 64 layers are gated deltanet: their recurrent state after token ``n``
cannot be cut out of the state after token ``m > n``. What can be kept is the
state at one exact position.

So each request that finishes its prompt snapshots, stream ordered right after
the forward that consumed its last prompt token:

* the gdn recurrent state and the causal conv state of its slot,
* the mtp head's ``h_prev`` carry (speculative decoding),

and when the request finishes, the kv pages holding its prompt (all 16
attention layers plus the mtp layer, one page set) are moved out of the slot
into the cache entry instead of back to the allocator. A later request whose
prompt *starts with* that whole prompt takes the entry: the pages are mapped
into its new slot, the snapshot is copied into the slot's state, and prefill
starts at the cached length. The entry is consumed (moved, not copied): an
agent conversation is a line, turn ``t + 1`` extends turn ``t``, and turn
``t + 1`` then leaves its own entry for turn ``t + 2``.

Correctness does not depend on the chat template reproducing anything: the
match is an exact token prefix test, and a request only ever resumes from a
state that was computed from exactly those tokens.

Memory: one snapshot is one slot of ssm + conv state (75 MiB on qwen3.8-27b),
held in a preallocated pool of ``n_entries``. The kv pages an entry holds are
taken from the shared pool and are the first thing given back under pressure:
admission and decode growth evict least recently used entries before they
ever preempt a running request.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


@dataclass
class PrefixEntry:
    tokens: List[int]
    snap: int
    pages: List[int] = field(default_factory=list)
    last_used: float = field(default_factory=time.monotonic)

    @property
    def length(self) -> int:
        return len(self.tokens)


@dataclass
class PrefixCacheStats:
    lookups: int = 0
    hits: int = 0
    hit_tokens: int = 0
    prompt_tokens: int = 0
    evictions: int = 0
    entries: int = 0
    pages_held: int = 0


class PrefixCache:
    """Snapshot pool plus the committed entries. Host side bookkeeping only;
    every device operation is a ``copy_`` issued on the current stream."""

    def __init__(self, model, spec=None, *, n_entries: int, min_tokens: int = 512):
        if n_entries <= 0:
            raise ValueError("n_entries must be positive")
        self.model = model
        self.spec = spec
        self.min_tokens = max(1, int(min_tokens))
        self.n_entries = int(n_entries)
        sp, cp = model.state_pool, model.conv_pool
        self.ssm = torch.empty((n_entries,) + tuple(sp.shape[1:]), dtype=sp.dtype, device=sp.device)
        self.conv = torch.empty((n_entries,) + tuple(cp.shape[1:]), dtype=cp.dtype, device=cp.device)
        self.hprev: Optional[torch.Tensor] = None
        if spec is not None and getattr(spec, "h_prev", None) is not None:
            hp = spec.h_prev
            self.hprev = torch.empty((n_entries,) + tuple(hp.shape[1:]), dtype=hp.dtype, device=hp.device)
        self._free_snaps: List[int] = list(range(n_entries))
        #: request id -> (snapshot index, prompt length it was taken at)
        self._pending: Dict[str, int] = {}
        self._entries: List[PrefixEntry] = []
        self.stats = PrefixCacheStats()

    # -- snapshots ------------------------------------------------------------ #
    def _alloc_snap(self) -> Optional[int]:
        if not self._free_snaps:
            # every snapshot is in use: drop the least recently used committed
            # entry (pending snapshots belong to live requests and stay)
            if not self._evict_lru():
                return None
        return self._free_snaps.pop()

    def snapshot(self, request_id: str, slot: int, n_prompt: int) -> None:
        """Called right after the forward that consumed the request's last
        prompt token was launched, on the same stream, before the slot's
        state is advanced again."""
        if n_prompt < self.min_tokens or request_id in self._pending:
            return
        idx = self._alloc_snap()
        if idx is None:
            return
        self.ssm[idx].copy_(self.model.state_pool[slot], non_blocking=True)
        self.conv[idx].copy_(self.model.conv_pool[slot], non_blocking=True)
        if self.hprev is not None:
            self.hprev[idx].copy_(self.spec.h_prev[slot], non_blocking=True)
        self._pending[request_id] = idx

    def drop_pending(self, request_id: str) -> None:
        idx = self._pending.pop(request_id, None)
        if idx is not None:
            self._free_snaps.append(idx)

    def commit(self, request_id: str, prompt: List[int], slot: int, kv_pool) -> bool:
        """On finish: turn the pending snapshot into an entry that owns the
        slot's prompt pages. Returns True when it took the pages (the caller
        must then not free them); False means the caller frees as usual."""
        idx = self._pending.pop(request_id, None)
        if idx is None:
            return False
        tokens = list(prompt)
        # an identical prompt already cached (a retried request): keep the older one
        for e in self._entries:
            if e.tokens == tokens:
                self._free_snaps.append(idx)
                return False
        pages = kv_pool.detach_pages(slot, len(tokens))
        self._entries.append(PrefixEntry(tokens=tokens, snap=idx, pages=pages))
        return True

    # -- lookup / restore ----------------------------------------------------- #
    def lookup(self, prompt: List[int], *, count: bool = True) -> Optional[PrefixEntry]:
        """Longest entry that is a strict prefix of ``prompt`` (at least one
        prompt token must still run to produce the first logits)."""
        if count:
            self.stats.lookups += 1
            self.stats.prompt_tokens += len(prompt)
        best: Optional[PrefixEntry] = None
        n = len(prompt)
        for e in self._entries:
            L = e.length
            if L >= n or L < self.min_tokens or (best is not None and L <= best.length):
                continue
            # cheap rejections first: both ends of the cached span
            if prompt[L - 1] != e.tokens[-1] or prompt[0] != e.tokens[0]:
                continue
            if prompt[:L] == e.tokens:
                best = e
        return best

    def restore(self, entry: PrefixEntry, slot: int, kv_pool) -> int:
        """Moves ``entry`` into ``slot`` (state + pages). Returns the number
        of prompt tokens the caller may skip."""
        self._entries.remove(entry)
        self.model.state_pool[slot].copy_(self.ssm[entry.snap], non_blocking=True)
        self.model.conv_pool[slot].copy_(self.conv[entry.snap], non_blocking=True)
        if self.hprev is not None:
            self.spec.h_prev[slot].copy_(self.hprev[entry.snap], non_blocking=True)
        kv_pool.attach_pages(slot, entry.pages, entry.length)
        self._free_snaps.append(entry.snap)
        self.stats.hits += 1
        self.stats.hit_tokens += entry.length
        return entry.length

    # -- memory pressure ------------------------------------------------------ #
    def _evict_lru(self, keep: Optional[PrefixEntry] = None) -> bool:
        victims = [e for e in self._entries if e is not keep]
        if not victims:
            return False
        e = min(victims, key=lambda x: x.last_used)
        self._entries.remove(e)
        if e.pages:
            self.model.kv_pool.allocator.free(e.pages)
        self._free_snaps.append(e.snap)
        self.stats.evictions += 1
        return True

    def reclaim_pages(self, need_free: int, keep: Optional[PrefixEntry] = None) -> None:
        """Evict entries until the kv pool has ``need_free`` free pages or
        nothing (other than ``keep``) is left to evict."""
        pool = self.model.kv_pool
        while pool.num_free_pages < need_free and self._evict_lru(keep):
            pass

    def pages_held(self) -> int:
        return sum(len(e.pages) for e in self._entries)

    def snapshot_stats(self) -> PrefixCacheStats:
        self.stats.entries = len(self._entries)
        self.stats.pages_held = self.pages_held()
        return self.stats

    def clear(self) -> None:
        while self._evict_lru():
            pass
