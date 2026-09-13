"""Launch-reduction kernels.

Each module here is one fused Triton kernel that replaces a small chain of
eager torch ops that a decode profile at B=128 showed as many-launches /
little-work: several kernel *launches* at the ~1-2 us dispatch floor, not
real compute. The norms had the same shape of problem and were fused the
same way; these are the next biggest groups in the ``elementwise/other`` /
``copy`` buckets after that.

Convention (matches ``fused_model.py``'s existing Triton norms):

* every kernel here is optional and self-contained -- importable on a
  CPU/no-triton host (``HAS_TRITON = False``, the ``triton_*`` entry point
  raises if called), never a hard dependency.
* the eager torch reference each one replaces stays the default
  (``RuntimeConfig.fused_ops_backend = "torch"``); the fused kernel is
  opt-in via ``--fused-ops-backend triton`` until its parity test has run
  on a GPU (``tests/test_runtime.py::TestFusedOps``).
* one launch, one pass, fp32 accumulate where the eager op's numerics
  contract needs it. These are not new numerics, just the
  same arithmetic issued as one kernel instead of three to five.
"""
