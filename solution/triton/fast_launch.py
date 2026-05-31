"""Fast Triton kernel launcher — bypasses JIT dispatch overhead after first call.

On first call: normal JIT path (handles compilation + autotuning).
On subsequent calls: direct CudaLauncher invocation, skipping:
  - specialize_impl × N args (Python→C++ per arg)
  - compute_cache_key string building
  - kernel_cache dict lookup

Cache key: (id(fn), T) — T uniquely determines all constexpr kwargs and
int-arg specializations for every kernel in this codebase.  The caller
passes T via the _cache_key keyword-only parameter.
"""

import os

from triton.runtime.jit import JITFunction

# PDL can be disabled for profiling (sequential kernel execution = cleaner traces)
_pdl_enabled = os.environ.get('DISABLE_PDL', '') == ''


class FastLauncher:
    """Cache compiled Triton kernels for direct CudaLauncher invocation."""

    __slots__ = ('_cache', '_stream')

    def __init__(self):
        self._cache = {}
        self._stream = None

    def __call__(self, fn, grid, *args, _cache_key=0, **kwargs):
        """Launch a Triton kernel, bypassing JIT on subsequent calls.

        _cache_key: pass T (sequence length) — uniquely identifies the
        compiled binary for a given fn, since T determines all constexpr
        values and int-arg specialization properties.
        """
        key = (id(fn), _cache_key)

        entry = self._cache.get(key)
        if entry is not None:
            # ── HOT PATH: direct launch ──
            launcher, function, packed_metadata, kwarg_values = entry

            # Compute grid
            if callable(grid):
                # Rare: lambda grid. Fall back to dict-based meta.
                jit_fn = fn
                while not isinstance(jit_fn, JITFunction):
                    jit_fn = jit_fn.fn
                meta = dict(zip(jit_fn.arg_names[:len(args)], args))
                meta.update(kwargs)
                if hasattr(fn, 'best_config'):
                    meta.update(fn.best_config.kwargs)
                grid_val = grid(meta)
            else:
                grid_val = grid

            n = len(grid_val) if isinstance(grid_val, (tuple, list)) else 1
            gx = grid_val[0] if n >= 1 else 1
            gy = grid_val[1] if n >= 2 else 1
            gz = grid_val[2] if n >= 3 else 1

            # Get stream (cached after first use)
            if self._stream is None:
                from triton.runtime.driver import driver
                device = driver.active.get_current_device()
                self._stream = driver.active.get_current_stream(device)

            # Launch: positional args + pre-built constexpr values
            launcher(gx, gy, gz, self._stream, function, packed_metadata,
                     None, None, None, *args, *kwarg_values)
            return

        # ── COLD PATH: first call, normal JIT ──
        ck = fn[grid](*args, **kwargs)
        if ck is None:
            return

        # Get the underlying JITFunction for arg_names
        jit_fn = fn
        while not isinstance(jit_fn, JITFunction):
            jit_fn = jit_fn.fn

        # For autotuned kernels, capture the best config's kernel kwargs
        config_kwargs = {}
        if hasattr(fn, 'best_config'):
            config_kwargs = dict(fn.best_config.kwargs)

        # Enable PDL on all Triton kernels
        ck._run.launch_pdl = _pdl_enabled

        # Pre-build the constexpr kwargs values in signature order.
        # On hot path we just append these after positional args.
        n_pos = len(args)
        kwarg_values = []
        full_kwargs = {**kwargs, **config_kwargs}
        for name in jit_fn.arg_names[n_pos:]:
            kwarg_values.append(full_kwargs[name])
        kwarg_values = tuple(kwarg_values)

        self._cache[key] = (
            ck._run,              # CudaLauncher instance
            ck.function,          # CUDA function handle
            ck.packed_metadata,   # packed kernel metadata
            kwarg_values,         # pre-built constexpr values in order
        )


# Singleton instance
launch = FastLauncher()
