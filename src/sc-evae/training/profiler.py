"""
profiler.py
-----------
Per-section wall-clock profiler for sc_evae training loops.

When enabled, accumulates timings for a configurable number of steps after a
short CUDA-kernel warmup, then dumps an aggregated JSON report and prints a
short summary to the log. Designed to be cheap when disabled (no-op context
manager) and accurate when enabled (CUDA syncs around each measured section).

Use from ``scripts/train.py``::

    from sc_evae.training.profiler import Profiler

    profiler = Profiler(
        enabled=cfg.profile,
        n_steps=cfg.profile_n_steps,
        warmup_steps=cfg.profile_warmup_steps,
        device=accelerator.device,
        batch_size=cfg.dataset.batch_size,
    )
    for step in range(...):
        profiler.begin_step()
        with profiler.section("data_fetch"):
            batch = next(train_iter)
        with profiler.section("forward"):
            with accelerator.autocast():
                loss, metrics = model(batch)
        with profiler.section("backward"):
            accelerator.backward(loss)
        with profiler.section("optimizer"):
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
        profiler.end_step()
        if profiler.should_finalize():
            profiler.finalize(out_dir=experiment_dir, logger=logger)
            # Training continues with profiling disabled — overhead drops to
            # zero (context managers short-circuit when not enabled).
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch

# Section names measured per training step. The order here drives both the
# JSON output and the printed table. Stored in __init__ rather than at module
# scope so subclasses can extend cleanly.
DEFAULT_SECTIONS: tuple[str, ...] = (
    "data_fetch",
    "forward",
    "backward",
    "optimizer",
    "logging",
    "total_step",
)


class Profiler:
    """
    Faithful wall-clock + CUDA-memory profiler for a training loop.

    Reports two timings per section:

    * **CPU wall** — time the CPU spent in the section, measured with
      ``time.perf_counter()``. No syncs are inserted, so the CPU is free to
      race ahead of the GPU stream (which is what real training does). For
      GPU-launching sections (forward/backward/optimizer) this is dominated
      by kernel-launch overhead unless the stream queue is full or a
      ``.item()`` / ``.cpu()`` call inside the section forces a sync.
    * **GPU stream wall** — elapsed time on the CUDA stream between a
      non-blocking start/end ``torch.cuda.Event`` pair recorded around the
      section. For GPU-heavy sections this equals kernel duration. For
      CPU-only sections it equals the GPU's *idle gap* during the section
      (the GPU was doing nothing, but stream wall-clock advanced). The sum
      of per-section GPU walls across one step equals ``total_step``, which
      makes it a useful "where is the GPU's time going" attribution. Events
      are resolved at :meth:`finalize` time after one ``cuda.synchronize``.

    ``total_step`` is the CPU wall between ``begin_step`` and ``end_step`` —
    equal to the real per-step time in steady state (CPU is rate-limited by
    GPU stream backpressure). The reported ``it_per_sec`` matches what
    unprofiled training would show, modulo the ~µs of bookkeeping overhead.

    The previous version of this profiler inserted ``cuda.synchronize()`` at
    every section boundary, which killed CPU/GPU overlap and inflated
    ``total_step`` by 5-10 %. This version reproduces real training
    throughput while still giving per-section GPU times.

    Parameters
    ----------
    enabled:
        When False, every method is a no-op so the loop pays nothing.
    n_steps:
        Number of measured steps to collect (after ``warmup_steps``). The
        profiler finalizes itself when ``warmup_steps + n_steps`` total
        ``begin_step`` calls have been made.
    warmup_steps:
        Steps to skip before recording measurements. Avoids the kernel-compile
        / cudnn-benchmark spike on iter 0 polluting the percentile stats.
        The window start is anchored at the *end* of warmup with a single
        ``cuda.synchronize()`` so steady-state numbers aren't contaminated by
        startup work.
    device:
        Torch device used for the CUDA-event pool. ``None`` (or CPU) disables
        GPU-side timing — only CPU walls are reported.
    batch_size:
        Logical batch size, used to compute ``samples_per_sec`` in the
        summary. ``None`` reports throughput only as ``it_per_sec``.
    sections:
        Names of sections expected; only these are tracked. Calling
        :meth:`section` with another name raises (catches typos at runtime).
    """

    def __init__(
        self,
        enabled: bool = False,
        n_steps: int = 500,
        warmup_steps: int = 50,
        device: torch.device | None = None,
        batch_size: int | None = None,
        sections: tuple[str, ...] = DEFAULT_SECTIONS,
    ) -> None:
        self.enabled = bool(enabled)
        self.n_steps = int(n_steps)
        self.warmup_steps = int(warmup_steps)
        self.device = device
        self.batch_size = batch_size
        self.sections = tuple(sections)
        # CPU wall times per section (seconds). ``total_step`` lives here too.
        self._times: dict[str, list[float]] = {name: [] for name in self.sections}
        # Non-blocking CUDA event pairs per section — resolved to durations
        # at finalize() time after one synchronize. Skipped for total_step
        # (covered by CPU wall) and for any section that didn't launch GPU
        # work; the Event objects are cheap (~few hundred bytes) so we
        # record them unconditionally and let the report drop empties.
        self._gpu_events: dict[str, list[tuple[Any, Any]]] = {
            name: [] for name in self.sections if name != "total_step"
        }
        self._gpu_times: dict[str, list[float]] = {}
        self._step_idx = 0  # increments on each begin_step()
        self._step_t0: float | None = None
        self._wall_start: float | None = None
        self._finalized = False
        # Use CUDA events / sync only when a CUDA device is supplied AND
        # CUDA is actually available. torch.cuda.synchronize(None) on CPU
        # would raise, so gate explicitly.
        self._cuda_sync = bool(device is not None and torch.cuda.is_available())

    # ------------------------------------------------------------------
    # Public hooks called from the training loop
    # ------------------------------------------------------------------

    def begin_step(self) -> None:
        if not self.enabled or self._finalized:
            return
        # Anchor the measurement window at the *end* of warmup with a single
        # synchronize. Inside the window, no syncs — the CPU loop races
        # ahead, and CUDA events provide GPU timings without blocking.
        if self._wall_start is None and self._step_idx == self.warmup_steps:
            if self._cuda_sync:
                torch.cuda.synchronize(self.device)
                # Peak memory tracking limited to the measurement window.
                torch.cuda.reset_peak_memory_stats(self.device)
            self._wall_start = time.perf_counter()
        self._step_t0 = time.perf_counter()

    def end_step(self) -> None:
        if not self.enabled or self._finalized:
            return
        # No sync here — total_step is CPU wall, which in steady state
        # equals real GPU per-step time via stream backpressure.
        if self._step_idx >= self.warmup_steps and self._step_t0 is not None:
            self._times["total_step"].append(time.perf_counter() - self._step_t0)
        self._step_idx += 1

    @contextmanager
    def section(self, name: str):
        if not self.enabled or self._finalized:
            yield
            return
        if name not in self._times:
            raise KeyError(
                f"Profiler.section({name!r}): unknown section. "
                f"Known: {sorted(self._times.keys())}"
            )
        # Record a non-blocking start event for GPU timing. This call only
        # queues the event on the current stream — it does not wait.
        in_window = self._step_idx >= self.warmup_steps
        start_evt = end_evt = None
        if self._cuda_sync and in_window:
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            cpu_dt = time.perf_counter() - t0
            if in_window:
                self._times[name].append(cpu_dt)
                if start_evt is not None:
                    end_evt.record()
                    self._gpu_events[name].append((start_evt, end_evt))

    def should_finalize(self) -> bool:
        """Return True once warmup_steps + n_steps begin_step() calls have run."""
        if not self.enabled or self._finalized:
            return False
        return self._step_idx >= self.warmup_steps + self.n_steps

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def _resolve_gpu_events(self) -> None:
        """
        Convert recorded CUDA event pairs into per-call durations (seconds).
        Must be called after a synchronize so all events have fired; we do
        that once at the top of :meth:`finalize`. Idempotent.
        """
        if not self._cuda_sync or self._gpu_times:
            return
        for name, event_pairs in self._gpu_events.items():
            if not event_pairs:
                continue
            # Event.elapsed_time returns milliseconds; convert to seconds for
            # parity with the CPU wall numbers in self._times.
            self._gpu_times[name] = [
                start.elapsed_time(end) / 1000.0 for start, end in event_pairs
            ]

    def summary(self) -> dict[str, Any]:
        """Aggregate the recorded timings into a JSON-serialisable dict."""
        wall = (
            time.perf_counter() - self._wall_start
            if self._wall_start is not None
            else 0.0
        )
        sections_out: dict[str, dict[str, Any]] = {}

        def _stats(vals: list[float]) -> dict[str, float]:
            a = np.asarray(vals, dtype=np.float64)
            return {
                "mean_ms": float(a.mean() * 1000.0),
                "median_ms": float(np.median(a) * 1000.0),
                "p95_ms": float(np.percentile(a, 95) * 1000.0),
                "p99_ms": float(np.percentile(a, 99) * 1000.0),
                "max_ms": float(a.max() * 1000.0),
                "total_s": float(a.sum()),
                "n_samples": int(a.size),
            }

        for name in self.sections:
            entry: dict[str, Any] = {}
            cpu_vals = self._times.get(name, [])
            if cpu_vals:
                entry["cpu"] = _stats(cpu_vals)
            gpu_vals = self._gpu_times.get(name, [])
            if gpu_vals:
                # Drop ~0 entries (CPU-only sections like data_fetch may still
                # have recorded events that fire instantly on an empty stream;
                # report only when there's meaningful GPU work).
                non_trivial = [v for v in gpu_vals if v > 1e-6]
                if non_trivial:
                    entry["gpu"] = _stats(non_trivial)
            if entry:
                sections_out[name] = entry

        effective_n = max(0, self._step_idx - self.warmup_steps)
        it_per_sec = effective_n / wall if wall > 0 else 0.0

        out: dict[str, Any] = {
            "profile_n_steps": int(self.n_steps),
            "warmup_steps": int(self.warmup_steps),
            "effective_n_steps": int(effective_n),
            "batch_size": (
                int(self.batch_size) if self.batch_size is not None else None
            ),
            "wall_time_s": float(wall),
            "it_per_sec": float(it_per_sec),
            "sections": sections_out,
        }
        if self.batch_size is not None:
            out["samples_per_sec"] = float(it_per_sec * self.batch_size)

        if torch.cuda.is_available():
            try:
                out["cuda"] = {
                    "device_name": torch.cuda.get_device_name(self.device),
                    "peak_alloc_mb": float(
                        torch.cuda.max_memory_allocated(self.device) / 1024.0**2
                    ),
                    "current_alloc_mb": float(
                        torch.cuda.memory_allocated(self.device) / 1024.0**2
                    ),
                    "peak_reserved_mb": float(
                        torch.cuda.max_memory_reserved(self.device) / 1024.0**2
                    ),
                }
            except Exception as e:
                out["cuda"] = {"error": str(e)}
        return out

    def finalize(
        self,
        out_dir: str | None = None,
        logger: logging.Logger | Any | None = None,
        filename: str = "profile.json",
    ) -> dict[str, Any]:
        """
        Build the summary, log it, optionally save JSON, and disable further
        profiling (so subsequent steps run without overhead).
        """
        # Single drain-the-pipe synchronize before reading event timings.
        # This is the *only* sync inside the measurement window — every
        # per-step / per-section sync the old profiler did has been removed.
        if self._cuda_sync:
            torch.cuda.synchronize(self.device)
        self._resolve_gpu_events()
        summary = self.summary()
        log = logger or logging.getLogger(__name__)
        self._log_block(summary, log)

        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, filename)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            self._log(log, f"Saved profile summary to {path}")
            summary["path"] = path

        # Disable further timing collection — the context-managers become
        # no-ops, so the rest of training pays nothing for an enabled flag.
        self.enabled = False
        self._finalized = True
        return summary

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _log(self, log, msg: str) -> None:
        # accelerate.logging loggers accept ``main_process_only`` as a kwarg;
        # stdlib loggers don't. Try / fall back.
        try:
            log.info(msg, main_process_only=True)
        except TypeError:
            log.info(msg)

    def _log_block(self, summary: dict[str, Any], log) -> None:
        lines: list[str] = []
        lines.append("=== Profile summary ===")
        lines.append(
            f"  steps={summary['effective_n_steps']}/{summary['profile_n_steps']} "
            f"(warmup={summary['warmup_steps']})  "
            f"wall={summary['wall_time_s']:.2f}s"
        )
        if summary.get("samples_per_sec") is not None:
            lines.append(
                f"  throughput: {summary['it_per_sec']:.2f} it/s, "
                f"{summary['samples_per_sec']:.0f} samples/s "
                f"(batch_size={summary['batch_size']})"
            )
        else:
            lines.append(f"  throughput: {summary['it_per_sec']:.2f} it/s")
        if "cuda" in summary and "device_name" in summary["cuda"]:
            cuda = summary["cuda"]
            lines.append(
                f"  CUDA ({cuda['device_name']}): "
                f"peak_alloc={cuda['peak_alloc_mb']:.0f} MiB, "
                f"peak_reserved={cuda['peak_reserved_mb']:.0f} MiB"
            )
        # Two rows per section: CPU wall and GPU duration (from events).
        # CPU is "where was the CPU in this step"; GPU is "what kernels ran"
        # — they overlap in steady state, so they don't sum to total_step.
        lines.append(
            "  {:<14s} {:>4s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s}".format(
                "section",
                "kind",
                "mean(ms)",
                "med(ms)",
                "p95(ms)",
                "p99(ms)",
                "total(s)",
            )
        )
        for name in self.sections:
            entry = summary["sections"].get(name)
            if entry is None:
                continue
            for kind in ("cpu", "gpu"):
                stats = entry.get(kind)
                if stats is None:
                    continue
                lines.append(
                    "  {:<14s} {:>4s} {:>10.2f} {:>10.2f} {:>10.2f} {:>10.2f} {:>10.2f}".format(
                        name,
                        kind,
                        stats["mean_ms"],
                        stats["median_ms"],
                        stats["p95_ms"],
                        stats["p99_ms"],
                        stats["total_s"],
                    )
                )
        self._log(log, "\n".join(lines))
