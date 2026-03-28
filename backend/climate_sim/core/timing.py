"""Timing utilities for performance profiling."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class TimingEntry:
    """Single timing entry with nested timings."""

    name: str
    total_time: float = 0.0
    call_count: int = 0
    children: Dict[str, TimingEntry] = field(default_factory=dict)
    parent: TimingEntry | None = None


class TimingProfiler:
    """Profiler that tracks nested timing information."""

    def __init__(self) -> None:
        self.root = TimingEntry("root")
        self._stack: List[TimingEntry] = [self.root]
        self._start_times: List[float] = []

    @contextmanager
    def time_block(self, name: str):
        """Context manager for timing a code block."""
        current = self._stack[-1]
        if name not in current.children:
            entry = TimingEntry(name=name, parent=current)
            current.children[name] = entry
        else:
            entry = current.children[name]

        self._stack.append(entry)
        start = time.perf_counter()
        self._start_times.append(start)

        try:
            yield
        finally:
            end = time.perf_counter()
            elapsed = end - start
            entry.total_time += elapsed
            entry.call_count += 1
            self._stack.pop()
            self._start_times.pop()

    def get_total_time(self) -> float:
        """Get total time spent in root (sum of all children)."""
        return sum(child.total_time for child in self.root.children.values())

    def print_summary(self, *, min_time_ms: float = 1.0, indent: str = "  ") -> None:
        """Print a formatted timing summary.

        Parameters
        ----------
        min_time_ms
            Minimum time in milliseconds to display an entry.
        indent
            String to use for indentation.
        """
        total = self.get_total_time()
        if total == 0:
            # Debug: check if we have any children at all
            if len(self.root.children) == 0:
                print("No timing data collected (no timing blocks were entered).")
            else:
                print(
                    f"No timing data collected (found {len(self.root.children)} entries but total time is 0)."
                )
                # Print debug info
                for name, child in self.root.children.items():
                    print(f"  {name}: {child.total_time} s, {child.call_count} calls")
            return

        print("\n" + "=" * 70)
        print("TIMING SUMMARY")
        print("=" * 70)
        # Print children directly, skipping the root entry
        sorted_children = sorted(
            self.root.children.values(),
            key=lambda e: e.total_time,
            reverse=True,
        )
        for child in sorted_children:
            self._print_entry(child, total, min_time_ms, indent, 0)
        print("=" * 70)
        print(f"Total time: {total:.3f} s")
        print()

    def _print_entry(
        self,
        entry: TimingEntry,
        total_time: float,
        min_time_ms: float,
        indent: str,
        level: int,
    ) -> None:
        """Recursively print timing entries."""
        if entry.total_time < min_time_ms / 1000.0:
            return

        prefix = indent * level
        pct = 100.0 * entry.total_time / total_time if total_time > 0 else 0.0
        avg_time = entry.total_time / entry.call_count if entry.call_count > 0 else 0.0

        name_display = entry.name if entry.name != "root" else "Total"
        if entry.call_count > 1:
            print(
                f"{prefix}{name_display:50s} "
                f"{entry.total_time:8.3f} s ({pct:5.1f}%) "
                f"[{entry.call_count} calls, {avg_time:.3f} s avg]"
            )
        else:
            print(f"{prefix}{name_display:50s} {entry.total_time:8.3f} s ({pct:5.1f}%)")

        # Sort children by total time (descending)
        sorted_children = sorted(
            entry.children.values(),
            key=lambda e: e.total_time,
            reverse=True,
        )

        for child in sorted_children:
            self._print_entry(child, total_time, min_time_ms, indent, level + 1)


# Global profiler instance
_default_profiler: TimingProfiler | None = None


def get_profiler() -> TimingProfiler:
    """Get or create the default profiler instance."""
    global _default_profiler
    if _default_profiler is None:
        _default_profiler = TimingProfiler()
    return _default_profiler


def reset_profiler() -> None:
    """Reset the default profiler."""
    global _default_profiler
    _default_profiler = TimingProfiler()


@contextmanager
def time_block(name: str):
    """Context manager for timing using the default profiler."""
    profiler = get_profiler()
    with profiler.time_block(name):
        yield
