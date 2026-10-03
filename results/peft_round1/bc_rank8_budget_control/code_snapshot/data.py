"""Real FALCON held-in data access for the SSM pilot.

This module deliberately refuses ``held_out`` files.  It is a local temporal
development protocol, not an EvalAI/FALCON evaluation implementation.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

APST_SRC = Path(__file__).resolve().parents[2] / "APST" / "src"
if str(APST_SRC) not in sys.path:
    sys.path.insert(0, str(APST_SRC))
from apst.data.load import list_sessions, load_nwb_file  # noqa: E402

DEFAULT_ROOT = Path("/mnt/data/work_host/SPINT/SPINT-main/data")
ALLOWED_SPLITS = frozenset(("held_in", "minival"))


@dataclass(frozen=True)
class Recording:
    task: str
    split: str
    session: str
    path: Path
    neural: np.ndarray
    behavior: np.ndarray
    trial_change: np.ndarray
    eval_mask: np.ndarray
    trial_bounds: tuple[tuple[int, int], ...]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _bounds(trial_change: np.ndarray, length: int) -> tuple[tuple[int, int], ...]:
    """Turn official trial-change marks into contiguous no-crossing ranges."""
    mark = np.asarray(trial_change).reshape(-1)
    if mark.size != length:
        # Some official readers expose trial transitions in a compact form.
        # A single recording is still a safe fallback, but is recorded below.
        return ((0, length),)
    starts = np.flatnonzero(mark.astype(bool))
    starts = np.unique(np.r_[0, starts[(starts > 0) & (starts < length)]])
    ends = np.r_[starts[1:], length]
    return tuple((int(a), int(b)) for a, b in zip(starts, ends) if b > a)


def load_recording(task: str, split: str, session: str, *, root: Path = DEFAULT_ROOT) -> Recording:
    if split not in ALLOWED_SPLITS:
        raise PermissionError(f"pilot only permits {sorted(ALLOWED_SPLITS)}, got {split!r}")
    rows = [r for r in list_sessions(task, split, root=root) if r["session"] == session]
    if len(rows) != 1:
        raise KeyError(f"expected exactly one {task}/{split}/{session}, got {len(rows)}")
    row = rows[0]
    neural, behavior, trial_change, eval_mask = load_nwb_file(row["path"], task)
    x = np.asarray(neural, dtype=np.float32)
    y = np.asarray(behavior, dtype=np.float32)
    mask = np.asarray(eval_mask, dtype=bool).reshape(-1)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError(f"invalid official array geometry x={x.shape}, y={y.shape}")
    if mask.size != x.shape[0]:
        raise ValueError(f"eval mask length {mask.size} != recording length {x.shape[0]}")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("official recording has non-finite neural or behavior values")
    return Recording(task, split, session, Path(row["path"]).resolve(), x, y,
                     np.asarray(trial_change), mask, _bounds(trial_change, x.shape[0]))


def visible_sessions(task: str, *, root: Path = DEFAULT_ROOT) -> dict[str, list[dict]]:
    """List only the explicitly authorized FALCON surfaces with file hashes."""
    result: dict[str, list[dict]] = {}
    for split in ("held_in", "minival"):
        result[split] = [
            {"session": str(r["session"]), "path": str(Path(r["path"]).resolve()),
             "sha256": sha256_file(Path(r["path"]))}
            for r in list_sessions(task, split, root=root)
        ]
    return result


def source_target_plan(task: str, *, root: Path = DEFAULT_ROOT) -> dict:
    """Produce a deterministic pilot plan without touching held-out data.

    Shared session IDs in held-in/minival are reported as within-session temporal
    development.  The latest held-in session is withheld from fitting as an
    additional local cross-session target whenever two held-in sessions exist.
    """
    held = visible_sessions(task, root=root)["held_in"]
    mini = visible_sessions(task, root=root)["minival"]
    held_ids = [r["session"] for r in held]
    mini_ids = [r["session"] for r in mini]
    if len(held_ids) < 2 or not mini_ids:
        raise ValueError("pilot needs >=2 held-in sessions and >=1 minival session")
    cross_target = sorted(held_ids)[-1]
    source = [s for s in held_ids if s != cross_target]
    return {
        "task": task, "data_root": str(Path(root).resolve()), "authorized_splits": ["held_in", "minival"],
        "source_held_in_sessions": source,
        "cross_session_local_dev": {"target_session": cross_target, "surface": "held_in", "role": "local_only"},
        "minival_targets": mini_ids,
        "minival_interpretation": "held-in within-session temporal development when session IDs overlap held-in",
        "held_in": held, "minival": mini,
        "held_out_accessed": False,
    }


def split_support_query(record: Recording, *, support_trials: int, sequence: int,
                        query_start_trials: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return valid starts for non-overlapping prefix support and tail query.

    Entire trials are assigned to either side.  A segment of ``sequence`` bins
    therefore never crosses a trial boundary or sees support labels in query.
    """
    if support_trials < 1 or sequence < 1:
        raise ValueError("support_trials and sequence must be positive")
    query_start_trials = support_trials if query_start_trials is None else query_start_trials
    if query_start_trials < support_trials:
        raise ValueError("query_start_trials must be >= support_trials")
    usable = [(a, b) for a, b in record.trial_bounds if b - a >= sequence]
    if len(usable) <= query_start_trials:
        raise ValueError(f"{record.session}: only {len(usable)} usable trials for query cutoff={query_start_trials}")
    support, query = usable[:support_trials], usable[query_start_trials:]
    def starts(bounds: Iterable[tuple[int, int]]) -> np.ndarray:
        pieces = [np.arange(a, b - sequence + 1, dtype=np.int64) for a, b in bounds]
        return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)
    return starts(support), starts(query)


def source_normalizer(records: Iterable[Recording]) -> dict[str, np.ndarray]:
    rows = list(records)
    if not rows:
        raise ValueError("source normalization needs records")
    x = np.concatenate([r.neural for r in rows], axis=0).astype(np.float64)
    y = np.concatenate([r.behavior for r in rows], axis=0).astype(np.float64)
    x_mean, x_std = x.mean(0), x.std(0)
    y_mean, y_std = y.mean(0), y.std(0)
    x_std[x_std < 1e-6] = 1.0; y_std[y_std < 1e-6] = 1.0
    return {"x_mean": x_mean.astype(np.float32), "x_std": x_std.astype(np.float32),
            "y_mean": y_mean.astype(np.float32), "y_std": y_std.astype(np.float32)}


def manifest_json(plan: dict) -> str:
    return json.dumps(plan, indent=2, sort_keys=True) + "\n"
