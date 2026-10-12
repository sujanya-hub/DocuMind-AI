"""
metrics_logger.py
-----------------
Safe, zero-risk metrics logger.
- Never crashes your app (every operation is try/except protected)
- Never modifies your existing code's return values
- Never touches your model, database, or FAISS index
- Writes only to metrics_log.json in the same folder
- Safe to leave in production forever

Usage:
    from metrics_logger import log_metric, timed

    # Wrap any function call with timed():
    result, elapsed_s = timed(your_function, arg1, arg2)

    # Log any numbers you want:
    log_metric("my_event", {"key": value})
"""

import json
import os
import time
from datetime import datetime


LOG_FILE = "metrics_log.json"


def timed(fn, *args, **kwargs):
    """
    Calls fn(*args, **kwargs) exactly as normal.
    Also measures how long it took in seconds.
    Returns (original_result, elapsed_seconds).

    If timing itself fails for any reason,
    still returns (original_result, 0.0) — never crashes.
    """
    t_start = time.perf_counter()
    result = fn(*args, **kwargs)          # your code runs normally
    try:
        elapsed = round(time.perf_counter() - t_start, 4)
    except Exception:
        elapsed = 0.0
    return result, elapsed


def log_metric(event: str, data: dict):
    """
    Appends one entry to metrics_log.json.
    Completely safe — wrapped in try/except.
    If the file or folder is not writable, silently does nothing.
    Your app is never affected.
    """
    try:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
        }
        # merge in the data dict
        for k, v in data.items():
            try:
                # ensure values are JSON-serialisable primitives
                json.dumps(v)
                entry[k] = v
            except (TypeError, ValueError):
                entry[k] = str(v)   # fallback: store as string

        # read existing log (or start fresh)
        logs = []
        if os.path.exists(LOG_FILE):
            try:
                with open(LOG_FILE, "r", encoding="utf-8") as f:
                    logs = json.load(f)
                if not isinstance(logs, list):
                    logs = []
            except (json.JSONDecodeError, OSError):
                logs = []

        logs.append(entry)

        # write back atomically (temp file → rename)
        tmp = LOG_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)
        os.replace(tmp, LOG_FILE)          # atomic on all OS

    except Exception:
        pass   # logging NEVER crashes your app


def summarise():
    """
    Call this after collecting metrics to print a summary.
    Run:  python -c "from metrics_logger import summarise; summarise()"
    """
    try:
        if not os.path.exists(LOG_FILE):
            print("No metrics_log.json found yet. Run your app first.")
            return

        with open(LOG_FILE, "r", encoding="utf-8") as f:
            logs = json.load(f)

        if not logs:
            print("Log file is empty.")
            return

        # group by event type
        events = {}
        for entry in logs:
            ev = entry.get("event", "unknown")
            events.setdefault(ev, []).append(entry)

        print(f"\n{'='*50}")
        print(f"  METRICS SUMMARY  ({len(logs)} total entries)")
        print(f"{'='*50}")

        for event_name, entries in events.items():
            print(f"\n[{event_name.upper()}]  —  {len(entries)} entries")

            # find all numeric keys
            numeric_keys = {}
            for entry in entries:
                for k, v in entry.items():
                    if k in ("timestamp", "event"):
                        continue
                    if isinstance(v, (int, float)):
                        numeric_keys.setdefault(k, []).append(v)

            for key, values in numeric_keys.items():
                avg = sum(values) / len(values)
                print(f"  {key}:")
                print(f"    avg={avg:.3f}  min={min(values):.3f}  "
                      f"max={max(values):.3f}  samples={len(values)}")

            # show non-numeric counts (e.g. predicted_class distribution)
            str_keys = {}
            for entry in entries:
                for k, v in entry.items():
                    if k in ("timestamp", "event"):
                        continue
                    if isinstance(v, str):
                        str_keys.setdefault(k, {})
                        str_keys[k][v] = str_keys[k].get(v, 0) + 1

            for key, counts in str_keys.items():
                print(f"  {key} distribution:")
                for val, cnt in sorted(counts.items(),
                                       key=lambda x: -x[1]):
                    print(f"    {val}: {cnt}")

        print(f"\n{'='*50}\n")

    except Exception as e:
        print(f"Could not read metrics: {e}")