"""Disk-based caching helpers for expensive external API calls.

Provides a decorator that caches string results from tool _run methods in
output/.cache/ with a configurable TTL.  This eliminates redundant Pegelonline,
eHYD, and Open-Meteo requests across agent iterations and crew reruns.
"""

import hashlib
import json
import os
from datetime import datetime, timedelta
from functools import wraps
from typing import Callable

OUTPUT_DIR = os.environ.get("PEGELHUB_OUTPUT_DIR", "output")
CACHE_DIR = os.path.join(OUTPUT_DIR, ".cache")


def _cache_key(func_name: str, *args, **kwargs) -> str:
    """Stable MD5 key from function name and arguments."""
    payload = json.dumps(
        {"func": func_name, "args": args, "kwargs": kwargs},
        sort_keys=True,
        default=str,
    )
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def disk_cache(ttl_hours: float = 24.0) -> Callable:
    """Decorator that caches a tool's string result on disk.

    Args:
        ttl_hours: How long a cached entry remains valid.  Use shorter TTLs for
            forecast data and longer TTLs for historical measurements.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            os.makedirs(CACHE_DIR, exist_ok=True)
            key = _cache_key(func.__name__, *args, **kwargs)
            cache_path = os.path.join(CACHE_DIR, f"{key}.json")

            if os.path.exists(cache_path):
                mtime = datetime.fromtimestamp(os.path.getmtime(cache_path))
                if datetime.now() - mtime < timedelta(hours=ttl_hours):
                    try:
                        with open(cache_path, "r", encoding="utf-8") as f:
                            cached = json.load(f)
                        return cached["result"]
                    except Exception:
                        # Fall through to re-fetch on any cache read error
                        pass

            result = func(*args, **kwargs)
            try:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump({"result": result}, f, ensure_ascii=False)
            except Exception:
                # Caching is best-effort; never break the tool call
                pass
            return result

        return wrapper
    return decorator
