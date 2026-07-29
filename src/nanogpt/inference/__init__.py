"""Public inference and caching API.

Generation is loaded lazily because model modules depend on the cache types;
eagerly importing generation here would create a model/inference import cycle.
"""

from collections.abc import Callable
from typing import Any, cast

from nanogpt.inference.cache import KVCache, LayerKVCache

__all__ = ["KVCache", "LayerKVCache", "generate", "sample_next_token"]


def __getattr__(name: str) -> Callable[..., Any]:
    if name in {"generate", "sample_next_token"}:
        from nanogpt.inference import generation

        return cast(Callable[..., Any], getattr(generation, name))
    raise AttributeError(name)
