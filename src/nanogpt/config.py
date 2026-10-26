"""Strict, typed configuration for models and training runs.

Training runs are expensive, so a misspelled key must fail before a GPU is
rented rather than being silently ignored. Every dataclass below is loaded by
one generic strict loader (``from_mapping``) that rejects unknown and missing
fields, and every field is consumed somewhere in the pipeline.
"""

from __future__ import annotations

import dataclasses
import math
import os
import types
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, Self, TypeVar, Union, get_args, get_origin, get_type_hints

import yaml

from nanogpt.utils import canonical_json, sha256_text

T = TypeVar("T")

# The project-wide hard ceiling; individual runs may only lower it.
MAX_TOTAL_BUDGET_USD = 100.0


# --------------------------------------------------------------------------- loader


def from_mapping(cls: type[T], values: Any, *, where: str | None = None) -> T:
    """Build dataclass ``cls`` from a mapping, recursing into nested fields."""

    where = where or cls.__name__
    if not isinstance(values, dict):
        raise ValueError(f"{where} must be a mapping")
    hints = get_type_hints(cls)
    known = {item.name: item for item in fields(cls)}  # type: ignore[arg-type]
    unknown = set(values) - set(known)
    if unknown:
        raise ValueError(f"unknown {where} fields: {', '.join(sorted(map(str, unknown)))}")
    kwargs: dict[str, Any] = {}
    for name, item in known.items():
        if name in values:
            kwargs[name] = _coerce(hints[name], values[name], f"{where}.{name}")
        elif item.default is dataclasses.MISSING and item.default_factory is dataclasses.MISSING:
            raise ValueError(f"missing {where} field: {name}")
    try:
        return cls(**kwargs)
    except TypeError as error:
        raise ValueError(f"invalid {where}: {error}") from error


def _coerce(hint: Any, value: Any, where: str) -> Any:
    origin = get_origin(hint)
    if origin in (Union, types.UnionType):
        options = get_args(hint)
        if value is None and type(None) in options:
            return None
        (inner,) = [option for option in options if option is not type(None)]
        return _coerce(inner, value, where)
    if origin is list:
        if not isinstance(value, list):
            raise ValueError(f"{where} must be a list")
        (inner,) = get_args(hint)
        return [_coerce(inner, item, f"{where}[{index}]") for index, item in enumerate(value)]
    if origin is Literal:
        if value not in get_args(hint):
            raise ValueError(f"{where} must be one of {list(get_args(hint))}, got {value!r}")
        return value
    if dataclasses.is_dataclass(hint):
        return from_mapping(hint, value, where=where)  # type: ignore[arg-type]
    if hint is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if hint in (int, float, str, bool) and (
        not isinstance(value, hint) or (hint is not bool and isinstance(value, bool))
    ):
        raise ValueError(f"{where} must be {hint.__name__}, got {value!r}")
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return values


# --------------------------------------------------------------------------- model


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Architecture parameters for the decoder-only Transformer.

    This is the single place architecture invariants are validated; model
    modules trust a constructed ``ModelConfig``.
    """

    vocab_size: int
    max_seq_len: int
    n_layers: int
    d_model: int
    n_query_heads: int
    n_kv_heads: int
    ffn_hidden_size: int
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-6
    tie_embeddings: bool = True
    qk_norm: bool = True

    def __post_init__(self) -> None:
        for name in (
            "vocab_size",
            "max_seq_len",
            "n_layers",
            "d_model",
            "n_query_heads",
            "n_kv_heads",
            "ffn_hidden_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.d_model % self.n_query_heads:
            raise ValueError("d_model must be divisible by n_query_heads")
        if self.n_query_heads % self.n_kv_heads:
            raise ValueError("n_query_heads must be divisible by n_kv_heads")
        if self.head_dim % 2:
            raise ValueError("attention head dimension must be even for RoPE")
        for name in ("rope_theta", "rms_norm_eps"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a finite positive number, got {value!r}")
        for name in ("tie_embeddings", "qk_norm"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean, got {getattr(self, name)!r}")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_query_heads

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> Self:
        try:
            return from_mapping(cls, values, where="ModelConfig")
        except TypeError as error:  # pragma: no cover - from_mapping normalizes these
            raise ValueError(str(error)) from error

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        return cls.from_mapping(_read_yaml(Path(path)))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def fingerprint(self) -> str:
        return sha256_text(self.canonical_json())


# --------------------------------------------------------------------------- run


@dataclass(frozen=True)
class BudgetConfig:
    hourly_rate_usd: float | None = None
    total_usd: float = 0.0
    checkpoint_reserve_seconds: float = 120.0


@dataclass(frozen=True)
class TokenizerConfig:
    backend: Literal["hf", "reference"] = "hf"
    vocab_size: int = 32_768
    # Characters of training text sampled (in deterministic stream order).
    train_chars: int = 1_000_000_000


@dataclass(frozen=True)
class DataConfig:
    source: Literal["fixture", "fineweb"]
    revision: str | None = None
    validation_fraction: float = 0.005
    shard_tokens: int = 100_000_000
    # Stop preparation after this many training tokens (None = whole source).
    max_tokens: int | None = None
    budget_usd: float = 0.0


@dataclass(frozen=True)
class SourceConfig:
    """One conversation/prompt source for SFT or GRPO."""

    kind: Literal["arithmetic", "gsm8k", "smoltalk"]
    examples: int
    seed: int = 0
    revision: str | None = None
    subsets: list[str] = field(default_factory=list)
    license: str | None = None


@dataclass(frozen=True)
class StageConfig:
    """Optimizer, schedule, and cadence shared by every gradient stage."""

    max_steps: int
    micro_batch_size: int
    learning_rate: float
    gradient_accumulation_steps: int = 1
    warmup_fraction: float = 0.0
    min_lr_fraction: float = 0.1
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    log_every: int = 10
    eval_every: int = 0
    checkpoint_every: int = 0
    budget_usd: float = 0.0
    # Overrides the run-level ``compile`` for this stage (None = inherit).
    compile: bool | None = None


@dataclass(frozen=True)
class BaseStageConfig(StageConfig):
    sequence_length: int = 1024


@dataclass(frozen=True)
class SFTStageConfig(StageConfig):
    sources: list[SourceConfig] = field(default_factory=list)


@dataclass(frozen=True)
class GRPOStageConfig(StageConfig):
    sources: list[SourceConfig] = field(default_factory=list)
    group_size: int = 8
    temperature: float = 0.8
    top_k: int | None = 50
    max_new_tokens: int = 256
    ratio_clip_epsilon: float = 0.2
    reference_kl_beta: float = 0.02
    pilot_prompts: int = 128
    min_correct_fraction: float = 0.05
    min_mixed_group_fraction: float = 0.2


@dataclass(frozen=True)
class EvalConfig:
    validation_batches: int = 20
    reasoning_prompts: int = 200
    max_new_tokens: int = 128
    budget_usd: float = 0.0


@dataclass(frozen=True)
class ProbeConfig:
    """Throughput probe: time base steps per candidate model, project base cost."""

    candidates: list[str]
    warmup_steps: int = 10
    measure_steps: int = 50
    safety_margin: float = 0.20


@dataclass(frozen=True)
class RunConfig:
    name: str
    seed: int
    model: str
    data: DataConfig
    base: BaseStageConfig
    sft: SFTStageConfig
    grpo: GRPOStageConfig
    device: str = "auto"
    dtype: Literal["float32", "bfloat16"] = "float32"
    compile: bool = False
    deterministic: bool = True
    # Keep only the newest N step checkpoints per stage (0 = keep all).
    keep_checkpoints: int = 0
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    probe: ProbeConfig | None = None

    def __post_init__(self) -> None:
        total = self.budget.total_usd
        if not math.isfinite(total) or not 0 <= total <= MAX_TOTAL_BUDGET_USD:
            raise ValueError(f"budget.total_usd must be between 0 and {MAX_TOTAL_BUDGET_USD}")
        rate = self.budget.hourly_rate_usd
        if rate is not None and (not math.isfinite(rate) or rate <= 0):
            raise ValueError("budget.hourly_rate_usd must be a finite positive number")
        stages = (self.data, self.base, self.sft, self.grpo, self.eval)
        allocated = sum(stage.budget_usd for stage in stages)
        if allocated > total + 1e-9:
            raise ValueError(f"stage budgets (${allocated:.2f}) exceed budget.total_usd ${total}")
        if self.keep_checkpoints < 0:
            raise ValueError("keep_checkpoints must be >= 0")
        if self.grpo.group_size < 2:
            raise ValueError("grpo.group_size must be at least 2")

    @property
    def paid(self) -> bool:
        return self.budget.total_usd > 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def apply_environment(config: RunConfig) -> RunConfig:
    """Apply the documented environment overrides for cost inputs."""

    budget = config.budget
    if rate := os.environ.get("NANOGPT_HOURLY_RATE_USD"):
        budget = dataclasses.replace(budget, hourly_rate_usd=float(rate))
    if total := os.environ.get("NANOGPT_TOTAL_BUDGET_USD"):
        budget = dataclasses.replace(budget, total_usd=float(total))
    return dataclasses.replace(config, budget=budget)


def load_run_config(path: str | Path) -> tuple[RunConfig, ModelConfig]:
    """Load a run YAML plus the model profile it names, applying env overrides."""

    source = Path(path)
    run = apply_environment(from_mapping(RunConfig, _read_yaml(source), where="run"))
    return run, ModelConfig.from_yaml(resolve_model_path(run.model, source))


def resolve_model_path(name: str, run_config_path: Path) -> Path:
    """A bare profile name refers to ``configs/models/<name>.yaml``."""

    if name.endswith((".yaml", ".yml")):
        candidate = Path(name)
        return candidate if candidate.is_absolute() else run_config_path.parent / candidate
    return run_config_path.resolve().parent.parent / "models" / f"{name}.yaml"


# --------------------------------------------------------------------------- paths


STAGES = ("base", "sft", "grpo")


@dataclass(frozen=True)
class RunPaths:
    """Runtime layout: ``$NANOGPT_HOME/runs/<run>/{tokenizer,data,checkpoints,eval,logs}``."""

    root: Path

    @classmethod
    def for_run(cls, run_name: str, home: str | Path | None = None) -> Self:
        base = home if home is not None else os.environ.get("NANOGPT_HOME", "artifacts")
        return cls(Path(base).expanduser().resolve() / "runs" / run_name)

    @property
    def tokenizer(self) -> Path:
        return self.root / "tokenizer"

    def data(self, stage: str) -> Path:
        return self.root / "data" / stage

    def checkpoints(self, stage: str) -> Path:
        return self.root / "checkpoints" / stage

    @property
    def eval(self) -> Path:
        return self.root / "eval"

    @property
    def logs(self) -> Path:
        return self.root / "logs"
