"""
General-purpose structured logging for deepcrew-ai.

This is deliberately separate from :mod:`deepcrew.observability` (OpenTelemetry
spans) and from the per-run :class:`~deepcrew.StreamEvent` queue (a view-only
event bus). It is a plain ``logging`` layer: opt-in, toggleable, with named
profiles, colour (via the optional ``rich`` extra), JSON output, and consistent
structured fields.

Nothing is emitted until :func:`configure_logging` is called or a ``DEEPCREW_LOG*``
environment variable is set — a bare ``import deepcrew`` stays silent, per the
Python logging HOWTO guidance for libraries.

Levels follow the stdlib convention (``CRITICAL``/``ERROR``/``WARNING``/``INFO``/
``DEBUG``) plus a ``TRACE`` (5) level for token-level detail, registered the same
way ``structlog``/``loguru`` add it.

Public API (also re-exported from ``deepcrew``):

* :func:`configure_logging` — turn logging on / pick a profile
* :func:`get_logger` — a structured logger for your own components
* :class:`LoggingConfig` — the full config object
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections.abc import MutableMapping
from dataclasses import dataclass, replace
from typing import Any, Literal, TextIO, cast

__all__ = [
    "TRACE",
    "LoggingConfig",
    "StructuredAdapter",
    "configure_logging",
    "elapsed_ms",
    "get_logger",
]

# --- TRACE level -----------------------------------------------------------

TRACE = 5
logging.addLevelName(TRACE, "TRACE")

_ROOT_LOGGER_NAME = "deepcrew"
_HANDLER_TAG = "_deepcrew_managed"
_configured = False

Profile = Literal["dev", "prod", "verbose", "quiet"]

# Profile -> (level, format, color, timestamps). Applied first; explicit
# LoggingConfig fields and env vars override afterwards.
_PROFILES: dict[str, dict[str, Any]] = {
    "dev": {"level": logging.DEBUG, "format": "text", "color": True, "timestamps": True},
    "prod": {"level": logging.INFO, "format": "json", "color": False, "timestamps": True},
    "verbose": {"level": TRACE, "format": "text", "color": True, "timestamps": True},
    "quiet": {"level": logging.WARNING, "format": "text", "color": None, "timestamps": False},
}

# LogRecord attributes that are *not* user-supplied structured extras.
_RESERVED_LOGRECORD_KEYS = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "taskName",
        "message",
        "asctime",
    }
)
_LOG_METHOD_KWARGS = frozenset({"exc_info", "stack_info", "stacklevel", "extra"})

_ANSI = {
    "TRACE": "\033[38;5;244m",  # grey
    "DEBUG": "\033[36m",  # cyan
    "INFO": "\033[32m",  # green
    "WARNING": "\033[33m",  # yellow
    "ERROR": "\033[31m",  # red
    "CRITICAL": "\033[1;37;41m",  # white on red
}
_ANSI_DIM = "\033[2m"
_ANSI_RESET = "\033[0m"


# --- config --------------------------------------------------------------


@dataclass
class LoggingConfig:
    """Full configuration for deepcrew-ai's logging layer.

    Every field is individually overridable. A ``profile`` only presets the
    four rendering knobs (level / format / color / timestamps); explicit fields
    and ``DEEPCREW_LOG*`` env vars win over it, in that order (env is highest so
    deployment always has the last word).
    """

    level: int | str = logging.INFO
    profile: Profile | None = None
    format: Literal["text", "json"] = "text"
    color: bool | None = None  # None -> auto (TTY, honouring NO_COLOR/FORCE_COLOR)
    timestamps: bool = True
    stream: TextIO | None = None  # default: sys.stderr
    logger_name: str = _ROOT_LOGGER_NAME
    propagate: bool = False
    """Whether the ``deepcrew`` logger propagates to the root logger. Off by
    default so deepcrew logs are not double-emitted by an app's root handler;
    set True if you want to route them through your own root configuration."""


def _coerce_level(value: int | str) -> int:
    if isinstance(value, int):
        return value
    named = logging.getLevelName(str(value).upper())
    return named if isinstance(named, int) else logging.INFO


def _env_bool(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    low = raw.strip().lower()
    if low in {"1", "true", "yes", "on"}:
        return True
    if low in {"0", "false", "no", "off"}:
        return False
    return None


def _has_log_env() -> bool:
    return any(k.startswith("DEEPCREW_LOG") for k in os.environ)


def _resolve_config(
    profile_or_config: str | LoggingConfig | None,
    overrides: dict[str, Any],
) -> LoggingConfig:
    """Merge, lowest precedence first: profile preset -> caller args -> env vars."""
    if isinstance(profile_or_config, LoggingConfig):
        base = profile_or_config
    elif isinstance(profile_or_config, str):
        base = LoggingConfig(profile=cast(Profile, profile_or_config))
    else:
        base = LoggingConfig()
    if overrides:
        base = replace(base, **overrides)

    profile = os.environ.get("DEEPCREW_LOG_PROFILE") or base.profile
    cfg = LoggingConfig()
    if profile in _PROFILES:
        for key, val in _PROFILES[profile].items():
            setattr(cfg, key, val)
        cfg.profile = cast(Profile, profile)

    # caller args win over the profile preset (only the knobs they actually set)
    default = LoggingConfig()
    _fields = ("level", "format", "color", "timestamps", "stream", "logger_name", "propagate")
    caller_set = set(overrides)
    if isinstance(profile_or_config, LoggingConfig):
        caller_set |= {f for f in _fields if getattr(base, f) != getattr(default, f)}
    for f in _fields:
        if f in caller_set:
            setattr(cfg, f, getattr(base, f))

    # env vars win over everything
    if "DEEPCREW_LOG_LEVEL" in os.environ:
        cfg.level = os.environ["DEEPCREW_LOG_LEVEL"]
    env_fmt = os.environ.get("DEEPCREW_LOG_FORMAT")
    if env_fmt in {"text", "json"}:
        cfg.format = cast(Literal["text", "json"], env_fmt)
    env_time = _env_bool("DEEPCREW_LOG_TIME")
    if env_time is not None:
        cfg.timestamps = env_time
    env_color_raw = os.environ.get("DEEPCREW_LOG_COLOR")
    if env_color_raw is not None:
        cfg.color = (
            None if env_color_raw.strip().lower() == "auto" else _env_bool("DEEPCREW_LOG_COLOR")
        )
    return cfg


def _should_color(stream: TextIO, explicit: bool | None) -> bool:
    if explicit is not None:
        return explicit
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("FORCE_COLOR") is not None:
        return True
    try:
        return bool(stream.isatty())
    except Exception:
        return False


# --- formatters ---------------------------------------------------------


def _record_extras(record: logging.LogRecord) -> dict[str, Any]:
    return {
        k: v
        for k, v in record.__dict__.items()
        if k not in _RESERVED_LOGRECORD_KEYS and not k.startswith("_")
    }


class ColorFormatter(logging.Formatter):
    """Human-readable single-line formatter with optional ANSI colour."""

    def __init__(self, *, color: bool, timestamps: bool) -> None:
        super().__init__()
        self.color = color
        self.timestamps = timestamps

    def format(self, record: logging.LogRecord) -> str:
        level = record.levelname
        parts: list[str] = []
        if self.timestamps:
            ts = time.strftime("%H:%M:%S", time.localtime(record.created))
            parts.append(f"{_ANSI_DIM}{ts}{_ANSI_RESET}" if self.color else ts)

        if self.color:
            tint = _ANSI.get(level, "")
            parts.append(f"{tint}{level:<8}{_ANSI_RESET}")
            parts.append(f"{_ANSI_DIM}{record.name}{_ANSI_RESET}")
        else:
            parts.append(f"{level:<8}")
            parts.append(record.name)

        parts.append(record.getMessage())

        extras = _record_extras(record)
        if extras:
            kv = " ".join(f"{k}={_render_value(v)}" for k, v in extras.items())
            parts.append(f"{_ANSI_DIM}{kv}{_ANSI_RESET}" if self.color else kv)

        line = "  ".join(parts)
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


class JsonFormatter(logging.Formatter):
    """One JSON object per line: ts, level, logger, msg, plus structured extras."""

    def __init__(self, *, timestamps: bool) -> None:
        super().__init__()
        self.timestamps = timestamps

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if self.timestamps:
            payload["ts"] = (
                time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
                + f".{int(record.msecs):03d}Z"
            )
        payload.update(_record_extras(record))
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _render_value(value: Any) -> str:
    text = str(value)
    return f'"{text}"' if (" " in text or not text) else text


def _escape_markup(text: str) -> str:
    """Escape rich console markup so literal ``[...]`` in messages is not parsed."""
    try:
        from rich.markup import escape

        return str(escape(text))
    except Exception:  # pragma: no cover - rich is present on this path
        return text


class _RichExtrasFilter(logging.Filter):
    """Fold structured extras into the message so RichHandler renders them."""

    def filter(self, record: logging.LogRecord) -> bool:
        extras = _record_extras(record)
        if extras:
            kv = " ".join(f"{k}={_render_value(v)}" for k, v in extras.items())
            record.msg = f"{_escape_markup(record.getMessage())}  [dim]{_escape_markup(kv)}[/dim]"
            record.args = ()
        return True


# --- adapter ----------------------------------------------------------


class StructuredAdapter(logging.LoggerAdapter):
    """A ``LoggerAdapter`` that turns unknown keyword args into structured fields.

    ``log.info("routing decision", mode="parallel", agents=3)`` records ``mode``
    and ``agents`` as extras that every deepcrew formatter renders (``key=value``
    suffix for text, top-level keys for JSON).

    Use :meth:`bind` to attach context carried on every subsequent call.
    """

    def __init__(self, logger: logging.Logger, context: dict[str, Any] | None = None) -> None:
        super().__init__(logger, context or {})

    def process(
        self, msg: Any, kwargs: MutableMapping[str, Any]
    ) -> tuple[Any, MutableMapping[str, Any]]:
        extra: dict[str, Any] = dict(self.extra or {})
        passthrough: dict[str, Any] = {}
        for key, val in list(kwargs.items()):
            if key in _LOG_METHOD_KWARGS:
                passthrough[key] = val
            else:
                extra[key] = val
        caller_extra = passthrough.pop("extra", None)
        if isinstance(caller_extra, dict):
            extra.update(caller_extra)
        passthrough["extra"] = extra
        return msg, passthrough

    def bind(self, **fields: Any) -> StructuredAdapter:
        merged = {**(self.extra or {}), **fields}
        return StructuredAdapter(self.logger, merged)

    def trace(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        if self.isEnabledFor(TRACE):
            m, kw = self.process(msg, kwargs)
            self.logger.log(TRACE, m, *args, **kw)


# --- public entry points ---------------------------------------------


def get_logger(name: str) -> StructuredAdapter:
    """Return a structured logger namespaced under ``deepcrew.<name>``.

    Safe to call at import time. Use it in your own tools, memory providers,
    verifiers and app glue so a single :func:`configure_logging` call styles
    library and application logs identically::

        from deepcrew import get_logger
        log = get_logger("myapp.tools.pricing")
        log.info("price lookup", sku=sku, region=region, duration_ms=12.4)
    """
    if _has_log_env() and not _configured:
        configure_logging(None)
    short = name.split("deepcrew.", 1)[-1] if name.startswith("deepcrew.") else name
    full = _ROOT_LOGGER_NAME if short == _ROOT_LOGGER_NAME else f"{_ROOT_LOGGER_NAME}.{short}"
    return StructuredAdapter(logging.getLogger(full))


def configure_logging(
    profile_or_config: str | LoggingConfig | None = "dev",
    /,
    **overrides: Any,
) -> logging.Logger:
    """Turn deepcrew-ai's logging on and return the configured ``deepcrew`` logger.

    ``configure_logging()``                -> the ``dev`` profile
    ``configure_logging("prod")``          -> a named profile
    ``configure_logging(level="DEBUG", format="json")``  -> ad-hoc knobs
    ``configure_logging(LoggingConfig(...))``            -> full control

    Idempotent: calling it again replaces the single managed handler rather than
    stacking. ``DEEPCREW_LOG*`` env vars always override the arguments.
    """
    global _configured

    cfg = _resolve_config(profile_or_config, overrides)
    stream = cfg.stream or sys.stderr
    use_color = _should_color(stream, cfg.color)
    level = _coerce_level(cfg.level)

    logger = logging.getLogger(cfg.logger_name)
    logger.setLevel(level)
    logger.propagate = cfg.propagate

    # drop any previously-managed handler (idempotency)
    for existing in list(logger.handlers):
        if getattr(existing, _HANDLER_TAG, False):
            logger.removeHandler(existing)

    handler: logging.Handler
    if cfg.format == "text" and use_color and _rich_available():
        handler = _make_rich_handler(stream, cfg.timestamps)
        handler.addFilter(_RichExtrasFilter())
    elif cfg.format == "json":
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JsonFormatter(timestamps=cfg.timestamps))
    else:
        handler = logging.StreamHandler(stream)
        handler.setFormatter(ColorFormatter(color=use_color, timestamps=cfg.timestamps))

    handler.setLevel(level)
    setattr(handler, _HANDLER_TAG, True)
    logger.addHandler(handler)

    _configured = True
    return logger


def _rich_available() -> bool:
    try:
        import rich.logging  # noqa: F401
    except Exception:
        return False
    return True


def _make_rich_handler(stream: TextIO, timestamps: bool) -> logging.Handler:
    from rich.console import Console
    from rich.logging import RichHandler

    return RichHandler(
        console=Console(file=stream, force_terminal=True),
        show_time=timestamps,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
        keywords=[],
    )


def elapsed_ms(start: float) -> float:
    """Milliseconds since a ``time.perf_counter()`` reading, rounded to 0.1 ms."""
    return round((time.perf_counter() - start) * 1000, 1)
