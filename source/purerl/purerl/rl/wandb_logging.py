"""Keep W&B terminal messages visible at the correct Isaac Sim log level."""

from __future__ import annotations

import re
import sys
import threading
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, TextIO

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


class _WandbConfigProxy:
    """Allow RSL-RL to refresh config values when resuming a W&B run."""

    def __init__(self, config: Any, *, allow_config_change: bool) -> None:
        self._config = config
        self._allow_config_change = allow_config_change

    def update(self, *args: Any, **kwargs: Any) -> Any:
        if self._allow_config_change:
            kwargs.setdefault("allow_val_change", True)
        return self._config.update(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._config, name)


class _WandbModuleProxy:
    """Forward W&B calls while wrapping its run-dependent config object."""

    def __init__(self, wandb: Any, *, allow_config_change: bool) -> None:
        self._wandb = wandb
        self._allow_config_change = allow_config_change

    @property
    def config(self) -> _WandbConfigProxy:
        return _WandbConfigProxy(
            self._wandb.config,
            allow_config_change=self._allow_config_change,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wandb, name)


def configure_rsl_rl_wandb(*, allow_config_change: bool) -> None:
    """Configure RSL-RL's W&B writer for new or resumed remote runs.

    RSL-RL stores the local log directory and its complete configs in W&B.
    Those values legitimately differ when a checkpoint continues in a new
    local directory, while W&B rejects changes to an existing run's config by
    default. Keep that validation for new runs and relax it only when the
    caller explicitly selected a W&B resume mode.
    """

    from rsl_rl.utils import wandb_utils

    wandb = wandb_utils.wandb
    if isinstance(wandb, _WandbModuleProxy):
        wandb = wandb._wandb
    wandb_utils.wandb = _WandbModuleProxy(
        wandb,
        allow_config_change=allow_config_change,
    )


def route_wandb_to_carb() -> None:
    """Send W&B info to stdout and warnings/errors to matching Carb levels."""

    import carb
    import wandb
    from wandb.errors import term

    _install_wandb_carb_logging(wandb, term, carb, stdout=sys.stdout)


def _install_wandb_carb_logging(
    wandb: Any,
    term: Any,
    carb: Any,
    *,
    stdout: TextIO = sys.stdout,
) -> None:
    printed_messages: set[tuple[str, str, bool]] = set()
    output_lock = threading.Lock()
    stdout_supports_color = bool(getattr(stdout, "isatty", lambda: False)())

    def log_info(message: str) -> None:
        # Carb info messages are filtered from Isaac Sim's console by default.
        # stdout keeps W&B status visible without Kit labelling it as py stderr.
        stdout.write(f"{message}\n")
        stdout.flush()

    def make_emitter(
        level: str,
        log_fn: Callable[[str], None],
        show_flag: str,
        *,
        preserve_style: bool = False,
    ) -> Callable[..., None]:
        def emit(
            message: str = "",
            newline: bool = True,
            repeat: bool = True,
            prefix: bool = True,
        ) -> None:
            del newline  # Carb records complete log entries instead of terminal fragments.
            raw_message = str(message)
            clean_message = _ANSI_ESCAPE.sub("", raw_message)
            message_key = (level, clean_message, prefix)
            rendered_message = raw_message if preserve_style else clean_message
            prefix_text = getattr(term, "LOG_STRING", "wandb") if preserve_style else "wandb"
            rendered_lines = [
                f"{prefix_text}: {line}" if prefix else line
                for line in (rendered_message.splitlines() or [""])
            ]
            if getattr(term, "_silent", False) or not getattr(term, show_flag, True):
                fallback_logger = getattr(term, "_logger", None)
                if fallback_logger is not None:
                    fallback = getattr(fallback_logger, level)
                    for line in rendered_lines:
                        fallback(line)
                return

            with output_lock:
                if not repeat and message_key in printed_messages:
                    return
                if not repeat:
                    printed_messages.add(message_key)
                for line in rendered_lines:
                    log_fn(line)

        return emit

    @contextmanager
    def no_dynamic_text():
        # W&B's spinner writes cursor-control sequences to stderr. Kit labels
        # those writes as errors and can splice them into the next stdout line.
        yield None

    info = make_emitter(
        "info",
        log_info,
        "_show_info",
        preserve_style=stdout_supports_color,
    )
    warning = make_emitter("warning", carb.log_warn, "_show_warnings")
    error = make_emitter("error", carb.log_error, "_show_errors")

    wandb.termlog = term.termlog = info
    wandb.termwarn = term.termwarn = warning
    wandb.termerror = term.termerror = error
    term.dynamic_text = no_dynamic_text
