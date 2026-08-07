"""Keep W&B terminal messages visible at the correct Isaac Sim log level."""

from __future__ import annotations

import re
import sys
import threading
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, TextIO

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


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
