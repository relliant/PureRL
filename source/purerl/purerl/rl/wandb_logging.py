"""Route W&B terminal messages through Isaac Sim's Carb logger."""

from __future__ import annotations

import re
import threading
from collections.abc import Callable
from typing import Any

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def route_wandb_to_carb() -> None:
    """Send W&B info, warnings, and errors to matching Carb log levels."""

    import carb
    import wandb
    from wandb.errors import term

    _install_wandb_carb_logging(wandb, term, carb)


def _install_wandb_carb_logging(wandb: Any, term: Any, carb: Any) -> None:
    printed_messages: set[tuple[str, str, bool]] = set()
    output_lock = threading.Lock()

    def make_emitter(
        level: str,
        log_fn: Callable[[str], None],
        show_flag: str,
    ) -> Callable[..., None]:
        def emit(
            message: str = "",
            newline: bool = True,
            repeat: bool = True,
            prefix: bool = True,
        ) -> None:
            del newline  # Carb records complete log entries instead of terminal fragments.
            clean_message = _ANSI_ESCAPE.sub("", str(message))
            message_key = (level, clean_message, prefix)
            rendered_lines = [
                f"wandb: {line}" if prefix else line
                for line in (clean_message.splitlines() or [""])
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

    info = make_emitter("info", carb.log_info, "_show_info")
    warning = make_emitter("warning", carb.log_warn, "_show_warnings")
    error = make_emitter("error", carb.log_error, "_show_errors")

    wandb.termlog = term.termlog = info
    wandb.termwarn = term.termwarn = warning
    wandb.termerror = term.termerror = error
