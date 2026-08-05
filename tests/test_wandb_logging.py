from types import SimpleNamespace

from purerl.rl.wandb_logging import _install_wandb_carb_logging


def test_wandb_messages_use_matching_carb_log_levels():
    messages = {"info": [], "warning": [], "error": []}
    carb = SimpleNamespace(
        log_info=messages["info"].append,
        log_warn=messages["warning"].append,
        log_error=messages["error"].append,
    )
    wandb = SimpleNamespace()
    term = SimpleNamespace(
        _silent=False,
        _show_info=True,
        _show_warnings=True,
        _show_errors=True,
    )

    _install_wandb_carb_logging(wandb, term, carb)
    wandb.termlog("Tracking run")
    wandb.termwarn("Network is slow")
    wandb.termerror("Upload failed")

    assert messages == {
        "info": ["wandb: Tracking run"],
        "warning": ["wandb: Network is slow"],
        "error": ["wandb: Upload failed"],
    }


def test_wandb_carb_logging_preserves_visibility_and_repeat_controls():
    info_messages = []
    carb = SimpleNamespace(
        log_info=info_messages.append,
        log_warn=lambda message: None,
        log_error=lambda message: None,
    )
    wandb = SimpleNamespace()
    term = SimpleNamespace(
        _silent=False,
        _show_info=True,
        _show_warnings=True,
        _show_errors=True,
    )

    _install_wandb_carb_logging(wandb, term, carb)
    wandb.termlog("\x1b[1mRun ready\x1b[0m", repeat=False)
    term.termlog("\x1b[1mRun ready\x1b[0m", repeat=False)
    wandb.termlog("without prefix", prefix=False)
    term._show_info = False
    wandb.termlog("hidden")

    assert info_messages == ["wandb: Run ready", "without prefix"]
