from io import StringIO
from types import SimpleNamespace

from purerl.rl.wandb_logging import _install_wandb_carb_logging


class TtyStringIO(StringIO):
    def isatty(self) -> bool:
        return True


def test_wandb_messages_use_stdout_and_matching_carb_log_levels():
    messages = {"info": [], "warning": [], "error": []}
    stdout = StringIO()
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

    _install_wandb_carb_logging(wandb, term, carb, stdout=stdout)
    wandb.termlog("Tracking run")
    wandb.termwarn("Network is slow")
    wandb.termerror("Upload failed")

    assert messages == {
        "info": [],
        "warning": ["wandb: Network is slow"],
        "error": ["wandb: Upload failed"],
    }
    assert stdout.getvalue() == "wandb: Tracking run\n"


def test_wandb_info_preserves_terminal_colors_and_disables_dynamic_stderr_output():
    stdout = TtyStringIO()
    carb = SimpleNamespace(
        log_info=lambda message: None,
        log_warn=lambda message: None,
        log_error=lambda message: None,
    )
    wandb = SimpleNamespace()
    term = SimpleNamespace(
        LOG_STRING="\x1b[34m\x1b[1mwandb\x1b[0m",
        _silent=False,
        _show_info=True,
        _show_warnings=True,
        _show_errors=True,
    )

    _install_wandb_carb_logging(wandb, term, carb, stdout=stdout)
    wandb.termlog("View \x1b[34mrun\x1b[0m")

    with term.dynamic_text() as dynamic_output:
        assert dynamic_output is None

    assert stdout.getvalue() == (
        "\x1b[34m\x1b[1mwandb\x1b[0m: View \x1b[34mrun\x1b[0m\n"
    )


def test_wandb_carb_logging_preserves_visibility_and_repeat_controls():
    stdout = StringIO()
    carb = SimpleNamespace(
        log_info=lambda message: None,
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

    _install_wandb_carb_logging(wandb, term, carb, stdout=stdout)
    wandb.termlog("\x1b[1mRun ready\x1b[0m", repeat=False)
    term.termlog("\x1b[1mRun ready\x1b[0m", repeat=False)
    wandb.termlog("without prefix", prefix=False)
    term._show_info = False
    wandb.termlog("hidden")

    assert stdout.getvalue().splitlines() == ["wandb: Run ready", "without prefix"]
