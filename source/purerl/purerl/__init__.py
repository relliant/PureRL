"""PureRL locomotion environments and training integration."""

from .contracts import ACTION_DIM, OBSERVATION_DIM, OBSERVATION_TERMS, TASK_IDS
from .registry import get_task_spec, list_task_ids, register_gymnasium_tasks

__all__ = [
    "ACTION_DIM",
    "OBSERVATION_DIM",
    "OBSERVATION_TERMS",
    "TASK_IDS",
    "get_task_spec",
    "list_task_ids",
    "register_gymnasium_tasks",
]
