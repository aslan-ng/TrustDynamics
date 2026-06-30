from enum import StrEnum


class TaskState(StrEnum):

    NOT_READY = "not_ready"  # Prereqruisites not satisfied yet
    READY = "ready"  # Ready for execution
    IN_PROGRESS = "in_progress"  # Execution started
    EXECUTED = "executed"  # Execution completed
    UNDER_REVIEW = "under_review"  # Execution completed and review started
    COMPLETED = "completed"  # Execution completed, and review passed (if applicable)
    FAILED = "failed"  # Either execution failed, or review not passed