class Task:

    def __init__(
        self,
        name: str,
        prerequisite_sets: list[list[str]] | None = None,  # Outer list is OR, and inner list is AND
        failure_transitions: dict | None = None,
    ):
        self.name = name  # Used as the stable task identifier
        self.prerequisite_sets = prerequisite_sets or []  # Default: no prerequisite tasks
        self.failure_transitions = failure_transitions or {
            self.name: 1.0,  # Default: repeat the task
        }
