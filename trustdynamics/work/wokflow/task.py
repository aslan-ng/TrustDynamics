class Task:

    def __init__(
        self,
        name: str,
        prerequisite_sets: list[list[str]] | None = None,  # Outer list is OR, and inner list is AND
    ):
        self.name = name  # Used as the stable task identifier
        self.prerequisite_sets = prerequisite_sets or []  # Default: no prerequisite tasks


if __name__ == "__main__":
    task = Task(
        name="Design",
        prerequisite_sets=None,
    )