class AssignmentPolicy:

    def __init__(
        self,
        workers: set[int] | None = None,
        exclude_workers: set[int] | None = None,
    ):
        self.workers = workers
        self.exclude_workers = exclude_workers or set()


if __name__ == "__main__":
    assignment_policy = AssignmentPolicy(
        workers={1, 2},
        exclude_workers=None,
    )