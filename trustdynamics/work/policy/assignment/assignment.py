class AssignmentPolicy:

    def __init__(
        self,
        assignees: set[int] | None = None,
        exclude_assignees: set[int] | None = None,
    ):
        self.assignees = assignees
        self.exclude_assignees = exclude_assignees or set()


if __name__ == "__main__":
    assignment_policy = AssignmentPolicy(
        assignees={1, 2},
        exclude_assignees=None,
    )