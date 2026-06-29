"""
Assignment policy for workflow example_2
"""

from trustdynamics.work.policy.assignment.assignment import AssignmentPolicy


assignment_policy = {
    "Task 2": AssignmentPolicy(
        assignees={1, 3},
        exclude_assignees={2},
    ),
    "Task 3": AssignmentPolicy(
        assignees={1, 2},
    ),
    "Task 4": AssignmentPolicy(
        exclude_assignees={3},
    ),
}