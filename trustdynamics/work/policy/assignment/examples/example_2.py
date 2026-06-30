"""
Assignment policy for workflow example_2
"""

from trustdynamics.work.policy.assignment.assignment import AssignmentPolicy


assignment_policy = {
    "Task 2": AssignmentPolicy(
        workers={1, 3},
        exclude_workers={2},
    ),
    "Task 3": AssignmentPolicy(
        workers={1, 2},
    ),
    "Task 4": AssignmentPolicy(
        exclude_workers={3},
    ),
}