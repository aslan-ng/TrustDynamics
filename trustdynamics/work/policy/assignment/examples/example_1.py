"""
Assignment policy for workflow example_1
"""

from trustdynamics.work.policy.assignment.assignment import AssignmentPolicy


assignment_policy = {
    "Task 1": AssignmentPolicy(
        workers={1, 2},
    ),
    "Task 4": AssignmentPolicy(
        exclude_workers={3},
    ),
}