"""
Assignment policy for workflow example_1
"""

from trustdynamics.work.policy.assignment.assignment import AssignmentPolicy


assignment_policy = {
    "Task 3": AssignmentPolicy(
        assignees={1, 2},
    ),
    "Task 4": AssignmentPolicy(
        exclude_assignees={3},
    ),
}