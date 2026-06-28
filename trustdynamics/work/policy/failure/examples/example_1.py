"""
Failure policy for workflow example_1
"""

from trustdynamics.work.policy.failure.failure import FailurePolicy


failure_policy = {
    "Task 3": FailurePolicy(
        transitions={
            "Task 1": 1.0 / 3,
            "Task 2": 1.0 / 3,
            "Task 3": 1.0 / 3,
        }
    ),
    "Task 4": FailurePolicy(
        transitions={
            "Task 2": 0.5,
            "Task 3": 0.5,
        }
    ),
}