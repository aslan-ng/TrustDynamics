"""
Failure policy for workflow example_2
"""

from trustdynamics.work.policy.failure.failure import FailurePolicy


failure_policy = {
    "Task 3": FailurePolicy(
        transitions={
            "Task 1": 0.5,
            "Task 3": 0.5,
        }
    ),
    "Task 4": FailurePolicy(
        transitions={
            "Task 1": 0.5,
            "Task 4": 0.5,
        }
    ),
    "Task 5": FailurePolicy(
        transitions={
            "Task 2": 1.0,
        }
    ),
}