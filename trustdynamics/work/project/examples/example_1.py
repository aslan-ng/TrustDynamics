"""
Example project for workflow example_1
"""

from trustdynamics.work.project.project import Project
from trustdynamics.work.wokflow.examples.example_1 import workflow
from trustdynamics.work.policy.failure.examples.example_1 import failure_policy
from trustdynamics.work.policy.review.examples.example_1 import review_policy


project = Project(
    workflow=workflow,
    review_policy=review_policy,
    failure_policy=failure_policy,
)