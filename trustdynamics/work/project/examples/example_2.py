"""
Example project for workflow example_2
"""

from trustdynamics.work.project.project import Project
from trustdynamics.work.workflow.examples.example_2 import workflow
from trustdynamics.work.policy.assignment.examples.example_2 import assignment_policy
from trustdynamics.work.policy.failure.examples.example_2 import failure_policy
from trustdynamics.work.policy.review.examples.example_2 import review_policy


project = Project(
    workflow=workflow,
    assignment_policy=assignment_policy,
    review_policy=review_policy,
    failure_policy=failure_policy,
)