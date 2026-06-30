"""
Review policy for workflow example_1
"""

from trustdynamics.work.policy.review.review import ReviewPolicy


review_policy = {
    "Task 1": ReviewPolicy(
        reviewers={1},
    ),
    "Task 4": ReviewPolicy(
        exclude_reviewers={3},
        minimum_reviews=2,
        score_threshold=0.8,
    ),
}