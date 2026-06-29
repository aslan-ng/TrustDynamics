"""
Review policy for workflow example_1
"""

from trustdynamics.work.policy.review.review import ReviewPolicy


review_policy = {
    "Task 1": ReviewPolicy(
        reviewers={1},
        exclude_reviewers=None,
        score_threshold=None,
    ),
    "Task 4": ReviewPolicy(
        reviewers={1, 2},
        exclude_reviewers=None,
        minimum_reviews=1,
        score_threshold=None,
    ),
}