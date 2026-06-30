"""
Review policy for workflow example_2
"""

from trustdynamics.work.policy.review.review import ReviewPolicy


review_policy = {
    "Task 1": ReviewPolicy(
        reviewers={1, 2},
        exclude_reviewers=None,
        minimum_reviews=2,
    ),
    "Task 3": ReviewPolicy(
        exclude_reviewers={1},
        minimum_reviews=2,
    ),
    "Task 4": ReviewPolicy(
        minimum_reviews=2,
        score_threshold=1.0,
    ),
}