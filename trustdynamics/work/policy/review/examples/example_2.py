"""
Review policy for workflow example_1
"""

from trustdynamics.work.policy.review.review import ReviewPolicy


review_policy = {
    ReviewPolicy(
        task="Task 1",
        reviewers={1, 2},
        exclude_reviewers=None,
        minimum_reviews=2,
        score_threshold=None,
    ),
    ReviewPolicy(
        task="Task 3",
        reviewers=None,
        exclude_reviewers={1},
        minimum_reviews=2,
        score_threshold=None,
    ),
    ReviewPolicy(
        task="Task 3",
        reviewers=None,
        exclude_reviewers=None,
        minimum_reviews=2,
        score_threshold=1.0,
    ),
}