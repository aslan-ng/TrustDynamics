class ReviewPolicy:

    def __init__(
        self,
        task: str,
        reviewers: set[int] | None = None,          # None means all eligible reviewers
        exclude_reviewers: set[int] | None = None,  # Excluded from reviewers
        minimum_reviews: int = 1,                   # Minimum completed reviews required
        score_threshold: float | None = None,       # None means no score threshold
    ):
        if minimum_reviews < 1:
            raise ValueError(
                "minimum_reviews must be at least 1."
            )

        if score_threshold is not None:
            if not (0.0 <= score_threshold <= 1.0):
                raise ValueError(
                    "score_threshold must be between 0 and 1."
                )

        self.task = task
        self.reviewers = reviewers
        self.exclude_reviewers = exclude_reviewers or set()
        self.minimum_reviews = minimum_reviews
        self.score_threshold = score_threshold


if __name__ == "__main__":
    review_policy = ReviewPolicy(
        task="Task 1",
        reviewers={1, 2, 3},
        exclude_reviewers=None,
        minimum_reviews=2,
        score_threshold=None,
    )