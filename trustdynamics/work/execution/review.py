from trustdynamics.work.project.project import Project
from trustdynamics.work.execution.states import TaskState


class ReviewMixin:
    project: Project
    workers: set[int]
    task_states: dict[str, TaskState]
    task_reviews: dict[str, dict[int, float]]
    def _update_ready_tasks(self) -> None:
        ...

    def _validate_review_feasibility(self) -> None:
        for task_name in self.project.review_policy:
            eligible_reviewers = self.eligible_reviewers(task_name)
            minimum_reviews = self.minimum_reviews(task_name)

            if len(eligible_reviewers) < minimum_reviews:
                raise ValueError(
                    f"Task '{task_name}' requires "
                    f"{minimum_reviews} reviewers, but only "
                    f"{len(eligible_reviewers)} are available in this execution."
                )

    def needs_review(self, task_name: str) -> bool:
        return self.task_states[task_name] == TaskState.UNDER_REVIEW

    def has_review_policy(self, task_name: str) -> bool:
        return task_name in self.project.review_policy

    def eligible_reviewers(self, task_name: str) -> set[int]:
        policy = self.project.review_policy.get(task_name)

        if policy is None:
            return set()

        if policy.reviewers is None:
            reviewers = set(self.workers)
        else:
            reviewers = set(policy.reviewers)

        return (reviewers - policy.exclude_reviewers) & self.workers

    def minimum_reviews(self, task_name: str) -> int:
        policy = self.project.review_policy.get(task_name)

        if policy is None:
            return 0

        return policy.minimum_reviews

    def score_threshold(self, task_name: str) -> float:
        policy = self.project.review_policy.get(task_name)

        if policy is None:
            return 1.0

        return policy.score_threshold

    def review_passed(self, task_name: str) -> bool:
        reviews = self.task_reviews[task_name]

        if len(reviews) < self.minimum_reviews(task_name):
            return False

        average_score = sum(reviews.values()) / len(reviews)

        return average_score >= self.score_threshold(task_name)

    def submit_review(
        self,
        task_name: str,
        reviewer: int,
        score: bool | float,
    ) -> None:
        if not self.needs_review(task_name):
            raise ValueError(
                f"Task '{task_name}' is not under review."
            )

        if reviewer not in self.eligible_reviewers(task_name):
            raise ValueError(
                f"Reviewer '{reviewer}' is not eligible to review "
                f"task '{task_name}'."
            )

        if isinstance(score, bool):
            normalized_score = float(score)
        else:
            normalized_score = score

        if not (0.0 <= normalized_score <= 1.0):
            raise ValueError(
                "Review score must be between 0 and 1."
            )

        self.task_reviews[task_name][reviewer] = normalized_score

    def finish_review(self, task_name: str) -> None:
        if not self.needs_review(task_name):
            raise ValueError(
                f"Task '{task_name}' is not under review."
            )

        if self.review_passed(task_name):
            self.task_states[task_name] = TaskState.COMPLETED
            self._update_ready_tasks()
        else:
            self.task_states[task_name] = TaskState.FAILED