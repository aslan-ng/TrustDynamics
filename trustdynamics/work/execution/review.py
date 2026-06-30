from trustdynamics.work.project.project import Project
from trustdynamics.work.execution.states import TaskState


class ReviewMixin:
    project: Project
    workers: set[int]
    task_states: dict

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
            reviewers = policy.reviewers
        return reviewers - policy.exclude_reviewers
        
    def minimum_reviews(self, task_name: str) -> int:
        policy = self.project.review_policy.get(task_name)
        if policy is None:
            return 0
        else:
            return policy.minimum_reviews
    
    def score_threshold(self, task_name: str) -> float | None:
        policy = self.project.review_policy.get(task_name)
        if policy is None:
            return None
        return policy.score_threshold
    
    def score_passed(self, task_name: str):
        pass

    def submit_review(self, task_name: str, reviewer: int, review):
        pass

    def finish_review(self, task_name: str):
        pass
