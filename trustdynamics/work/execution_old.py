from trustdynamics.work.project.project import Project


class Execution:

    def __init__(self, project: Project):
        if project.selected_plan is None:
            raise ValueError("Project must have a selected plan.")

        self.project = project
        self.plan = project.selected_plan

        self.task_status = {
            task_name: "remaining"
            for task_name in self.project.workflow.tasks
        }

        self.review_scores: dict[str, dict[int, float]] = {
            task_name: {}
            for task_name in self.project.review_policy
        }

        self.task_workers: dict[str, int | None] = {
            task_name: None
            for task_name in self.project.workflow.tasks
        }

        self._update_ready_tasks()
        self.state = {}
        self._update_state()

    def _update_ready_tasks(self) -> None:
        completed_tasks = {
            task
            for task, status in self.task_status.items()
            if status == "completed"
        }

        for task_name, task in self.project.workflow.tasks.items():
            if self.task_status[task_name] != "remaining":
                continue

            if not task.prerequisite_sets:
                self.task_status[task_name] = "ready"
                continue

            for prerequisite_set in task.prerequisite_sets:
                if set(prerequisite_set).issubset(completed_tasks):
                    self.task_status[task_name] = "ready"
                    break

    @property
    def assignment_requirements(self) -> dict[str, dict]:
        return {
            task_name: self.task_assignment_requirements(task_name)
            for task_name in self.startable_tasks
        }

    @property
    def startable_tasks(self) -> set[str]:
        return {
            task
            for task, status in self.task_status.items()
            if status == "ready"
        }

    @property
    def in_progress_tasks(self) -> set[str]:
        return {
            task
            for task, status in self.task_status.items()
            if status == "in_progress"
        }

    @property
    def tasks_under_review(self) -> set[str]:
        return {
            task
            for task, status in self.task_status.items()
            if status == "under_review"
        }

    @property
    def completed_tasks(self) -> set[str]:
        return {
            task
            for task, status in self.task_status.items()
            if status == "completed"
        }

    @property
    def failed_tasks(self) -> set[str]:
        return {
            task
            for task, status in self.task_status.items()
            if status == "failed"
        }

    @property
    def is_complete(self) -> bool:
        return all(
            self.task_status[task_name] == "completed"
            for task_name in self.project.workflow.tasks
        )
    def start_task(self, task_name: str, worker: int) -> dict:
        self._validate_known_task(task_name)

        if self.task_status[task_name] != "ready":
            raise ValueError(
                f"Task '{task_name}' is not ready to start."
            )

        self._validate_worker_assignment(task_name, worker)

        self.task_status[task_name] = "in_progress"
        self.task_workers[task_name] = worker

        self._update_state()
        return self.state
    
    def task_assignment_requirements(self, task_name: str) -> dict:
        self._validate_known_task(task_name)

        assignment_policy = self.project.assignment_policy.get(task_name)

        if assignment_policy is None:
            return {
                "eligible_workers": None,
                "excluded_workers": set(),
            }

        if assignment_policy.workers is None:
            eligible_workers = None
        else:
            eligible_workers = (
                assignment_policy.workers
                - assignment_policy.exclude_workers
            )

        return {
            "eligible_workers": eligible_workers,
            "excluded_workers": assignment_policy.exclude_workers,
        }

    def _validate_worker_assignment(
        self,
        task_name: str,
        worker: int,
    ) -> None:
        assignment_policy = self.project.assignment_policy.get(task_name)

        if assignment_policy is None:
            return

        if worker in assignment_policy.exclude_workers:
            raise ValueError(
                f"Worker '{worker}' is excluded from task '{task_name}'."
            )

        if assignment_policy.workers is not None:
            if worker not in assignment_policy.workers:
                raise ValueError(
                    f"Worker '{worker}' is not eligible for task "
                    f"'{task_name}'."
                )

    def finish_task(
        self,
        task_name: str,
        success: bool,
    ) -> dict:
        self._validate_known_task(task_name)

        if self.task_status[task_name] != "in_progress":
            raise ValueError(
                f"Task '{task_name}' is not in progress."
            )

        if not success:
            self.task_status[task_name] = "failed"
            self._update_state()
            return self.state

        if task_name in self.project.review_policy:
            self.task_status[task_name] = "under_review"
        else:
            self.task_status[task_name] = "completed"
            self._update_ready_tasks()
        self._update_state()
        return self.state

    def submit_review(
        self,
        task_name: str,
        reviewer: int,
        score: float,
    ) -> dict:
        self._validate_known_task(task_name)

        if self.task_status[task_name] != "under_review":
            raise ValueError(
                f"Task '{task_name}' is not under review."
            )

        review_policy = self.project.review_policy[task_name]

        if not (0.0 <= score <= 1.0):
            raise ValueError("Review score must be between 0 and 1.")

        if reviewer in review_policy.exclude_reviewers:
            raise ValueError(
                f"Reviewer '{reviewer}' is excluded from reviewing "
                f"task '{task_name}'."
            )

        if review_policy.reviewers is not None:
            if reviewer not in review_policy.reviewers:
                raise ValueError(
                    f"Reviewer '{reviewer}' is not eligible to review "
                    f"task '{task_name}'."
                )

        self.review_scores[task_name][reviewer] = score
        self._update_state()
        return self.state

    def finish_review(self, task_name: str) -> dict:
        self._validate_known_task(task_name)

        if self.task_status[task_name] != "under_review":
            raise ValueError(
                f"Task '{task_name}' is not under review."
            )

        if self._review_passes(task_name):
            self.task_status[task_name] = "completed"
            self._update_ready_tasks()
        else:
            self.task_status[task_name] = "failed"
        self._update_state()
        return self.state

    def _review_passes(self, task_name: str) -> bool:
        review_policy = self.project.review_policy[task_name]
        scores = self.review_scores[task_name]

        if len(scores) < review_policy.minimum_reviews:
            raise ValueError(
                f"Task '{task_name}' needs at least "
                f"{review_policy.minimum_reviews} reviews before review "
                f"can be finished."
            )

        if review_policy.score_threshold is None:
            return True

        total_score = sum(scores.values())
        return total_score >= review_policy.score_threshold

    def review_requirements(self, task_name: str) -> dict:
        self._validate_known_task(task_name)

        if task_name not in self.project.review_policy:
            raise ValueError(
                f"Task '{task_name}' does not have a review policy."
            )

        review_policy = self.project.review_policy[task_name]
        scores = self.review_scores[task_name]

        submitted_reviewers = set(scores)
        submitted_count = len(scores)
        remaining_reviews_needed = max(
            review_policy.minimum_reviews - submitted_count,
            0,
        )
        current_score = sum(scores.values())

        if review_policy.reviewers is None:
            eligible_reviewers = None
        else:
            eligible_reviewers = (
                review_policy.reviewers
                - review_policy.exclude_reviewers
                - submitted_reviewers
            )

        return {
            "eligible_reviewers": eligible_reviewers,
            "excluded_reviewers": review_policy.exclude_reviewers,
            "minimum_reviews": review_policy.minimum_reviews,
            "submitted_reviewers": submitted_reviewers,
            "remaining_reviews_needed": remaining_reviews_needed,
            "score_threshold": review_policy.score_threshold,
            "current_score": current_score,
            "can_finish_review": remaining_reviews_needed == 0,
        }

    def return_to_task(
        self,
        failed_task: str,
        return_task: str,
    ) -> dict:
        self._validate_known_task(failed_task)
        self._validate_known_task(return_task)

        if self.task_status[failed_task] != "failed":
            raise ValueError(
                f"Task '{failed_task}' has not failed."
            )

        if failed_task not in self.project.failure_policy:
            raise ValueError(
                f"Task '{failed_task}' failed, but no failure policy "
                f"is defined."
            )

        failure_policy = self.project.failure_policy[failed_task]

        if return_task not in failure_policy.transitions:
            raise ValueError(
                f"Task '{failed_task}' cannot return to "
                f"task '{return_task}' according to its failure policy."
            )

        return_stage_index = self._stage_index_containing_task(return_task)

        for stage in self.plan[return_stage_index:]:
            for task_name in stage:
                self.task_status[task_name] = "remaining"

                if task_name in self.review_scores:
                    self.review_scores[task_name] = {}

        self.task_status[return_task] = "ready"
        self._update_ready_tasks()
        self._update_state()
        return self.state

    def _stage_index_containing_task(self, task_name: str) -> int:
        for index, stage in enumerate(self.plan):
            if task_name in stage:
                return index

        raise ValueError(
            f"Task '{task_name}' is not in the selected plan."
        )

    def _validate_known_task(self, task_name: str) -> None:
        if task_name not in self.task_status:
            raise ValueError(f"Unknown task: {task_name}")

    def _update_state(self) -> None:
        review_requirements = {}
        for task_name in self.tasks_under_review:
            review_requirements[task_name] = self.review_requirements(task_name)

        failure_options = {}
        for task_name in self.failed_tasks:
            if task_name in self.project.failure_policy:
                failure_options[task_name] = set(
                    self.project.failure_policy[task_name].transitions
                )
            else:
                failure_options[task_name] = None

        self.state = {
            "assignment_requirements": self.assignment_requirements,
            "task_workers": self.task_workers,
            "startable_tasks": self.startable_tasks,
            "in_progress_tasks": self.in_progress_tasks,
            "tasks_under_review": self.tasks_under_review,
            "review_requirements": review_requirements,
            "failed_tasks": self.failed_tasks,
            "failure_options": failure_options,
            "completed_tasks": self.completed_tasks,
            "is_complete": self.is_complete,
        }

    @property
    def summary(self) -> dict:
        return {
            "Task status": self.task_status,
            "Task workers": self.task_workers,
            "Review scores": self.review_scores,
            "State": self.state,
        }


if __name__ == "__main__":
    from pprint import pprint

    from trustdynamics.work.project.examples.example_1 import project

    possible_plans = project.possible_plans()
    plan = possible_plans[0]
    project.choose_plan(plan)

    execution = Execution(project)

    pprint(execution.summary, sort_dicts=False)

    # Starting a startable task
    print(">>> starting a startable task")
    state = execution.state
    startable_tasks = list(state["startable_tasks"])
    task = startable_tasks[0]

    state = execution.state
    task = list(state["startable_tasks"])[0]

    assignment_requirements = state["assignment_requirements"][task]
    eligible_workers = assignment_requirements["eligible_workers"]
    worker = list(eligible_workers)[0]  # Choose a worker

    state = execution.start_task(task, worker=worker)
    pprint(state, sort_dicts=False)

    # Task is being worked by the worker (human or AI)
    print(">>> task in progress")

    # Finish the task with success or failure
    print(">>> task executed")
    state = execution.finish_task(task, success=True)
    pprint(state, sort_dicts=False)

    # Review process
    if task in state["tasks_under_review"]:
        requirements = state["review_requirements"][task]
        pprint(requirements, sort_dicts=False)

        eligible_reviewers = requirements["eligible_reviewers"]
        remaining_reviews_needed = requirements["remaining_reviews_needed"]

        if eligible_reviewers is None:
            reviewers_to_submit = list(
                range(remaining_reviews_needed)
            )
        else:
            reviewers_to_submit = list(eligible_reviewers)[
                :remaining_reviews_needed
            ]

        for reviewer in reviewers_to_submit:
            state = execution.submit_review(
                task,
                reviewer=reviewer,
                score=1.0,
            )

        if state["review_requirements"][task]["can_finish_review"]:
            state = execution.finish_review(task)

    if task in state["failed_tasks"]:
        failure_options = state["failure_options"][task]

        if failure_options is not None:
            return_task = list(failure_options)[0]  # NEEDS UPDATING
            state = execution.return_to_task(
                failed_task=task,
                return_task=return_task,
            )

    pprint(execution.summary, sort_dicts=False)