from enum import Enum
import random

from trustdynamics.work.wokflow.workflow import Workflow


class TaskStatus(Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    UNDER_REVIEW = "under_review"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class Execution:

    def __init__(
        self,
        workflow: Workflow,
        failure_policy: dict[str, dict[str, float]] | None = None,
    ):
        self.workflow = workflow

        self.task_statuses: dict[str, TaskStatus] = {
            task_name: TaskStatus.NOT_STARTED
            for task_name in workflow.task_names
        }

        self.failure_policy = failure_policy or {
            task_name: {task_name: 1.0}
            for task_name in workflow.task_names
        }

        self.history: list[dict] = []

    @property
    def accepted_tasks(self) -> set[str]:
        return {
            task_name
            for task_name, status in self.task_statuses.items()
            if status == TaskStatus.ACCEPTED
        }

    @property
    def incomplete_tasks(self) -> set[str]:
        return self.workflow.task_names - self.accepted_tasks

    @property
    def available_tasks(self) -> set[str]:
        result = set()

        for task_name in self.incomplete_tasks:
            status = self.task_statuses[task_name]

            if status not in {
                TaskStatus.NOT_STARTED,
                TaskStatus.REJECTED,
            }:
                continue

            task = self.workflow.tasks[task_name]

            if not task.prerequisite_sets:
                result.add(task_name)
                continue

            for prerequisite_set in task.prerequisite_sets:
                if all(
                    prerequisite in self.accepted_tasks
                    for prerequisite in prerequisite_set
                ):
                    result.add(task_name)
                    break

        return result

    def start_task(self, task_name: str) -> None:
        self._validate_known_task(task_name)

        if task_name not in self.available_tasks:
            raise ValueError(f"Task '{task_name}' is not available.")

        self.task_statuses[task_name] = TaskStatus.IN_PROGRESS
        self.history.append({
            "event": "start",
            "task": task_name,
        })

    def submit_task(self, task_name: str) -> None:
        self._validate_known_task(task_name)

        if self.task_statuses[task_name] != TaskStatus.IN_PROGRESS:
            raise ValueError(f"Task '{task_name}' is not in progress.")

        self.task_statuses[task_name] = TaskStatus.UNDER_REVIEW
        self.history.append({
            "event": "submit",
            "task": task_name,
        })

    def accept_task(self, task_name: str) -> None:
        self._validate_known_task(task_name)

        if self.task_statuses[task_name] != TaskStatus.UNDER_REVIEW:
            raise ValueError(f"Task '{task_name}' is not under review.")

        self.task_statuses[task_name] = TaskStatus.ACCEPTED
        self.history.append({
            "event": "accept",
            "task": task_name,
        })

    def reject_task(self, task_name: str) -> str:
        self._validate_known_task(task_name)

        if self.task_statuses[task_name] != TaskStatus.UNDER_REVIEW:
            raise ValueError(f"Task '{task_name}' is not under review.")

        self.task_statuses[task_name] = TaskStatus.REJECTED

        restart_task = self._sample_restart_task(task_name)

        self.reset_from_task(restart_task)

        self.history.append({
            "event": "reject",
            "task": task_name,
            "restart_task": restart_task,
        })

        return restart_task

    def reset_from_task(self, task_name: str) -> None:
        """
        Reset task_name and all downstream tasks to NOT_STARTED.

        This is useful after rejection/rework.
        """
        self._validate_known_task(task_name)

        downstream_tasks = self.downstream_tasks(task_name)
        tasks_to_reset = downstream_tasks | {task_name}

        for task in tasks_to_reset:
            if self.task_statuses[task] != TaskStatus.ACCEPTED:
                self.task_statuses[task] = TaskStatus.NOT_STARTED
            else:
                self.task_statuses[task] = TaskStatus.NOT_STARTED

        self.history.append({
            "event": "reset",
            "from_task": task_name,
            "reset_tasks": sorted(tasks_to_reset),
        })

    def downstream_tasks(self, task_name: str) -> set[str]:
        """
        Return all tasks that depend directly or indirectly on task_name.
        """
        self._validate_known_task(task_name)

        downstream = set()
        changed = True

        while changed:
            changed = False

            for candidate_name, candidate_task in self.workflow.tasks.items():
                if candidate_name in downstream:
                    continue

                prerequisites = {
                    prerequisite
                    for prerequisite_set in candidate_task.prerequisite_sets
                    for prerequisite in prerequisite_set
                }

                if task_name in prerequisites or prerequisites & downstream:
                    downstream.add(candidate_name)
                    changed = True

        return downstream

    def possible_execution_plans(
        self,
        start_tasks: set[str] | None = None,
        target_tasks: set[str] | None = None,
    ) -> list[list[set[str]]]:
        """
        Planning should probably move here later.

        For now, this only returns a simple staged plan based on current
        available tasks.
        """
        if start_tasks is None:
            start_tasks = self.available_tasks

        if target_tasks is None:
            target_tasks = self.workflow.last

        return [
            [start_tasks]
        ]

    def _sample_restart_task(self, failed_task: str) -> str:
        if failed_task not in self.failure_policy:
            return failed_task

        transitions = self.failure_policy[failed_task]

        restart_tasks = list(transitions.keys())
        probabilities = list(transitions.values())

        return random.choices(
            restart_tasks,
            weights=probabilities,
            k=1,
        )[0]

    def _validate_known_task(self, task_name: str) -> None:
        if task_name not in self.workflow.tasks:
            raise ValueError(f"Unknown task: {task_name}")

    @property
    def summary(self) -> dict:
        return {
            "Accepted tasks": self.accepted_tasks,
            "Incomplete tasks": self.incomplete_tasks,
            "Available tasks": self.available_tasks,
            "Statuses": self.task_statuses,
        }