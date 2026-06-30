from trustdynamics.work.project.project import Project
from trustdynamics.work.execution.assignment import AssignmentMixin
from trustdynamics.work.execution.review import ReviewMixin
from trustdynamics.work.execution.states import TaskState


class Execution(
    AssignmentMixin,
    ReviewMixin,
):

    def __init__(
            self,
            project: Project,
            workers: set[int],
        ):
        if project.selected_plan is None:
            raise ValueError("Project must have a selected plan.")

        self.project = project
        self.plan = project.selected_plan
        self.workers = workers

        self.task_states = {
            task_name: TaskState.NOT_READY
            for task_name in self.project.workflow.tasks
        }
        self.task_workers: dict[str, set[int] | None] = {
            task_name: None
            for task_name in self.project.workflow.tasks
        }
        self.task_reviews = {
            task_name: {}
            for task_name in self.project.workflow.tasks
        }

        self._update_ready_tasks()
        self._validate_review_feasibility()

    def _update_ready_tasks(self) -> None:
        completed_tasks = {
            task_name
            for task_name, state in self.task_states.items()
            if state == TaskState.COMPLETED
        }

        for task_name, task in self.project.workflow.tasks.items():
            if self.task_states[task_name] != TaskState.NOT_READY:
                continue

            if not task.prerequisite_sets:
                self.task_states[task_name] = TaskState.READY
                continue

            for prerequisite_set in task.prerequisite_sets:
                if set(prerequisite_set).issubset(completed_tasks):
                    self.task_states[task_name] = TaskState.READY
                    break

    @property
    def ready_tasks(self):
        return {task_name for task_name in self.task_states if self.task_states[task_name] == TaskState.READY}

    def start_task(self, task_name: str, task_workers: set[int]):
        if self.task_states[task_name] != TaskState.READY:
            raise ValueError(f"Task '{task_name}' is not ready.")
        self.task_states[task_name] = TaskState.IN_PROGRESS
        self.task_workers[task_name] = task_workers

    def finish_task(self, task_name: str, success: bool):
        if self.task_states[task_name] != TaskState.IN_PROGRESS:
            raise ValueError(f"Task '{task_name}' is not in progress.")

        if not success:
            self.task_states[task_name] = TaskState.FAILED
            self.task_workers[task_name] = None
            return

        if self.has_review_policy(task_name):
            self.task_states[task_name] = TaskState.UNDER_REVIEW
        else:
            self.task_states[task_name] = TaskState.COMPLETED
            self.task_workers[task_name] = None
            self._update_ready_tasks()



if __name__ == "__main__":
    from pprint import pprint

    from trustdynamics.work.project.examples.example_1 import project

    possible_plans = project.possible_plans()
    plan = possible_plans[0]
    project.choose_plan(plan)

    workers = {1, 2, 3, 4, 5}

    execution = Execution(
        project=project,
        workers=workers
    )
    #print(execution.task_states)
    
    # Choose tasks
    ready_tasks = execution.ready_tasks
    #print(ready_tasks)
    task_name = list(ready_tasks)[0]
    #print(task_name)

    # Choose worker for the task
    eligible_workers = execution.eligible_workers(task_name=task_name)
    #print(eligible_workers)

    # Start task
    execution.start_task(task_name=task_name, task_workers={1})
    #print(execution.task_states)

    # Finish task
    execution.finish_task(task_name=task_name, success=True)
    #print(execution.task_states)

    # Does it need review?
    needs_review = execution.needs_review(task_name=task_name)
    #print(needs_review)

    if needs_review:
        pass
        # Who can review it?
        eligible_reviewers = execution.eligible_reviewers(task_name=task_name)
        #print(eligible_reviewers) # better validation

        # Submitting reviews
        execution.submit_review(
            task_name=task_name,
            reviewer=list(eligible_reviewers)[0],
            score=True
        )

        # Finish reviews
        execution.finish_review(task_name=task_name)
        print(execution.task_states)