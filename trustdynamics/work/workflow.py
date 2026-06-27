from itertools import product

from task import Task


class Workflow:

    def __init__(
        self,
    ):
        self.tasks: dict[str, Task] = {}

    @property
    def task_names(self) -> set[str]:
        return set(self.tasks)

    def add_task(self, task: Task):
        """
        Add a new task.
        """
        if task.name in self.tasks:
            raise ValueError(f"Task name '{task.name}' already exists.")
        self.tasks[task.name] = task
    
    @property
    def completed_tasks(self) -> set[str]:
        """
        A list of completed tasks names.
        """
        return {
            task.name
            for task in self.tasks.values()
            if task.done
        }
    
    @property
    def incomplete_tasks(self) -> set[str]:
        """
        A list of all incomplete tasks names.
        """
        return {
            task.name
            for task in self.tasks.values()
            if not task.done
        }
    
    @property
    def first(self) -> set[str]:
        """
        First tasks names in the workflow.
        """
        return {
            task.name
            for task in self.tasks.values()
            if not task.prerequisite_sets
        }
    
    @property
    def last(self) -> set[str]:
        """
        Last tasks names in the workflow.
        """
        prerequisite_tasks = set()

        for task in self.tasks.values():
            for prerequisite_set in task.prerequisite_sets:
                prerequisite_tasks.update(prerequisite_set)

        return {
            task.name
            for task in self.tasks.values()
            if task.name not in prerequisite_tasks
        }
    
    @property
    def available_tasks(self) -> set[str]:
        """
        Tasks names that are available.
        """
        result = set()

        completed = self.completed_tasks
        incomplete_tasks = self.incomplete_tasks

        for task_name in incomplete_tasks:
            task = self.tasks[task_name]

            if not task.prerequisite_sets:
                result.add(task.name)
                continue

            for prerequisite_set in task.prerequisite_sets:
                if all(
                    prerequisite in completed
                    for prerequisite in prerequisite_set
                ):
                    result.add(task.name)
                    break

        return result
    
    def possible_execution_plans(
        self,
        start_tasks: set[str] | None = None,
        target_tasks: set[str] | None = None,
    ) -> list[list[set[str]]]:
        """
        Return possible staged execution plans.

        Return type:
            list[list[set[str]]]

        Meaning:
            outer list = alternative plans
            inner list = sequential stages
            set[str] = tasks that can be done in parallel in that stage
        """

        completed = self.completed_tasks

        if start_tasks is None:
            start_tasks = self.available_tasks

        if target_tasks is None:
            target_tasks = self.last

        def clean_plan(plan: list[set[str]]) -> list[set[str]]:
            """
            Remove completed tasks and duplicate tasks from later stages.
            """
            seen = set(completed)
            cleaned = []

            for stage in plan:
                new_stage = {
                    task_name
                    for task_name in stage
                    if task_name not in seen
                }

                if new_stage:
                    cleaned.append(new_stage)
                    seen.update(new_stage)

            return cleaned

        def merge_parallel_plans(plans: tuple[list[set[str]], ...]) -> list[set[str]]:
            """
            Merge multiple prerequisite plans into staged parallel execution.
            """
            if not plans:
                return []

            max_length = max(len(plan) for plan in plans)
            merged = []

            for i in range(max_length):
                stage = set()

                for plan in plans:
                    if i < len(plan):
                        stage.update(plan[i])

                if stage:
                    merged.append(stage)

            return clean_plan(merged)

        def plans_to_complete(task_name: str) -> list[list[set[str]]]:
            """
            Return all possible staged plans needed to complete one task.
            """
            if task_name not in self.tasks:
                raise ValueError(f"Unknown task: {task_name}")

            if task_name in completed:
                return [[]]

            task = self.tasks[task_name]

            if task_name in start_tasks:
                return [[{task_name}]]

            if not task.prerequisite_sets:
                return [[{task_name}]]

            result = []

            for prerequisite_set in task.prerequisite_sets:
                prerequisite_plans_options = [
                    plans_to_complete(prerequisite)
                    for prerequisite in prerequisite_set
                    if prerequisite not in completed
                ]

                if not prerequisite_plans_options:
                    result.append([{task_name}])
                    continue

                for prerequisite_plan_combination in product(*prerequisite_plans_options):
                    merged_prerequisite_plan = merge_parallel_plans(
                        prerequisite_plan_combination
                    )

                    full_plan = merged_prerequisite_plan + [{task_name}]
                    result.append(clean_plan(full_plan))

            return result

        target_plan_options = [
            plans_to_complete(target_task)
            for target_task in target_tasks
        ]

        all_plans = []

        for target_plan_combination in product(*target_plan_options):
            merged_plan = merge_parallel_plans(target_plan_combination)
            all_plans.append(clean_plan(merged_plan))

        return all_plans
    
    @property
    def summary(self):
        """
        Stat of the workflow.
        """
        return {
            "All tasks": self.task_names,
            "First tasks": self.first,
            "Last tasks": self.last,
            "Completed tasks": self.completed_tasks,
            "Incomplete tasks": self.incomplete_tasks,
        }

                

if __name__ == "__main__":

    from pprint import pprint
    from example_2 import workflow

    print("Step 0:")
    pprint(workflow.possible_execution_plans())

