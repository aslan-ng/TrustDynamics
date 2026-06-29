from trustdynamics.work.workflow.workflow import Workflow
from trustdynamics.work.policy.review.review import ReviewPolicy
from trustdynamics.work.policy.failure.failure import FailurePolicy


class Project:

    def __init__(
        self,
        workflow: Workflow,
        failure_policy: dict[str, FailurePolicy] | None = None,
        review_policy: dict[str, ReviewPolicy] | None = None,
    ):
        self.workflow = workflow

        self.failure_policy = failure_policy or {}
        self.review_policy = review_policy or {}
        self._validate_policies()

        self.selected_plan: list[set[str]] | None = None

    def _validate_policies(self) -> None:
        for task_name in self.review_policy:
            if task_name not in self.workflow.tasks:
                raise ValueError(
                    f"Review policy refers to unknown task: {task_name}"
                )
        for failed_task, policy in self.failure_policy.items():
            if failed_task not in self.workflow.tasks:
                raise ValueError(
                    f"Failure policy refers to unknown task: {failed_task}"
                )
            for return_task in policy.transitions:
                if return_task not in self.workflow.tasks:
                    raise ValueError(
                        f"Failure policy for task '{failed_task}' "
                        f"returns to unknown task: {return_task}"
                    )
    
    def _plans_to_complete_targets(
        self,
        target_tasks: set[str],
    ) -> list[list[set[str]]]:
        from itertools import product

        target_plan_options = [
            self._plans_to_complete_task(task_name)
            for task_name in target_tasks
        ]

        all_plans = []

        for plan_combination in product(*target_plan_options):
            all_plans.append(
                self._merge_parallel_plans(plan_combination)
            )

        return all_plans

    def _plans_to_complete_task(
        self,
        task_name: str,
    ) -> list[list[set[str]]]:
        from itertools import product

        if task_name not in self.workflow.tasks:
            raise ValueError(f"Unknown task: {task_name}")

        task = self.workflow.tasks[task_name]

        if not task.prerequisite_sets:
            return [[{task_name}]]

        result = []

        for prerequisite_set in task.prerequisite_sets:
            prerequisite_plan_options = [
                self._plans_to_complete_task(prerequisite)
                for prerequisite in prerequisite_set
            ]

            for prerequisite_plan_combination in product(
                *prerequisite_plan_options
            ):
                prerequisite_plan = self._merge_parallel_plans(
                    prerequisite_plan_combination
                )
                result.append(prerequisite_plan + [{task_name}])

        return result

    def _merge_parallel_plans(
        self,
        plans: tuple[list[set[str]], ...],
    ) -> list[set[str]]:
        if not plans:
            return []

        max_length = max(len(plan) for plan in plans)
        merged = []

        seen = set()

        for i in range(max_length):
            stage = set()

            for plan in plans:
                if i < len(plan):
                    stage.update(plan[i])

            stage -= seen

            if stage:
                merged.append(stage)
                seen.update(stage)

        return merged
    
    def possible_plans(self) -> list[list[set[str]]]:
        return self._plans_to_complete_targets(
            target_tasks=self.workflow.last,
        )

    def choose_plan(self, plan: list[set[str]]) -> None:
        possible_plans = self.possible_plans()

        if plan not in possible_plans:
            raise ValueError(
                "Selected plan is not one of the possible plans."
            )

        self.selected_plan = plan

    @property
    def summary(self) -> dict:
        return {
            "Tasks": self.workflow.task_names,
            "First tasks": self.workflow.first,
            "Last tasks": self.workflow.last,
            "Review policy tasks": set(self.review_policy),
            "Failure policy tasks": set(self.failure_policy),
            "Selected plan": self.selected_plan,
        }



if __name__ == "__main__":
    from pprint import pprint

    from trustdynamics.work.workflow.examples.example_1 import workflow
    from trustdynamics.work.policy.failure.examples.example_1 import failure_policy
    from trustdynamics.work.policy.review.examples.example_1 import review_policy

    project = Project(
        workflow=workflow,
        failure_policy=failure_policy,
        review_policy=review_policy,
    )
    possible_plans = project.possible_plans()
    plan = possible_plans[0]
    project.choose_plan(plan)

    pprint(project.summary, sort_dicts=False)