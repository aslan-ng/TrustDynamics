"""
A workflow with a set of simple tasks. Multiple plans possible with AND and OR relationships.
"""

from trustdynamics.work.workflow.task import Task
from trustdynamics.work.workflow.workflow import Workflow


t1 = Task(
    name="Task 1",
)
t2 = Task(
    name="Task 2",
    prerequisite_sets=[["Task 1"]],
)
t3 = Task(
    name="Task 3",
    prerequisite_sets=[["Task 1"]],
)
t4 = Task(
    name="Task 4",
    prerequisite_sets=[["Task 1"]],
)
t5 = Task(
    name="Task 5",
    prerequisite_sets=[
        ["Task 2", "Task 3"],
        ["Task 4"],
    ]
)

workflow = Workflow()
workflow.add_task(task=t1)
workflow.add_task(task=t2)
workflow.add_task(task=t3)
workflow.add_task(task=t4)
workflow.add_task(task=t5)


if __name__ == "__main__":
    workflow.show()