from trustdynamics.work.workflow.task import Task


class Workflow:

    def __init__(self):
        self.tasks: dict[str, Task] = {}

    def add_task(self, task: Task) -> None:
        if task.name in self.tasks:
            raise ValueError(f"Task name '{task.name}' already exists.")
        self.tasks[task.name] = task

    @property
    def task_names(self) -> set[str]:
        return set(self.tasks)

    @property
    def first(self) -> set[str]:
        return {
            task.name
            for task in self.tasks.values()
            if not task.prerequisite_sets
        }

    @property
    def last(self) -> set[str]:
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
    def summary(self) -> dict:
        return {
            "All tasks": self.task_names,
            "First tasks": self.first,
            "Last tasks": self.last,
        }


    def show(self) -> None:
        import matplotlib.pyplot as plt
        import networkx as nx

        graph = nx.DiGraph()

        # Add task nodes
        for task_name in self.task_names:
            graph.add_node(
                task_name,
                kind="task",
                label=task_name,
            )

        # Add explicit AND nodes.
        # OR is represented by multiple incoming alternatives to the same task.
        for task in self.tasks.values():
            for i, prerequisite_set in enumerate(task.prerequisite_sets):
                if len(prerequisite_set) == 1:
                    graph.add_edge(prerequisite_set[0], task.name)
                    continue

                and_node = f"{task.name}__AND_{i + 1}"

                graph.add_node(
                    and_node,
                    kind="and",
                    label="AND",
                )

                for prerequisite in prerequisite_set:
                    graph.add_edge(prerequisite, and_node)

                graph.add_edge(and_node, task.name)

        # Compute task layers: first tasks on the left, last tasks on the right
        task_layers: dict[str, int] = {}

        def get_task_layer(task_name: str) -> int:
            if task_name in task_layers:
                return task_layers[task_name]

            task = self.tasks[task_name]

            if not task.prerequisite_sets:
                task_layers[task_name] = 0
                return 0

            prerequisites = {
                prerequisite
                for prerequisite_set in task.prerequisite_sets
                for prerequisite in prerequisite_set
            }

            task_layers[task_name] = 1 + max(
                get_task_layer(prerequisite)
                for prerequisite in prerequisites
            )

            return task_layers[task_name]

        for task_name in self.task_names:
            get_task_layer(task_name)

        # Assign graph layers
        node_layers = {}

        for node, data in graph.nodes(data=True):
            if data["kind"] == "task":
                node_layers[node] = task_layers[node] * 2
            else:
                successor = next(graph.successors(node))
                node_layers[node] = task_layers[successor] * 2 - 1

        # Deterministic manual positions
        nodes_by_layer: dict[int, list[str]] = {}

        for node, layer in node_layers.items():
            nodes_by_layer.setdefault(layer, []).append(node)

        pos = {}

        for layer, nodes in nodes_by_layer.items():
            nodes = sorted(nodes)

            y_start = (len(nodes) - 1) / 2

            for i, node in enumerate(nodes):
                pos[node] = (layer, y_start - i)

        labels = {
            node: data["label"]
            for node, data in graph.nodes(data=True)
        }

        task_nodes = [
            node
            for node, data in graph.nodes(data=True)
            if data["kind"] == "task"
        ]

        and_nodes = [
            node
            for node, data in graph.nodes(data=True)
            if data["kind"] == "and"
        ]

        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=task_nodes,
            node_size=2200,
            node_shape="o",
        )

        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=and_nodes,
            node_size=900,
            node_shape="s",
            node_color="orange",
        )

        nx.draw_networkx_labels(
            graph,
            pos,
            labels=labels,
        )

        nx.draw_networkx_edges(
            graph,
            pos,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=20,
            connectionstyle="arc3,rad=0.05",
        )

        plt.axis("off")
        plt.tight_layout()
        plt.show()
    

if __name__ == "__main__":

    from pprint import pprint
    from trustdynamics.work.workflow.examples.example_2 import workflow

    pprint(workflow.summary)
    workflow.show()