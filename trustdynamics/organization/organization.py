import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class Organization:

    def __init__(
            self,
            name: str = "Organization",
            ceo_name: str = "CEO",
        ):
        self.G = nx.Graph()
        self.name = name
        self.add_agent(name=ceo_name, parent=None, department="CEO")

    def new_id(self) -> int:
        """
        Generates a new unique ID for an agent in the organization.
        """
        if len(self.G.nodes) == 0:
            return 0
        else:
            return max(self.G.nodes) + 1
        
    def add_agent(
            self,
            department: str,
            parent: int | str,
            name: str | None = None,
        ) -> int:
        """
        Adds a new agent to the organization.
        """
        new_id = self.new_id()
        if not self.is_name_unique(name=name):
            raise ValueError(f"Agent name '{name}' is not unique.")
        self.G.add_node(new_id, name=name, department=department)
        if parent is not None:
            parent_id = self.get_agent_id(parent) if isinstance(parent, str) else parent
            self.G.add_edge(parent_id, new_id)
        return new_id
    
    def get_agent_id(self, name: str) -> int | None:
        """
        Retrieves the ID of an agent by their name.
        """
        for node_id, data in self.G.nodes(data=True):
            if data.get('name') == name:
                return node_id
        return None
    
    def is_name_unique(self, name: str | None) -> bool:
        """
        Checks if the given name is unique among the agents in the organization.
        """
        if name is None:
            return True
        for _, data in self.G.nodes(data=True):
            if data.get('name') == name:
                return False
        return True
    
    def draw(self):
        """
        Visualizes the organizational network.
        """
        pos = nx.spring_layout(self.G, seed=42)  # stable layout

        # Node labels: Name (Department), use node ID if name is None
        labels = {}
        for node, data in self.G.nodes(data=True):
            label_name = data.get('name') if data.get('name') is not None else str(node)
            labels[node] = f"{label_name}\n({data.get('department')})"

        # Color CEO (node 0) differently
        node_colors = []
        for node in self.G.nodes:
            if node == 0:
                node_colors.append("gold")  # CEO
            else:
                node_colors.append("lightblue")

        # Scale node size based on organization size to reduce overlap
        n_nodes = self.G.number_of_nodes()
        base_size = 2500
        min_size = 300
        node_size = max(min_size, int(base_size / np.sqrt(n_nodes)))

        nx.draw(
            self.G,
            pos,
            labels=labels,
            node_color=node_colors,
            node_size=node_size,
            font_size=9,
            edge_color="gray"
        )

        plt.title(self.name)
        plt.show()

    def serialize(self) -> dict:
        """
        Serialize object into a dictionary.
        """
        return {
            "name": self.name,
            "agents": [
                {
                    "id": node_id,
                    "name": data.get("name"),
                    "department": data.get("department"),
                    "parent": next(iter(self.G.neighbors(node_id)), None)  # Get the parent (first neighbor)
                }
                for node_id, data in self.G.nodes(data=True)
            ]
        }
    
    def deserialize(self, data: dict):
        """
        Deserialize object from a dictionary.
        """
        self.name = data.get("name", "Organization")
        self.G.clear()
        for agent in data.get("agents", []):
            node_id = agent["id"]
            name = agent.get("name")
            department = agent.get("department")
            parent = agent.get("parent")
            self.G.add_node(node_id, name=name, department=department)
            if parent is not None:
                self.G.add_edge(parent, node_id)

    @property
    def depth(self) -> int:
        """
        Return the maximum number of edges from the CEO (id=0) to any node.
        """
        if 0 not in self.G:
            raise ValueError("Organization must contain a CEO with id=0.")
        if self.G.number_of_nodes() == 1:
            return 0
        lengths = nx.single_source_shortest_path_length(self.G, 0)
        return int(max(lengths.values(), default=0))
    
    @property
    def population(self) -> int:
        """
        Return the total number of agents.
        """
        return self.G.number_of_nodes()
    
    def agents(self, department: str | None = None) -> list:
        """
        Return a list of agents in the organization, optionally filtered by department.
        """
        agents_list = []
        for node_id, data in self.G.nodes(data=True):
            if department is None or data.get("department") == department:
                agents_list.append(node_id)
        return agents_list
    
    def departments(self) -> set:
        """
        Return the set of departments.
        """
        departments = set(data.get("department") for _, data in self.G.nodes(data=True))
        departments.discard("CEO")
        return departments

    def __eq__(self, value) -> bool:
        if self.serialize() == value.serialize():
            return True
        return False


if __name__ == "__main__":
    org = Organization(name="MyCompany", ceo_name="Alice")
    org.add_agent(name="Bob", parent="Alice", department="R&D")
    org.add_agent(name="Charlie", parent="Bob", department="R&D")
    org.add_agent(name="Chris", parent="Bob", department="R&D")
    org.add_agent(name="David", parent="Alice", department="Sales")
    org.draw()
