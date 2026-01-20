import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

from trustdynamics.organization.organization import Organization


class OrganizationalTrust:

    def __init__(
        self,
        organization: Organization,
        seed: int | None = None,
    ):
        self.rng = np.random.default_rng(seed)
        self.org = organization
        self.G_personal = nx.DiGraph()
        self.create_personal()
        self.G_departmental = nx.DiGraph()
        self.create_departmental()

    def random(self, low: float = 0.0, high: float = 1.0):
        return self.rng.uniform(low, high)
    
    def get_trust(self, from_agent: int, to_agent: int) -> float | None:
        if self.G_personal.has_edge(from_agent, to_agent):
            return self.G_personal[from_agent][to_agent].get('trust', None)
        else:
            return None
    
    def create_personal(self):
        agents = self.org.agents()
        org_depth = self.org.depth
        trust_steps = 1 / org_depth

        # Create nodes
        for node_id in agents:
            self.G_personal.add_node(node_id)
        # Create edges with None trust values
        for node_id in agents:
            for other_id in agents:
                self.G_personal.add_edge(node_id, other_id, trust=0.0)
        
        # CEO
        ceo_id = 0
        ceo_self_trust = self.random(max(1-trust_steps, 0), 1)
        self.G_personal.add_edge(ceo_id, ceo_id, trust=ceo_self_trust) # CEO's self-trust

        # Rest of the agents
        for depth in range(1, org_depth + 1):
            agents_at_depth = self.org.agents_from_level(depth)
            for agent_id in agents_at_depth:
                parent_id = self.org.parent(agent_id)
                parent_self_trust = self.get_trust(parent_id, parent_id)

                agent_self_trust = self.random(max(parent_self_trust-trust_steps, 0), parent_self_trust)
                self.G_personal.add_edge(agent_id, agent_id, trust=agent_self_trust) # Agent's self-trust

                parent_to_agent_trust = self.random(max(parent_self_trust-trust_steps, 0), parent_self_trust)
                self.G_personal.add_edge(parent_id, agent_id, trust=parent_to_agent_trust) # Parent to agent trust

                agent_to_parent_trust = self.random(max(parent_self_trust-trust_steps, 0), parent_self_trust)
                self.G_personal.add_edge(agent_id, parent_id, trust=agent_to_parent_trust) # Agent to parent trust
    
    def create_departmental(self):
        ceo_id = 0
        ceo_self_trust = self.get_trust(ceo_id, ceo_id)
        department_name = self.org.get_agent_department(ceo_id)
        self.G_departmental.add_node(ceo_id, department=department_name)
        self.G_departmental.add_edge(ceo_id, ceo_id, trust=ceo_self_trust)

        departments = self.org.children(ceo_id)
        for department_head in departments:
            department_name = self.org.get_agent_department(department_head)
            self.G_departmental.add_node(department_head, department=department_name)
            ceo_department_trust = self.get_trust(ceo_id, department_head)
            self.G_departmental.add_edge(
                ceo_id,
                department_head,
                trust=ceo_department_trust
            )
            department_ceo_trust = self.get_trust(department_head, ceo_id)
            self.G_departmental.add_edge(department_head, ceo_id, trust=department_ceo_trust)
            for other_department in departments:
                if other_department == department_head:
                    department_self_trust = self.get_trust(department_head, department_head)
                    self.G_departmental.add_edge(
                        department_head,
                        department_head,
                        trust=department_self_trust
                    )
                else:
                    inter_department_trust = self.random(0, 1)
                    self.G_departmental.add_edge(
                        department_head,
                        other_department,
                        trust=inter_department_trust
                    )

    def personal_adjacency_dataframe(
        self,
        none_value: float | None = 0.0,
        order: list[int] | None = None,
        dtype=float,
    ) -> pd.DataFrame:
        """
        Return the directed trust adjacency matrix as a pandas DataFrame.

        Rows: source agents
        Columns: target agents
        Entry (i, j): trust from agent i -> agent j

        Parameters
        ----------
        none_value : float | None
            Value to use when trust is None.
            - Use 0.0 for 'no trust'
            - Use np.nan to preserve missingness
        order : list[int] | None
            Order of agents. Defaults to sorted node IDs.
        dtype : type
            Data type for the DataFrame values.

        Returns
        -------
        pd.DataFrame
        """
        if order is None:
            order = sorted(self.G_personal.nodes())

        n = len(order)
        data = np.empty((n, n), dtype=dtype)

        for i, src in enumerate(order):
            for j, dst in enumerate(order):
                trust = self.get_trust(src, dst)
                if trust is None:
                    data[i, j] = none_value
                else:
                    data[i, j] = trust

        return pd.DataFrame(data, index=order, columns=order)
    
    def departmental_adjacency_dataframe(
        self,
        none_value: float | None = 0.0,
        dtype=float,
    ) -> pd.DataFrame:

        # Map node_id -> department name
        node_to_dept = {
            node: self.G_departmental.nodes[node]["department"]
            for node in self.G_departmental.nodes()
        }

        departments = list(node_to_dept.values())
        nodes = list(node_to_dept.keys())

        n = len(nodes)
        data = np.empty((n, n), dtype=dtype)

        for i, src in enumerate(nodes):
            for j, dst in enumerate(nodes):
                if self.G_departmental.has_edge(src, dst):
                    trust = self.G_departmental[src][dst].get("trust", None)
                else:
                    trust = None
                data[i, j] = none_value if trust is None else trust

        return pd.DataFrame(data, index=departments, columns=departments)


if __name__ == "__main__":

    from trustdynamics.organization.generate import generate_organization

    org = generate_organization(
        n_departments=3,
        n_people=6,
        max_depth=2,
        seed=42
    )
    org_trust = OrganizationalTrust(org, seed=42)
    print(org_trust.personal_adjacency_dataframe())
    print(org_trust.departmental_adjacency_dataframe())
    org.draw()
