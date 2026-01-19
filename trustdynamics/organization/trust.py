import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

from trustdynamics.organization.organization import Organization


class OrganizationTrust:

    def __init__(
        self,
        organization: Organization,
        seed: int | None = None,
    ):
        self.org = organization
        self.G = nx.DiGraph()
        self.rng = np.random.default_rng(seed)
        self.create()

    def random(self, low: float = 0.0, high: float = 1.0):
        return self.rng.uniform(low, high)
    
    def get_trust(self, from_agent: int, to_agent: int) -> float | None:
        if self.G.has_edge(from_agent, to_agent):
            return self.G[from_agent][to_agent].get('trust', None)
        else:
            return None
    
    def create(self):
        agents = self.org.agents()
        org_depth = self.org.depth
        trust_steps = 1 / org_depth

        # Create nodes
        for node_id in agents:
            self.G.add_node(node_id)
        # Create edges with None trust values
        for node_id in agents:
            for other_id in agents:
                self.G.add_edge(node_id, other_id, trust=0.0)
        
        # CEO
        ceo_id = 0
        ceo_self_trust = self.random(max(1-trust_steps, 0), 1)
        self.G.add_edge(ceo_id, ceo_id, trust=ceo_self_trust) # CEO's self-trust

        # Rest of the agents
        for depth in range(1, org_depth + 1):
            agents_at_depth = self.org.agents_from_level(depth)
            for agent_id in agents_at_depth:
                parent_id = self.org.parent(agent_id)
                parent_self_trust = self.get_trust(parent_id, parent_id)

                agent_self_trust = self.random(max(parent_self_trust-trust_steps, 0), parent_self_trust)
                self.G.add_edge(agent_id, agent_id, trust=agent_self_trust) # Agent's self-trust

                parent_to_agent_trust = self.random(max(parent_self_trust-trust_steps, 0), parent_self_trust)
                self.G.add_edge(parent_id, agent_id, trust=parent_to_agent_trust) # Parent to agent trust

                agent_to_parent_trust = self.random(max(parent_self_trust-trust_steps, 0), parent_self_trust)
                self.G.add_edge(agent_id, parent_id, trust=agent_to_parent_trust) # Agent to parent trust
    
    def adjacency_dataframe(
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
            order = sorted(self.G.nodes())

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

    def draw(
        self,
        show_none: bool = False,
        min_trust: float | None = None,
        seed: int = 42,
        node_size: int | None = None,
        font_size: int = 9,
        edge_font_size: int = 8,
    ):
        """
        Draw the communication/trust graph and show trust values on edges
        with 2 decimal precision.

        Parameters
        ----------
        show_none : bool
            If False, edges with trust=None are not drawn/labeled.
        min_trust : float | None
            If set, only edges with trust >= min_trust are drawn/labeled.
        """
        n = self.G.number_of_nodes()
        if node_size is None:
            node_size = max(300, int(2500 / np.sqrt(max(n, 1))))

        pos = nx.spring_layout(self.G, seed=seed)

        # Build an edge list to draw (optionally filtered)
        edges_to_draw = []
        edge_labels = {}

        for u, v, data in self.G.edges(data=True):
            t = data.get("trust", None)

            if t is None and not show_none:
                continue

            if t is not None and min_trust is not None and t < min_trust:
                continue

            edges_to_draw.append((u, v))

            if t is None:
                edge_labels[(u, v)] = "None"
            else:
                edge_labels[(u, v)] = f"{float(t):.2f}"

        plt.figure(figsize=(10, 7))

        # Nodes
        nx.draw_networkx_nodes(self.G, pos, node_size=node_size)

        # Node labels (IDs)
        nx.draw_networkx_labels(self.G, pos, font_size=font_size)

        # Edges
        nx.draw_networkx_edges(
            self.G,
            pos,
            edgelist=edges_to_draw,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=12,
            width=1.0,
            alpha=0.6,
        )

        # Edge labels (trust values)
        nx.draw_networkx_edge_labels(
            self.G,
            pos,
            edge_labels=edge_labels,
            font_size=edge_font_size,
            rotate=False,
            label_pos=0.5,
        )

        plt.title("Trust Network")
        plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    from trustdynamics.organization.generate import generate_organization

    org = generate_organization(
        n_departments=3,
        n_people=6,
        max_depth=2,
        seed=42
    )
    comm = OrganizationTrust(org, seed=42)
    print(comm.adjacency_dataframe())
    org.draw()