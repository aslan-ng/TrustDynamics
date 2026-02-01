import networkx as nx

from trustdynamics.utils import (
    bounded_random_with_exact_mean,
    map_to_range,
)


class Initialization:

    def initialize(self):
        """
        Initialize the model
        """
        self.initialize_agents_opinion()
        self.initialize_agents_trust()
        self.initialize_teams_trust()
        self.initialized = True

    def initialize_agents_opinion(self):
        """
        Assign initial agents opinions.
        """
        agent_ids = list(self.organization.all_agent_ids)
        n = len(agent_ids)
        if n == 0:
            return
        opinions = bounded_random_with_exact_mean(
            n=n,
            target_mean=self.average_initial_opinion,
            seed=self.rng, # pass the Generator to keep reproducibility tied to Model
            min_value=-1.0,
            max_value=1.0,
        )
        for agent_id, opinion in zip(agent_ids, opinions):
            self.organization.set_agent_opinion(agent_id, float(opinion))

    def initialize_agents_trust(self):
        """
        Assign initial trust values between agents from degree centrality.
        """
        trust_min = 0.01
        trust_max = 0.99

        G = self.organization.G_agents
        if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
            return

        dc_in = nx.in_degree_centrality(G)
        for u, v in G.edges():
            trust = map_to_range(dc_in.get(v, 0.0), trust_min, trust_max)
            self.organization.set_agent_trust(u, v, trust)

    def initialize_teams_trust(self):
        """
        Assign initial trust values between teams from degree centrality.
        """
        trust_min = 0.01
        trust_max = 0.99

        G = self.organization.G_teams
        if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
            return

        dc_in = nx.in_degree_centrality(G)
        for u, v in G.edges():
            trust = map_to_range(dc_in.get(v, 0.0), trust_min, trust_max)
            self.organization.set_team_trust(u, v, trust)