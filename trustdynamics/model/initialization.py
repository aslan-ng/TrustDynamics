import networkx as nx

from trustdynamics.utils import (
    bounded_random_with_exact_mean,
    map_to_range,
)


class Initialization:
    """
    Mixin class responsible for initializing model state.

    Initialization establishes:
    - initial agent opinions
    - initial trust networks (agents and teams)

    These choices strongly influence early transient dynamics,
    but do not constrain long-term behavior.
    """

    def initialize(self):
        """
        Initialize the model once before the first update cycle.

        This method is idempotent by design and guarded by the
        `self.initialized` flag in the update loop.
        """
        self.initialize_agents_opinion()
        self.initialize_agents_trust()
        self.initialize_teams_trust()
        self.initialized = True

    def initialize_agents_opinion(self):
        """
        Assign initial opinions to agents.

        Strategy:
        - Keep already agents opinions if they were set initially
        - Sample opinions in [-1, 1]
        - Enforce an *exact* global mean equal to `average_initial_opinion`
        - Preserve reproducibility by using the model RNG

        This allows controlled experiments on initial ideological bias
        while maintaining heterogeneity across agents.
        """
        agent_ids = list(self.organization.all_agent_ids)
        n = len(agent_ids)
        if n == 0:
            return
        
        existing_opinions = []
        for agent_id in agent_ids:
            existing_opinion = self.organization.get_agent_opinion(agent_id)
            if existing_opinion is not None:
                existing_opinions.append(existing_opinion)

        opinions = bounded_random_with_exact_mean(
            n_total=n,
            target_mean=self.agents_average_initial_opinion,
            fixed_values=existing_opinions,
            seed=self.rng, # pass the Generator to keep reproducibility tied to Model
            min_value=self.agents_initial_opinion_min,
            max_value=self.agents_initial_opinion_max,
        )
        for agent_id, opinion in zip(agent_ids, opinions):
            self.organization.set_agent_opinion(agent_id, float(opinion))

    def initialize_agents_trust(self):
        """
        Assign initial trust values between agents.

        Strategy:
        - Use in-degree centrality as a proxy for perceived importance
        - Trust flows *toward* structurally central agents
        - Map centrality values into (0, 1) to avoid degenerate dynamics

        Interpretation:
        Agents who are more "listened to" initially receive higher trust.
        """
        trust_min = self.agents_initial_trust_min
        trust_max = self.agents_initial_trust_max

        G = self.organization.G_agents
        if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
            return

        dc_in = nx.in_degree_centrality(G)
        for u, v in G.edges():
            trust = map_to_range(dc_in.get(v, 0.0), trust_min, trust_max)
            self.organization.set_agent_trust(u, v, trust)

    def initialize_teams_trust(self):
        """
        Assign initial trust values between teams.

        Mirrors agent-level initialization, but at the team layer.

        Interpretation:
        Teams that are structurally central in the organization
        are initially trusted more by other teams.
        """
        trust_min = self.teams_initial_trust_min
        trust_max = self.teams_initial_trust_max

        G = self.organization.G_teams
        if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
            return

        dc_in = nx.in_degree_centrality(G)
        for u, v in G.edges():
            trust = map_to_range(dc_in.get(v, 0.0), trust_min, trust_max)
            self.organization.set_team_trust(u, v, trust)