import numpy as np
import networkx as nx

from trustdynamics.organization import Organization
from trustdynamics.utils import (
    bounded_random_with_exact_mean,
    map_to_range,
    normalize_01,
)


class Model:

    def __init__(
        self,
        org: Organization,
        technology_success_rate: float = 1.0,
        tech_successful_delta: float = 0.05,
        tech_failure_delta: float = -0.15,
        average_initial_opinion: float = 0.0,
        seed: int | None = None,
    ):
        self.rng = np.random.default_rng(seed)
        self.org = org

        if technology_success_rate < 0.0 or technology_success_rate > 1.0:
            raise ValueError("technology_success_rate must be between 0.0 and 1.0")
        self.technology_success_rate = technology_success_rate
        self.tech_successful_delta = tech_successful_delta
        self.tech_failure_delta = tech_failure_delta

        if average_initial_opinion < -1.0 or average_initial_opinion > 1.0:
            raise ValueError("average_initial_opinion must be between -1.0 and 1.0")
        self.initialize_agents_opinion(average_initial_opinion)
        self.initialize_agents_trust()
        self.initialize_teams_trust()

    def initialize_agents_opinion(self, average_initial_opinion: float):
        """
        Assign initial agents opinions.
        """
        agent_ids = list(self.org.all_agent_ids)
        n = len(agent_ids)
        if n == 0:
            return
        opinions = bounded_random_with_exact_mean(
            n=n,
            target_mean=average_initial_opinion,
            seed=self.rng, # pass the Generator to keep reproducibility tied to Model
            min_value=-1.0,
            max_value=1.0,
        )
        for agent_id, opinion in zip(agent_ids, opinions):
            self.org.set_agent_opinion(agent_id, float(opinion))

    def initialize_agents_trust(self):
        """
        Assign initial trust values between agents from degree centrality.

        Strategy:
        - Compute node degree centrality on org.G_agents
        - Normalize to [0,1]
        - For each directed edge u->v, set trust based on centrality(u)
        - Writes via org.set_agent_trust(u, v, trust)
        """
        trust_min = 0.00
        trust_max = 1.00

        G = self.org.G_agents
        if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
            return

        dc_in = nx.in_degree_centrality(G)
        for u, v in G.edges():
            trust = map_to_range(dc_in.get(v, 0.0), trust_min, trust_max)
            self.org.set_agent_trust(u, v, trust)

    def initialize_teams_trust(self):
        """
        Assign initial trust values between teams from degree centrality.

        Strategy:
        - Compute node degree centrality on org.G_teams
        - Normalize to [0,1]
        - For each directed edge u->v, set trust based on centrality(u)
        - Writes via org.set_team_trust(u, v, trust)
        """
        trust_min = 0.00
        trust_max = 1.00

        G = self.org.G_teams
        if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
            return

        dc_in = nx.in_degree_centrality(G)
        for u, v in G.edges():
            trust = map_to_range(dc_in.get(v, 0.0), trust_min, trust_max)
            self.org.set_team_trust(u, v, trust)

    def update(self):
        """
        1. Update the agents opinions based on communication
        2. Update the agents opinions
        3. Update the teams opinions based on communication
        """
        self.agents_use_technology()
        self.agents_communicate_within_teams()
        self.teams_communicate_with_teams()

    def agents_communicate_within_teams(self):
        """
        Agents communicate with agents inside their teams and agents from connected teams.
        Shared opinions are aggregated as team opinion.
        """
        for team_id in self.org.all_team_ids:
            # Calculate aggregate opinion for the teams
            agents = self.org.agents_from_team(team_id)
            # Add agents from other teams that are connected to this team
            #### 
            # Update opinions of agents based on group opinion
            team_opinion = 0.0 ####
            self.org.set_team_opinion(team_id, team_opinion)
            # Update trust between agents based on aggregated opinions

    def agents_use_technology(self):
        agents = self.org.all_agent_ids
        for agent_id in agents:
            current_opinion = self.org.get_agent_opinion(agent_id)
            tech_successful: bool = self.rng.random() < self.technology_success_rate
            if tech_successful:
                new_opinion = min(current_opinion + self.tech_successful_delta, 1.0)
            else:
                new_opinion = max(current_opinion + self.tech_failure_delta, -1.0)
            self.org.set_agent_opinion(agent_id, new_opinion)

    def teams_communicate_with_teams(self):
        """
        Aggregate team opinions to form organization opinion.
        """
        team_ids = self.org.all_team_ids
        for team_id in team_ids:
            ####
            # Update opinions of agents based on group opinion
            team_opinion = 0.0 ####
            self.org.set_team_opinion(team_id, team_opinion)
        







if __name__ == "__main__":
    
    from trustdynamics.organization.samples import organization_0 as org

    model = Model(
        org=org,
        technology_success_rate=0.9,
        average_initial_opinion=0.0,
        seed=42
    )
    print(model.org.get_agent_trust("Agent 5", "Agent 2"))
    print(model.org.get_agent_trust("Agent 2", "Agent 5"))

    print(model.org.get_agent_trust("Agent 4", "Agent 2"))
    print(model.org.get_agent_trust("Agent 2", "Agent 4"))

    print(model.org.get_agent_trust("Agent 4", "Agent 3"))
    print(model.org.get_agent_trust("Agent 3", "Agent 4"))