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
        self.initialize_agents_influence()
        self.initialize_teams_influence()

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

    def initialize_agents_influence(self):
        """
        Assign initial influence values between agents from betweenness centrality.
        """
        """
        Assign initial influence values between agents from betweenness centrality.

        Strategy:
        - Compute node betweenness centrality on org.G_agents
        - Normalize to [0,1]
        - For each directed edge u->v, set influence based on centrality(u)
        - Writes via org.set_agent_influence(u, v, influence)
        """
        influence_min = 0.00
        influence_max = 1.00

        G = self.org.G_agents
        if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
            return

        # betweenness centrality on directed graph (normalized)
        bc = nx.betweenness_centrality(G, normalized=True)
        bc01 = normalize_01(bc)

        # Initialize every directed edge u -> v using SOURCE node centrality (u)
        for u, v in G.edges():
            infl = map_to_range(bc01.get(u, 0.0), influence_min, influence_max)
            self.org.set_agent_influence(u, v, infl)

    def initialize_teams_influence(self):
        """
        Assign initial influence values between teams from betweenness centrality.

        Strategy:
        - Compute node betweenness centrality on org.G_teams
        - Normalize to [0,1]
        - For each directed edge u->v, set influence based on centrality(u)
        - Writes via org.set_team_influence(u, v, influence)
        """
        influence_min = 0.00
        influence_max = 1.00

        G = self.org.G_teams
        if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
            return

        bc = nx.betweenness_centrality(G, normalized=True)
        bc01 = normalize_01(bc)

        for u, v in G.edges():
            infl = map_to_range(bc01.get(u, 0.0), influence_min, influence_max)
            self.org.set_team_influence(u, v, infl)

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
            # Update influence between agents based on aggregated opinions

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
    print(model.org.get_agent_influence("Agent 5", "Agent 2"))