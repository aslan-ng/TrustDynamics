import numpy as np
import networkx as nx
import tqdm

from trustdynamics.organization import Organization
from trustdynamics.trust import Degroot
from trustdynamics.utils import (
    bounded_random_with_exact_mean,
    map_to_range,
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
        """
        trust_min = 0.05
        trust_max = 0.95

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
        """
        trust_min = 0.05
        trust_max = 0.95

        G = self.org.G_teams
        if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
            return

        dc_in = nx.in_degree_centrality(G)
        for u, v in G.edges():
            trust = map_to_range(dc_in.get(v, 0.0), trust_min, trust_max)
            self.org.set_team_trust(u, v, trust)

    def run(
        self,
        steps: int = 1,
        show_progress: bool = True,
    ):
        """
        Run the model for a fixed number of steps.
        """
        if steps <= 0:
            raise ValueError("steps must be > 0")

        iterator = range(steps)
        if show_progress:
            iterator = tqdm(iterator, total=steps, desc="Running model")

        for _ in iterator:
            self.update()

    def update(self):
        """
        1. Update the agents opinions based on communication
        2. Update the agents opinions
        3. Update the teams opinions based on communication
        """
        self.update_teams_opinion()
        self.update_organization_opinion()
        self.update_teams_trust()
        #self.update_agents_trust()
        #self.agents_use_technology()

    def update_teams_opinion(self):
        """
        Assign initial team opinions by aggregating member agents' opinions
        via DeGroot dynamics within each team.

        Team opinion = mean of DeGroot final opinions of its agents.
        """
        # If there are no teams, nothing to do
        if len(self.org.all_team_ids) == 0:
            return

        for team_id in self.org.all_team_ids:
            agents = list(self.org.agents_from_team(team_id))
            if len(agents) == 0:
                # Optional: define behavior for empty teams
                # self.org.set_team_opinion(team_id, 0.0)
                continue

            if len(agents) == 1:
                # Team opinion equals the only agent's current opinion
                only_agent = next(iter(agents))
                self.org.set_team_opinion(team_id, float(self.org.get_agent_opinion(only_agent)))
                continue
            
            # Run DeGroot to convergence
            W, x0 = self.org.agent_influence_and_opinions(team_id)
            dg = Degroot(W)
            res = dg.run_steps(x0, steps=None, threshold=1e-6, max_steps=10_000)
            x_final = res["final_opinions"]
            team_opinion = float(x_final.mean()) # Aggregate to a single team opinion (robust choice)
            self.org.set_team_opinion(team_id, team_opinion)

    def update_organization_opinion(self):
        # Run DeGroot to convergence
        W, x0 = self.org.team_influence_and_opinions()
        dg = Degroot(W)
        res = dg.run_steps(x0, steps=None, threshold=1e-6, max_steps=10_000)
        x_final = res["final_opinions"]        
        organization_opinion = float(x_final.mean()) # Aggregate to a single team opinion (robust choice)
        self.org.set_organization_opinion(organization_opinion)

    def update_teams_trust(self):
        self_trust_learning_rate = 0.1
        neighbor_trust_learning_rate = 0.1
        w_agree = 0.5  # 0..1, higher => more homophily, lower => more normative
        
        organization_opinion = self.org.get_organization_opinion()
        for team_id in self.org.all_team_ids:
            team_opinion = self.org.get_team_opinion(team_id)
            team_self_trust = self.org.get_team_trust(team_1=team_id, team_2=team_id) # confidence
            team_organization_opinion_distance = abs(team_opinion - organization_opinion) # how “deviant” team is from org
            # Update self-trust
            d_TO = 1.0 - team_organization_opinion_distance / 2.0 # 1 when aligned, 0 when maximally deviant
            new_team_self_trust = (1 - self_trust_learning_rate) * team_self_trust + \
                                  (self_trust_learning_rate * d_TO) # Smooth update (inertia)
            new_team_self_trust = float(np.clip(new_team_self_trust, 0.0, 1.0)) # assure it is between 0 and 1
            self.org.set_team_trust(team_1=team_id, team_2=team_id, trust=new_team_self_trust)

            # Update trust in neighbors
            teams_connected = self.org.teams_connected_to(team_id)
            for neighbor_id in teams_connected:
                neighbor_opinion = self.org.get_team_opinion(neighbor_id)
                neighbor_trust = self.org.get_team_trust(team_1=team_id, team_2=neighbor_id)
                team_neighbor_opinion_distance = abs(team_opinion - neighbor_opinion) # agreement / homophily
                organization_neighbor_opinion_distance = abs(organization_opinion - neighbor_opinion) # normative
                
                # Map distances (0..2) to similarities (1..0)
                agree = 1.0 - (team_neighbor_opinion_distance / 2.0)          # team vs neighbor
                align = 1.0 - (organization_neighbor_opinion_distance / 2.0)  # neighbor vs org

                target_neighbor_trust = (w_agree * agree) + ((1.0 - w_agree) * align)
                new_neighbor_trust = (1.0 - neighbor_trust_learning_rate) * neighbor_trust + \
                                                    (neighbor_trust_learning_rate * target_neighbor_trust)
                new_neighbor_trust = float(np.clip(new_neighbor_trust, 0.0, 1.0)) # keep in [0, 1]
                self.org.set_team_trust(team_1=team_id, team_2=neighbor_id, trust=new_neighbor_trust)
                
    def update_agents_trust(self):
        pass

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


if __name__ == "__main__":
    
    from trustdynamics.organization.samples import organization_0 as org

    model = Model(
        org=org,
        technology_success_rate=0.9,
        average_initial_opinion=0.0,
        seed=42
    )
    #print(model.org.get_agent_trust("Agent 5", "Agent 2"))
    #print(model.org.get_agent_trust("Agent 2", "Agent 5"))

    #print(model.org.get_agent_trust("Agent 4", "Agent 2"))
    #print(model.org.get_agent_trust("Agent 2", "Agent 4"))

    #print(model.org.get_agent_trust("Agent 4", "Agent 3"))
    #print(model.org.get_agent_trust("Agent 3", "Agent 4"))
    model.update()
    print(model.org.get_team_trust_history("Team A", "Team B"))
    print(model.org.get_agent_trust_history("Agent 2", "Agent 3"))