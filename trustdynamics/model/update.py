import numpy as np
import networkx as nx
from tqdm import tqdm

from trustdynamics.trust import Degroot
from trustdynamics.utils import (
    bounded_random_with_exact_mean,
    map_to_range,
)


class Update:

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

    def run(
        self,
        steps: int = 1,
        show_progress: bool = True,
    ):
        """
        Advance the simulation forward in discrete time steps.

        At each step, the model updates agent opinions, team opinions,
        trust relationships, and organization-level aggregates according
        to the current model configuration.

        Parameters
        ----------
        steps : int, optional
            Number of discrete simulation steps to execute. Must be greater
            than zero.

        show_progress : bool, optional
            If True, display a progress bar during execution.

        Raises
        ------
        ValueError
            If `steps` is less than or equal to zero.
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
        Update model for a single cycle.
        """
        if self.initialized is False:
            self.initialize()

        self.update_teams_opinion()
        self.update_organization_opinion()
        self.update_teams_trust()
        self.update_agents_trust()
        self.agents_use_technology()

    def update_teams_opinion(self):
        """
        Calculate team opinion from aggregating agents opinions.
        """
        # If there are no teams, nothing to do
        if len(self.organization.all_team_ids) == 0:
            return

        for team_id in self.organization.all_team_ids:
            agents = list(self.organization.agents_from_team(team_id))
            if len(agents) == 0:
                # Optional: define behavior for empty teams
                # self.organization.set_team_opinion(team_id, 0.0)
                continue

            if len(agents) == 1:
                # Team opinion equals the only agent's current opinion
                only_agent = next(iter(agents))
                self.organization.set_team_opinion(team_id, float(self.organization.get_agent_opinion(only_agent)))
                continue
            
            # Run DeGroot to convergence
            W, x0 = self.organization.agent_influence_and_opinions(team_id)
            dg = Degroot(W)
            res = dg.run_steps(x0, steps=None, threshold=1e-6, max_steps=10_000)
            x_final = res["final_opinions"]
            team_opinion = float(x_final.mean()) # Aggregate to a single team opinion (robust choice)
            self.organization.set_team_opinion(team_id, team_opinion)

    def update_organization_opinion(self):
        """
        Calculate organization opinion from aggregating team opinions.
        """
        # Run DeGroot to convergence
        W, x0 = self.organization.team_influence_and_opinions()
        dg = Degroot(W)
        res = dg.run_steps(x0, steps=None, threshold=1e-6, max_steps=10_000)
        x_final = res["final_opinions"]        
        organization_opinion = float(x_final.mean()) # Aggregate to a single team opinion (robust choice)
        self.organization.set_organization_opinion(organization_opinion)

    def update_teams_trust(self):
        """
        Update trust between teams and self-trust (confidence) of each team.
        """
        self_trust_learning_rate = self.teams_self_trust_learning_rate
        neighbor_trust_learning_rate = self.teams_neighbor_trust_learning_rate
        w_agree = self.teams_homophily_normative_tradeoff  # 0..1, higher => more homophily, lower => more normative
        
        organization_opinion = self.organization.get_organization_opinion()
        for team_id in self.organization.all_team_ids:
            team_opinion = self.organization.get_team_opinion(team_id)
            team_self_trust = self.organization.get_team_trust(team_1=team_id, team_2=team_id) # confidence
            team_organization_opinion_distance = abs(team_opinion - organization_opinion) # how “deviant” team is from org
            
            # Update self-trust
            d_TO = 1.0 - team_organization_opinion_distance / 2.0 # 1 when aligned, 0 when maximally deviant
            new_team_self_trust = (1 - self_trust_learning_rate) * team_self_trust + \
                                  (self_trust_learning_rate * d_TO) # Smooth update (inertia)
            new_team_self_trust = float(np.clip(new_team_self_trust, 0.0, 1.0)) # assure it is between 0 and 1
            self.organization.set_team_trust(team_1=team_id, team_2=team_id, trust=new_team_self_trust)

            # Update trust in neighbors
            teams_connected = self.organization.teams_connected_to(team_id)
            for neighbor_id in teams_connected:
                neighbor_opinion = self.organization.get_team_opinion(neighbor_id)
                neighbor_trust = self.organization.get_team_trust(team_1=team_id, team_2=neighbor_id)
                team_neighbor_opinion_distance = abs(team_opinion - neighbor_opinion) # agreement / homophily
                organization_neighbor_opinion_distance = abs(organization_opinion - neighbor_opinion) # normative
                
                # Map distances (0..2) to similarities (1..0)
                agree = 1.0 - (team_neighbor_opinion_distance / 2.0)          # team vs neighbor
                align = 1.0 - (organization_neighbor_opinion_distance / 2.0)  # neighbor vs org

                target_neighbor_trust = (w_agree * agree) + ((1.0 - w_agree) * align)
                new_neighbor_trust = (1.0 - neighbor_trust_learning_rate) * neighbor_trust + \
                                                    (neighbor_trust_learning_rate * target_neighbor_trust)
                new_neighbor_trust = float(np.clip(new_neighbor_trust, 0.0, 1.0)) # keep in [0, 1]
                self.organization.set_team_trust(team_1=team_id, team_2=neighbor_id, trust=new_neighbor_trust)
                
    def update_agents_trust(self):
        """
        Update trust between agents and self-trust (confidence) of each agent.
        """
        self_trust_learning_rate = self.agents_self_trust_learning_rate
        neighbor_trust_learning_rate = self.agents_neighbor_trust_learning_rate
        w_agree = self.agents_homophily_normative_tradeoff  # 0..1, higher => more homophily, lower => more normative

        for team_id in self.organization.all_team_ids:
            team_opinion = self.organization.get_team_opinion(team_id)
            agent_ids = self.organization.agents_from_team(team_id)
            for agent_id in agent_ids:
                agent_opinion = self.organization.get_agent_opinion(agent_id)
                agent_self_trust = self.organization.get_agent_trust(agent_1=agent_id, agent_2=agent_id)

                # Update self-trust
                agent_team_distance = abs(agent_opinion - team_opinion)  # 0..2 if opinions in [-1, 1]
                d_AT = 1.0 - (agent_team_distance / 2.0)                 # similarity in [0, 1]
                new_agent_self_trust = (1.0 - self_trust_learning_rate) * agent_self_trust + \
                                    (self_trust_learning_rate * d_AT)
                new_agent_self_trust = float(np.clip(new_agent_self_trust, 0.0, 1.0))
                self.organization.set_agent_trust(agent_1=agent_id, agent_2=agent_id, trust=new_agent_self_trust)

                agents_connected = self.organization.agents_connected_to(agent_id)
                for neighbor_id in agents_connected:
                    neighbor_opinion = self.organization.get_agent_opinion(neighbor_id)
                    neighbor_trust = self.organization.get_agent_trust(agent_1=agent_id, agent_2=neighbor_id)

                    # Update trust in neighbors
                    agent_neighbor_distance = abs(agent_opinion - neighbor_opinion)
                    agree = 1.0 - (agent_neighbor_distance / 2.0) # Homophily: agent vs neighbor

                    team_neighbor_distance = abs(team_opinion - neighbor_opinion)
                    align = 1.0 - (team_neighbor_distance / 2.0) # Normative: neighbor vs TEAM (same team as focal agent)

                    target_neighbor_trust = (w_agree * agree) + ((1.0 - w_agree) * align)

                    new_neighbor_trust = (1.0 - neighbor_trust_learning_rate) * neighbor_trust + \
                                        (neighbor_trust_learning_rate * target_neighbor_trust)
                    new_neighbor_trust = float(np.clip(new_neighbor_trust, 0.0, 1.0))
                    self.organization.set_agent_trust(agent_1=agent_id, agent_2=neighbor_id, trust=new_neighbor_trust)

    def agents_use_technology(self):
        """
        Agents opinions change when they use technology.
        """
        agents = self.organization.all_agent_ids
        for agent_id in agents:
            current_opinion = self.organization.get_agent_opinion(agent_id)
            tech_successful: bool = self.rng.random() < self.technology_success_rate
            if tech_successful:
                new_opinion = min(current_opinion + self.tech_successful_delta, 1.0)
            else:
                new_opinion = max(current_opinion + self.tech_failure_delta, -1.0)
            self.organization.set_agent_opinion(agent_id, new_opinion)