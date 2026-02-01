import numpy as np
import networkx as nx
from tqdm import tqdm

from trustdynamics.organization import Organization
from trustdynamics.trust import Degroot
from trustdynamics.utils import (
    bounded_random_with_exact_mean,
    map_to_range,
)


class Model:
    """
    Core simulation model for trust and opinion dynamics in a multi-level organization.

    The model operates on three coupled layers:
    (1) agents within teams,
    (2) teams within an organization,
    (3) an organization-wide aggregate opinion.

    At each step, opinions evolve via DeGroot influence dynamics,
    while trust relationships adapt based on a tradeoff between homophily
    (agreement with others) and normative alignment with higher-level beliefs.
    """

    def __init__(
        self,
        org: Organization,
        technology_success_rate: float = 1.0,
        tech_successful_delta: float = 0.05,
        tech_failure_delta: float = -0.15,
        average_initial_opinion: float = 0.0,
        agents_self_trust_learning_rate: float = 0.1,
        agents_neighbor_trust_learning_rate: float = 0.1,
        agents_homophily_normative_tradeoff: float = 0.5,
        teams_self_trust_learning_rate: float = 0.1,
        teams_neighbor_trust_learning_rate: float = 0.1,
        teams_homophily_normative_tradeoff: float = 0.5,
        seed: int | None = None,
    ):
        """
        Initialize the trust–opinion dynamics model.

        Parameters
        ----------
        org : Organization
            Organizational structure containing agents, teams, network topology.
            Contains trust relationships and opinions.

        technology_success_rate : float, optional
            Probability that an interaction with the technology is successful
            at each step. Must lie in [0, 1].

        tech_successful_delta : float, optional
            Opinion shift applied when a technology interaction succeeds. Must lie in [0, 1].

        tech_failure_delta : float, optional
            Opinion shift applied when a technology interaction fails. Must lie in [-1, 0].

        average_initial_opinion : float, optional
            Mean initial opinion assigned to agents at model initialization.
            Opinions are assumed to lie in [-1, 1].

        agents_self_trust_learning_rate : float, optional
            Learning rate controlling how quickly an agent updates its
            self-trust (confidence) based on alignment with its team's opinion.

        agents_neighbor_trust_learning_rate : float, optional
            Learning rate controlling how quickly an agent updates trust in
            neighboring agents.

        agents_homophily_normative_tradeoff : float, optional
            Tradeoff parameter in [0, 1] governing agent-level trust updates.
            Values closer to 1 emphasize homophily (agreement with neighbors),
            while values closer to 0 emphasize normative alignment with team-level
            beliefs.

        teams_self_trust_learning_rate : float, optional
            Learning rate controlling how quickly a team updates its
            self-trust (confidence) based on alignment with the organization.

        teams_neighbor_trust_learning_rate : float, optional
            Learning rate controlling how quickly a team updates trust in
            neighboring teams.

        teams_homophily_normative_tradeoff : float, optional
            Tradeoff parameter in [0, 1] governing team-level trust updates.
            Values closer to 1 emphasize homophily (inter-team agreement), while values
            closer to 0 emphasize alignment with the organization-wide opinion (normative).

        seed : int or None, optional
            Random seed for reproducibility. If None, randomness is not seeded.
        """
        self.rng = np.random.default_rng(seed)
        self.org = org

        if technology_success_rate < 0.0 or technology_success_rate > 1.0:
            raise ValueError("technology_success_rate must be between 0.0 and 1.0")
        self.technology_success_rate = technology_success_rate

        if tech_successful_delta < 0.0 or tech_successful_delta > 1.0:
            raise ValueError("tech_successful_delta must be between 0.0 and 1.0")
        self.tech_successful_delta = tech_successful_delta

        if tech_failure_delta > 0.0 or tech_failure_delta < -1.0:  
            raise ValueError("tech_failure_delta must be between -1.0 and 0.0")
        self.tech_failure_delta = tech_failure_delta

        if average_initial_opinion < -1.0 or average_initial_opinion > 1.0:
            raise ValueError("average_initial_opinion must be between -1.0 and 1.0")
        
        if agents_self_trust_learning_rate < 0.0 or agents_self_trust_learning_rate > 1.0:
            raise ValueError("agents_self_trust_learning_rate must be between 0.0 and 1.0")
        self.agents_self_trust_learning_rate = agents_self_trust_learning_rate

        if agents_neighbor_trust_learning_rate < 0.0 or agents_neighbor_trust_learning_rate > 1.0:
            raise ValueError("agents_neighbor_trust_learning_rate must be between 0.0 and 1.0")
        self.agents_neighbor_trust_learning_rate = agents_neighbor_trust_learning_rate

        if agents_homophily_normative_tradeoff < 0.0 or agents_homophily_normative_tradeoff > 1.0:
            raise ValueError("agents_homophily_normative_tradeoff must be between 0.0 and 1.0")
        self.agents_homophily_normative_tradeoff = agents_homophily_normative_tradeoff

        if teams_self_trust_learning_rate < 0.0 or teams_self_trust_learning_rate > 1.0:
            raise ValueError("teams_self_trust_learning_rate must be between 0.0 and 1.0")
        self.teams_self_trust_learning_rate = teams_self_trust_learning_rate

        if teams_neighbor_trust_learning_rate < 0.0 or teams_neighbor_trust_learning_rate > 1.0:
            raise ValueError("teams_neighbor_trust_learning_rate must be between 0.0 and 1.0")
        self.teams_neighbor_trust_learning_rate = teams_neighbor_trust_learning_rate

        if teams_homophily_normative_tradeoff < 0.0 or teams_homophily_normative_tradeoff > 1.0:
            raise ValueError("teams_homophily_normative_tradeoff must be between 0.0 and 1.0")
        self.teams_homophily_normative_tradeoff = teams_homophily_normative_tradeoff
 
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
        trust_min = 0.01
        trust_max = 0.99

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
        trust_min = 0.01
        trust_max = 0.99

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
        """
        Calculate organization opinion from aggregating team opinions.
        """
        # Run DeGroot to convergence
        W, x0 = self.org.team_influence_and_opinions()
        dg = Degroot(W)
        res = dg.run_steps(x0, steps=None, threshold=1e-6, max_steps=10_000)
        x_final = res["final_opinions"]        
        organization_opinion = float(x_final.mean()) # Aggregate to a single team opinion (robust choice)
        self.org.set_organization_opinion(organization_opinion)

    def update_teams_trust(self):
        """
        Update trust between teams and self-trust (confidence) of each team.
        """
        self_trust_learning_rate = self.teams_self_trust_learning_rate
        neighbor_trust_learning_rate = self.teams_neighbor_trust_learning_rate
        w_agree = self.teams_homophily_normative_tradeoff  # 0..1, higher => more homophily, lower => more normative
        
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
        """
        Update trust between agents and self-trust (confidence) of each agent.
        """
        self_trust_learning_rate = self.agents_self_trust_learning_rate
        neighbor_trust_learning_rate = self.agents_neighbor_trust_learning_rate
        w_agree = self.agents_homophily_normative_tradeoff  # 0..1, higher => more homophily, lower => more normative

        for team_id in self.org.all_team_ids:
            team_opinion = self.org.get_team_opinion(team_id)
            agent_ids = self.org.agents_from_team(team_id)
            for agent_id in agent_ids:
                agent_opinion = self.org.get_agent_opinion(agent_id)
                agent_self_trust = self.org.get_agent_trust(agent_1=agent_id, agent_2=agent_id)

                # Update self-trust
                agent_team_distance = abs(agent_opinion - team_opinion)  # 0..2 if opinions in [-1, 1]
                d_AT = 1.0 - (agent_team_distance / 2.0)                 # similarity in [0, 1]
                new_agent_self_trust = (1.0 - self_trust_learning_rate) * agent_self_trust + \
                                    (self_trust_learning_rate * d_AT)
                new_agent_self_trust = float(np.clip(new_agent_self_trust, 0.0, 1.0))
                self.org.set_agent_trust(agent_1=agent_id, agent_2=agent_id, trust=new_agent_self_trust)

                agents_connected = self.org.agents_connected_to(agent_id)
                for neighbor_id in agents_connected:
                    neighbor_opinion = self.org.get_agent_opinion(neighbor_id)
                    neighbor_trust = self.org.get_agent_trust(agent_1=agent_id, agent_2=neighbor_id)

                    # Update trust in neighbors
                    agent_neighbor_distance = abs(agent_opinion - neighbor_opinion)
                    agree = 1.0 - (agent_neighbor_distance / 2.0) # Homophily: agent vs neighbor

                    team_neighbor_distance = abs(team_opinion - neighbor_opinion)
                    align = 1.0 - (team_neighbor_distance / 2.0) # Normative: neighbor vs TEAM (same team as focal agent)

                    target_neighbor_trust = (w_agree * agree) + ((1.0 - w_agree) * align)

                    new_neighbor_trust = (1.0 - neighbor_trust_learning_rate) * neighbor_trust + \
                                        (neighbor_trust_learning_rate * target_neighbor_trust)
                    new_neighbor_trust = float(np.clip(new_neighbor_trust, 0.0, 1.0))
                    self.org.set_agent_trust(agent_1=agent_id, agent_2=neighbor_id, trust=new_neighbor_trust)

    def agents_use_technology(self):
        """
        Agents opinions change when they use technology.
        """
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
    model.run(10)
    print(model.org.get_organization_opinion_history())