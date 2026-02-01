import numpy as np

from trustdynamics.model.update import Update
from trustdynamics.model.initialization import Initialization
from trustdynamics.model.serialization import Serialization
from trustdynamics.organization import Organization


class Model(Initialization, Update, Serialization):
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

    SERIALIZATION_VERSION = 1

    def __init__(
        self,
        organization: Organization,
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
        self.organization = organization
        self.initialized: bool = False

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
        self.average_initial_opinion = average_initial_opinion
        
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


if __name__ == "__main__":
    
    from trustdynamics.organization.samples import organization_0 as organization

    model = Model(
        organization=organization,
        technology_success_rate=0.9,
        average_initial_opinion=0.0,
        seed=42
    )
    model.run(10)
    print(model.organization.get_organization_opinion_history())