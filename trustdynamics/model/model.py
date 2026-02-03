import numpy as np

from trustdynamics.model.update import Update
from trustdynamics.model.initialization import Initialization
from trustdynamics.model.serialization import Serialization
from trustdynamics.organization import Organization


class Model(Initialization, Update, Serialization):
    """
    Core simulation model for trust and opinion dynamics in a multi-level organization.

    The model is hierarchical with three coupled layers:
    1) Agents within teams (micro layer)
    2) Teams within an organization (meso layer)
    3) Organization-wide aggregate opinion (macro layer)

    At each step:
    - Opinions propagate via consensus influence dynamics within teams and across teams.
    - Trust adapts via a convex combination of:
        * homophily (agreement with peers), and
        * normative alignment (agreement with the higher-level reference belief).
    - Agents experience an exogenous stochastic "technology shock" that shifts opinions.
    """

    SERIALIZATION_VERSION = 1

    def __init__(
        self,
        organization: Organization,
        technology_success_rate: float = 1.0,
        tech_successful_delta: float = 0.05,
        tech_failure_delta: float = -0.15,
        agents_average_initial_opinion: float = 0.0,
        agents_initial_opinion_min: float = -1.0,
        agents_initial_opinion_max: float = 1.0,
        agents_initial_trust_min: float = 0.01,
        agents_initial_trust_max: float = 0.99,
        teams_initial_trust_min: float = 0.01,
        teams_initial_trust_max: float = 0.99,
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
        organization : Organization
            Organizational state and topology. Contains:
            - agent graph and team graph (directed)
            - opinions and trust values stored on nodes/edges (via setters/getters)
            - methods to compute influence matrices for DeGroot updates

        technology_success_rate : float, optional
            Probability an agent's technology interaction succeeds at each step.
            Must lie in [0, 1].

        tech_successful_delta : float, optional
            Opinion increment applied to an agent upon successful technology use.
            Expected nonnegative; opinions are clipped to +1.

        tech_failure_delta : float, optional
            Opinion increment (typically negative) applied upon technology failure.
            Expected nonpositive; opinions are clipped to -1.

        average_initial_opinion : float, optional
            Target mean for initial agent opinions sampled in [-1, 1].
            Initialization uses a bounded sampling routine that enforces this mean
            (up to numerical precision) while preserving heterogeneity.

        agents_initial_trust_min : float, optional
            Lower bound for *initial* directed agent→agent trust values.
            During initialization, trust(u, v) is computed from in-degree centrality(v)
            and then mapped into [agents_initial_trust_min, agents_initial_trust_max].

        agents_initial_trust_max : float, optional
            Upper bound for *initial* directed agent→agent trust values.

        teams_initial_trust_min : float, optional
            Lower bound for *initial* directed team→team trust values.
            During initialization, trust(u, v) is computed from in-degree centrality(v)
            on the team graph and mapped into [teams_initial_trust_min, teams_initial_trust_max].

        teams_initial_trust_max : float, optional
            Upper bound for *initial* directed team→team trust values.

        agents_self_trust_learning_rate : float, optional
            Learning rate for updating agent self-trust (confidence) based on alignment
            with the agent's team opinion. Must lie in [0, 1].

        agents_neighbor_trust_learning_rate : float, optional
            Learning rate for updating trust from an agent to its neighbors.
            Must lie in [0, 1].

        agents_homophily_normative_tradeoff : float, optional
            Convex mixing weight in [0, 1] for agent-level neighbor trust updates:
            - 1.0 → purely homophily-driven (agent↔neighbor agreement)
            - 0.0 → purely normative (neighbor↔team alignment)

        teams_self_trust_learning_rate : float, optional
            Learning rate for updating team self-trust (confidence) based on alignment
            with organization opinion. Must lie in [0, 1].

        teams_neighbor_trust_learning_rate : float, optional
            Learning rate for updating trust from a team to neighboring teams.
            Must lie in [0, 1].

        teams_homophily_normative_tradeoff : float, optional
            Convex mixing weight in [0, 1] for team-level neighbor trust updates:
            - 1.0 → purely homophily-driven (team↔neighbor team agreement)
            - 0.0 → purely normative (neighbor team↔organization alignment)

        seed : int or None, optional
            Seed used to initialize the model RNG. If None, randomness is not seeded.
            The same RNG is passed into initialization routines to ensure end-to-end
            reproducibility across opinion sampling and stochastic technology outcomes.
        """
        self.rng = np.random.default_rng(seed)
        self.organization = organization
        self.initialized: bool = False

        # Technology
        if technology_success_rate < 0.0 or technology_success_rate > 1.0:
            raise ValueError("technology_success_rate must be between 0.0 and 1.0")
        self.technology_success_rate = technology_success_rate
        if tech_successful_delta < 0.0 or tech_successful_delta > 1.0:
            raise ValueError("tech_successful_delta must be between 0.0 and 1.0")
        self.tech_successful_delta = tech_successful_delta
        if tech_failure_delta > 0.0 or tech_failure_delta < -1.0:  
            raise ValueError("tech_failure_delta must be between -1.0 and 0.0")
        self.tech_failure_delta = tech_failure_delta

        # Agents initial opinion
        if agents_average_initial_opinion < -1.0 or agents_average_initial_opinion > 1.0:
            raise ValueError("agents_average_initial_opinion must be between -1.0 and 1.0")
        self.agents_average_initial_opinion = agents_average_initial_opinion
        if agents_initial_opinion_min < -1.0 or agents_initial_opinion_min > 1.0:
            raise ValueError("agents_initial_opinion_min must be between -1.0 and 1.0")
        if agents_initial_opinion_max < -1.0 or agents_initial_opinion_max > 1.0:
            raise ValueError("agents_initial_opinion_max must be between -1.0 and 1.0")
        if agents_initial_opinion_max < agents_initial_opinion_min:
            raise ValueError("agents_initial_opinion_max must not be smaller than agents_initial_opinion_min.")
        self.agents_initial_opinion_min = agents_initial_opinion_min
        self.agents_initial_opinion_max = agents_initial_opinion_max

        # Agents initial trust
        if agents_initial_trust_min < 0.0 or agents_initial_trust_min > 1.0:
            raise ValueError("agents_initial_trust_min must be between 0.0 and 1.0")
        if agents_initial_trust_max < 0.0 or agents_initial_trust_max > 1.0:
            raise ValueError("agents_initial_trust_max must be between 0.0 and 1.0")
        if agents_initial_trust_max < agents_initial_trust_min:
            raise ValueError("agents_initial_trust_max must not be smaller than agents_initial_trust_min.")
        self.agents_initial_trust_min = agents_initial_trust_min
        self.agents_initial_trust_max = agents_initial_trust_max

        # Teams initial trust
        if teams_initial_trust_min < 0.0 or teams_initial_trust_min > 1.0:
            raise ValueError("teams_initial_trust_min must be between 0.0 and 1.0")
        if teams_initial_trust_max < 0.0 or teams_initial_trust_max > 1.0:
            raise ValueError("teams_initial_trust_max must be between 0.0 and 1.0")
        if teams_initial_trust_max < teams_initial_trust_min:
            raise ValueError("teams_initial_trust_max must not be smaller than teams_initial_trust_min.")
        self.teams_initial_trust_min = teams_initial_trust_min
        self.teams_initial_trust_max = teams_initial_trust_max
        
        # Agent trust update
        if agents_self_trust_learning_rate < 0.0 or agents_self_trust_learning_rate > 1.0:
            raise ValueError("agents_self_trust_learning_rate must be between 0.0 and 1.0")
        self.agents_self_trust_learning_rate = agents_self_trust_learning_rate
        
        if agents_neighbor_trust_learning_rate < 0.0 or agents_neighbor_trust_learning_rate > 1.0:
            raise ValueError("agents_neighbor_trust_learning_rate must be between 0.0 and 1.0")
        self.agents_neighbor_trust_learning_rate = agents_neighbor_trust_learning_rate
        
        if agents_homophily_normative_tradeoff < 0.0 or agents_homophily_normative_tradeoff > 1.0:
            raise ValueError("agents_homophily_normative_tradeoff must be between 0.0 and 1.0")
        self.agents_homophily_normative_tradeoff = agents_homophily_normative_tradeoff

        # Team trust update
        if teams_self_trust_learning_rate < 0.0 or teams_self_trust_learning_rate > 1.0:
            raise ValueError("teams_self_trust_learning_rate must be between 0.0 and 1.0")
        self.teams_self_trust_learning_rate = teams_self_trust_learning_rate

        if teams_neighbor_trust_learning_rate < 0.0 or teams_neighbor_trust_learning_rate > 1.0:
            raise ValueError("teams_neighbor_trust_learning_rate must be between 0.0 and 1.0")
        self.teams_neighbor_trust_learning_rate = teams_neighbor_trust_learning_rate

        if teams_homophily_normative_tradeoff < 0.0 or teams_homophily_normative_tradeoff > 1.0:
            raise ValueError("teams_homophily_normative_tradeoff must be between 0.0 and 1.0")
        self.teams_homophily_normative_tradeoff = teams_homophily_normative_tradeoff

    @property
    def step(self):
        """
        Return the number of recorded organization-level opinion steps.

        Returns
        -------
        int
            Length of the organization opinion history.
        """
        return self.organization.__len__()


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