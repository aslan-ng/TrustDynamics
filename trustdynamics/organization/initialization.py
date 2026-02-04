import numpy as np

from trustdynamics.utils import (
    bounded_random_with_exact_mean,
    map_to_range,
)


class Initialization:
    """
    Mixin class responsible for initializing organization state.

    Initialization establishes:
    - initial agent opinions, if they don't exist from manual inputs
    - initial trust networks (agents and teams), if they don't exist from manual inputs

    These choices strongly influence early transient dynamics,
    but do not constrain long-term behavior.
    """

    def initialize(
        self,
        *,
        # Agents' opinion
        agents_average_initial_opinion: float = 0.0,
        agents_initial_opinion_min: float = -1.0,
        agents_initial_opinion_max: float = 1.0,
        # Agents' trust
        agents_initial_trust_min: float = 0.01,
        agents_initial_trust_max: float = 0.99,
        # Teams' trust
        teams_initial_trust_min: float = 0.01,
        teams_initial_trust_max: float = 0.99,
        # Random generator
        seed: int | None | np.random.Generator = None,
    ):
        """
        Initialize the model once before the first update cycle.

        This method is idempotent by design and guarded by the
        `self.initialized` flag in the update loop.

        Parameters
        ----------
        average_initial_opinion : float, optional
            Target mean for initial agent opinions sampled in [-1, 1].
            Initialization uses a bounded sampling routine that enforces this mean
            (up to numerical precision) while preserving heterogeneity.

        agents_initial_opinion_min : float, optional
            Lower bound for *initial* agents' opinion values.

        agents_initial_opinion_max : float, optional
            Upper bound for *initial* agents' opinion values.           
        
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
        """
        # Random generator
        if isinstance(seed, int) or seed is None:
            self.rng = np.random.default_rng(seed)
        elif isinstance(seed, np.random.Generator):
            self.rng = seed
        
        self.require_initialization: bool = False

        # Agents' opinion
        if agents_average_initial_opinion < -1.0 or agents_average_initial_opinion > 1.0:
            raise ValueError("agents_average_initial_opinion must be between -1.0 and 1.0")
        if agents_initial_opinion_min < -1.0 or agents_initial_opinion_min > 1.0:
            raise ValueError("agents_initial_opinion_min must be between -1.0 and 1.0")
        if agents_initial_opinion_max < -1.0 or agents_initial_opinion_max > 1.0:
            raise ValueError("agents_initial_opinion_max must be between -1.0 and 1.0")
        if agents_initial_opinion_max < agents_initial_opinion_min:
            raise ValueError("agents_initial_opinion_max must not be smaller than agents_initial_opinion_min.")
        self._initialize_agents_opinion(
            average_initial_opinion=agents_average_initial_opinion,
            initial_opinion_min=agents_initial_opinion_min,
            initial_opinion_max=agents_initial_opinion_max,
        )

        # Agents' trust
        if agents_initial_trust_min < 0.0 or agents_initial_trust_min > 1.0:
            raise ValueError("agents_initial_trust_min must be between 0.0 and 1.0")
        if agents_initial_trust_max < 0.0 or agents_initial_trust_max > 1.0:
            raise ValueError("agents_initial_trust_max must be between 0.0 and 1.0")
        if agents_initial_trust_max < agents_initial_trust_min:
            raise ValueError("agents_initial_trust_max must not be smaller than agents_initial_trust_min.")
        self._initialize_agents_trust(
            initial_trust_min=agents_initial_trust_min,
            initial_trust_max=agents_initial_trust_max,
        )

        # Teams' trust
        if teams_initial_trust_min < 0.0 or teams_initial_trust_min > 1.0:
            raise ValueError("teams_initial_trust_min must be between 0.0 and 1.0")
        if teams_initial_trust_max < 0.0 or teams_initial_trust_max > 1.0:
            raise ValueError("teams_initial_trust_max must be between 0.0 and 1.0")
        if teams_initial_trust_max < teams_initial_trust_min:
            raise ValueError("teams_initial_trust_max must not be smaller than teams_initial_trust_min.")
        self._initialize_teams_trust(
            initial_trust_min=teams_initial_trust_min,
            initial_trust_max=teams_initial_trust_max,
        )

        self.require_initialization = False

    def _initialize_agents_opinion(
        self,
        average_initial_opinion: float = 0.0,
        initial_opinion_min: float = -1.0,
        initial_opinion_max: float = 1.0,
    ):
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
        agent_ids = list(self.all_agent_ids)
        n = len(agent_ids)
        if n == 0:
            return
        
        # Stop overwriting already existing initial opinions (if any)
        existing_opinions = []
        for agent_id in agent_ids:
            existing_opinion = self.get_agent_opinion(agent_id)
            if existing_opinion is not None:
                existing_opinions.append(existing_opinion)

        opinions = bounded_random_with_exact_mean(
            n_total=n,
            target_mean=average_initial_opinion,
            fixed_values=existing_opinions,
            seed=self.rng, # pass the Generator to keep reproducibility tied to Model
            min_value=initial_opinion_min,
            max_value=initial_opinion_max,
        )
        for agent_id, opinion in zip(agent_ids, opinions):
            self.set_agent_opinion(agent_id, float(opinion))

    def _initialize_agents_trust(
        self,
        initial_trust_min: float = 0.01,
        initial_trust_max: float = 0.99,
    ):
        """
        Assign initial trust values between agents.

        Strategy:
        - Use in-degree centrality as a proxy for perceived importance
        - Trust flows *toward* structurally central agents
        - Map centrality values into (0, 1) to avoid degenerate dynamics

        Interpretation:
        Agents who are more "listened to" initially receive higher trust.
        """
        G = self.G_agents
        if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
            return

        agent_ids = sorted(self.all_agent_ids)

        deg_arr = np.array([len(self.agents_connected_to(t)) for t in agent_ids], dtype=float)

        deg_min = float(deg_arr.min()) if deg_arr.size else 0.0
        deg_max = float(deg_arr.max()) if deg_arr.size else 0.0

        if deg_arr.size == 0 or deg_max == deg_min:
            c_arr = np.zeros_like(deg_arr)  # all equal centrality
        else:
            c_arr = (deg_arr - deg_min) / (deg_max - deg_min)  # in [0,1]

        centrality01 = dict(zip(agent_ids, c_arr))        
        
        # Self-trust
        for agent_id in agent_ids:
            existing_value = self.get_agent_trust(agent_1=agent_id, agent_2=agent_id)
            if existing_value is None: # not overwriting already existing values
                trust = map_to_range(centrality01[agent_id], initial_trust_min, initial_trust_max)
                self.set_agent_trust(agent_id, agent_id, trust)

        # Trust in others
        for agent_id in agent_ids:
            connected_agents = self.agents_connected_to(agent_id)
            for other_agent_id in connected_agents:
                existing_value = self.get_agent_trust(agent_1=agent_id, agent_2=other_agent_id)
                if existing_value is None: # not overwriting already existing values
                    trust = map_to_range(centrality01[other_agent_id], initial_trust_min, initial_trust_max)
                    self.set_agent_trust(agent_id, other_agent_id, trust)

    def _initialize_teams_trust(
        self,
        initial_trust_min: float = 0.01,
        initial_trust_max: float = 0.99,
    ):
        """
        Assign initial trust values between teams.

        Mirrors agent-level initialization, but at the team layer.

        Interpretation:
        Teams that are structurally central in the organization
        are initially trusted more by other teams.
        """
        G = self.G_teams
        if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
            return

        team_ids = sorted(self.all_team_ids)

        deg_arr = np.array([len(self.teams_connected_to(t)) for t in team_ids], dtype=float)

        deg_min = float(deg_arr.min()) if deg_arr.size else 0.0
        deg_max = float(deg_arr.max()) if deg_arr.size else 0.0

        if deg_arr.size == 0 or deg_max == deg_min:
            c_arr = np.zeros_like(deg_arr)  # all equal centrality
        else:
            c_arr = (deg_arr - deg_min) / (deg_max - deg_min)  # in [0,1]

        centrality01 = dict(zip(team_ids, c_arr))        
        
        # Self-trust
        for team_id in team_ids:
            existing_value = self.get_team_trust(team_1=team_id, team_2=team_id)
            if existing_value is None: # not overwriting already existing values
                trust = map_to_range(centrality01[team_id], initial_trust_min, initial_trust_max)
                self.set_team_trust(team_id, team_id, trust)

        # Trust in others
        for team_id in team_ids:
            connected_teams = self.teams_connected_to(team_id)
            for other_team_id in connected_teams:
                existing_value = self.get_team_trust(team_1=team_id, team_2=other_team_id)
                if existing_value is None: # not overwriting already existing values
                    trust = map_to_range(centrality01[other_team_id], initial_trust_min, initial_trust_max)
                    self.set_team_trust(team_id, other_team_id, trust)