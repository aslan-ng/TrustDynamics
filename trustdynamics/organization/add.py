from trustdynamics.utils import new_unique_id


class Add:

    def add_team(
            self,
            name: str | None = None,
            initial_self_trust: float | None = None,
        ) -> int:
        """
        Add a new team to the organization.

        The team is created as a node in ``G_teams`` with an empty team-opinion
        history. A self-loop edge (team → team) is also created to represent
        the team's self-trust history.

        Parameters
        ----------
        name : str or None, optional
            Human-readable team name. If provided, it must be unique across all
            agents and teams in the organization. If ``None``, the team is added
            without a name.
        initial_self_trust : float or None, optional
            Initial value for the team's self-trust history. If provided, the
            self-loop edge ``(team_id, team_id)`` is initialized with
            ``trusts=[initial_self_trust]``; otherwise it is initialized with an
            empty list ``trusts=[]`` and will be populated during model initialization
            or simulation updates.

        Returns
        -------
        int
            Unique identifier of the newly created team.

        Raises
        ------
        ValueError
            If the provided team name is not unique.

        Notes
        -----
        - This method does not validate the numeric range of ``initial_self_trust``.
          If your trust domain is bounded (e.g., [0, 1]), consider validating here
          for stronger invariants.
        """
        if not self._is_name_unique(name):
            raise ValueError(f"Team name must be unique in the organization. '{name}' already exists.")
        
        team_id = new_unique_id(existing_values=self.all_ids)
        
        self.G_teams.add_node(
            team_id,
            name=name,
            opinions=[], # History of team opinions
        )

        if initial_self_trust is not None:
            trusts = [initial_self_trust]
        else:
            trusts = []

        self.G_teams.add_edge(
            team_id,
            team_id,
            trusts=trusts, # History of self-trust values
        )

        return team_id

    def add_agent(
            self,
            name: str | None = None,
            team: int | str = None,
            initial_opinion: float = None,
            initial_self_trust: float | None = None,
        ) -> int:
        """
        Add a new agent to an existing team.

        The agent is created as a node in ``G_agents``. The node stores the agent's
        team assignment and a history of opinion values. A self-loop edge
        (agent → agent) is also created to represent the agent's self-trust
        history.

        Parameters
        ----------
        name : str or None, optional
            Human-readable agent name. If provided, it must be unique across all
            agents and teams in the organization. If ``None``, the agent is added
            without a name.
        team : int or str
            Team identifier or team name to which the agent will belong. The team
            must already exist in the organization.
        initial_opinion : float or None, optional
            Initial opinion value for the agent. If provided, the agent node is
            initialized with ``opinions=[initial_opinion]``; otherwise the opinion
            history is initialized as an empty list and is expected to be populated
            during model initialization or later simulation steps.
        initial_self_trust : float or None, optional
            Initial value for the agent's self-trust history. If provided, the
            self-loop edge ``(agent_id, agent_id)`` is initialized with
            ``trusts=[initial_self_trust]``; otherwise it is initialized with an
            empty list ``trusts=[]``.

        Returns
        -------
        int
            Unique identifier of the newly created agent.

        Raises
        ------
        ValueError
            If the provided agent name is not unique.
        ValueError
            If ``team`` is ``None`` or cannot be resolved to an existing team.

        Notes
        -----
        - This method does not validate the numeric range of ``initial_opinion`` or
          ``initial_self_trust``. If your model expects bounded opinions (e.g., [-1, 1])
          and bounded trust (e.g., [0, 1]), consider validating here.
        """
        if not self._is_name_unique(name):
            raise ValueError(f"Agent name must be unique in the organization. '{name}' already exists.")
        
        agent_id = new_unique_id(existing_values=self.all_ids)
        
        if team is not None:
            team_id = self.search(team)
            if team_id is None:
                raise ValueError("Team must exist in the organization to add an agent.")
        else:
            team_id = None
            raise ValueError("Team cannot be None.") #####
        
        if initial_opinion is not None:
            opinions = [initial_opinion]
        else:
            opinions = []
        
        self.G_agents.add_node(
            agent_id,
            name=name,
            team=team_id,
            opinions=opinions, # History of agent opinions
        )

        if initial_self_trust is not None:
            trusts = [initial_self_trust]
        else:
            trusts = []

        self.G_agents.add_edge(
            agent_id,
            agent_id,
            trusts=trusts, # History of self-trust values
        )

        return agent_id

    def add_agent_connection(
        self,
        agent_1: int | str,
        agent_2: int | str,
        initial_trust_1_to_2: float | None = None,
        initial_trust_2_to_1: float | None = None,
    ):
        """
        Create a bidirectional directed trust connection between two agents.

        This method adds two directed edges: ``agent_1 → agent_2`` and
        ``agent_2 → agent_1``. Each edge stores a trust *history* list.

        Agents must belong to the same team.

        Parameters
        ----------
        agent_1, agent_2 : int or str
            Agent identifiers or names.
        initial_trust_1_to_2 : float or None, optional
            Initial trust value for the directed edge ``agent_1 → agent_2``.
            If provided, the edge is initialized with ``trusts=[initial_trust_1_to_2]``;
            otherwise with ``trusts=[]``.
        initial_trust_2_to_1 : float or None, optional
            Initial trust value for the directed edge ``agent_2 → agent_1``.
            If provided, the edge is initialized with ``trusts=[initial_trust_2_to_1]``;
            otherwise with ``trusts=[]``.

        Raises
        ------
        ValueError
            If either agent does not exist.
        ValueError
            If the agents are not assigned to the same team.

        Notes
        -----
        - This method creates directed edges. Many higher-level statistics in this
          codebase assume bidirectionality is enforced (i.e., both directions exist).
        - No numeric validation is performed on the initial trust values.
        """
        agent_1_id = self.search(agent_1)
        agent_2_id = self.search(agent_2)
        agent_1_team_id = self.agent_team_id(agent_1_id)
        agent_2_team_id = self.agent_team_id(agent_2_id)
        if agent_1_team_id is None or agent_2_team_id is None or agent_1_team_id != agent_2_team_id:
            raise ValueError("Both agents must belong to the same team.")
        
        if agent_1_id is not None and agent_2_id is not None:
            if initial_trust_1_to_2 is not None:
                trusts = [initial_trust_1_to_2]
            else:
                trusts = []
            self.G_agents.add_edge(
                agent_1_id,
                agent_2_id,
                trusts=trusts, # History of trust values
            )
            if initial_trust_2_to_1 is not None:
                trusts = [initial_trust_2_to_1]
            else:
                trusts = []
            self.G_agents.add_edge(
                agent_2_id,
                agent_1_id,
                trusts=trusts, # History of trust values
            )
        else:
            raise ValueError("Both agents must exist in the organization to add a connection.")

    def add_team_connection(
        self,
        team_1: int | str,
        team_2: int | str,
        initial_trust_1_to_2: float | None = None,
        initial_trust_2_to_1: float | None = None,
    ):
        """
        Create a bidirectional directed trust connection between two teams.

        This method adds two directed edges: ``team_1 → team_2`` and
        ``team_2 → team_1``. Each edge stores a trust *history* list.

        Parameters
        ----------
        team_1, team_2 : int or str
            Team identifiers or names.
        initial_trust_1_to_2 : float or None, optional
            Initial trust value for the directed edge ``team_1 → team_2``.
            If provided, the edge is initialized with ``trusts=[initial_trust_1_to_2]``;
            otherwise with ``trusts=[]``.
        initial_trust_2_to_1 : float or None, optional
            Initial trust value for the directed edge ``team_2 → team_1``.
            If provided, the edge is initialized with ``trusts=[initial_trust_2_to_1]``;
            otherwise with ``trusts=[]``.

        Raises
        ------
        ValueError
            If either team does not exist.

        Notes
        -----
        - No numeric validation is performed on the initial trust values.
        """
        team_1_id = self.search(team_1)
        team_2_id = self.search(team_2)
        if team_1_id is not None and team_2_id is not None:
            if initial_trust_1_to_2 is not None:
                trusts = [initial_trust_1_to_2]
            else:
                trusts = []
            self.G_teams.add_edge(
                team_1_id,
                team_2_id,
                trusts=trusts, # History of trust values
            )
            if initial_trust_2_to_1 is not None:
                trusts = [initial_trust_2_to_1]
            else:
                trusts = []
            self.G_teams.add_edge(
                team_2_id,
                team_1_id,
                trusts=trusts, # History of trust values
            )
        else:
            raise ValueError("Both teams must exist in the organization to add a connection.")