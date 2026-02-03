import numpy as np
import networkx as nx
import pandas as pd


from trustdynamics.organization.add import Add
from trustdynamics.organization.initialization import Initialization
from trustdynamics.organization.serialization import Serialization
from trustdynamics.organization.graphics import Graphics
from trustdynamics.organization.stat import Stat
from trustdynamics.utils import row_stochasticize


class Organization(
    Add,
    Initialization,
    Serialization,
    Graphics,
    Stat
):
    """
    Hierarchical organization model composed of teams and agents.

    This class represents a multi-level social system with:
    - agents belonging to teams,
    - directed trust relationships within and between levels,
    - evolving opinions at agent, team, and organization scales.

    Internally, the organization is represented using two directed graphs:
    - ``G_agents`` for agent-to-agent trust relationships (intra-team only),
    - ``G_teams`` for team-to-team trust relationships.

    Each node and edge stores *time series* (histories) of opinions or trust
    values, enabling dynamic simulations of opinion and trust evolution.

    Inherits
    --------
    Serialization
        Provides versioned save/load support.
    Graphics
        Provides visualization utilities for teams and agents.
    """

    def __init__(
        self,
        name: str = "Organization",
        seed: int | None | np.random.Generator = None,
    ):
        """
        Initialize an empty organization.

        Parameters
        ----------
        name : str, optional
            Human-readable name of the organization.
        """
        self.name = name # human-readable name of the organization.

        self.G_teams = nx.DiGraph() # directional graph to save teams data
        self.G_agents = nx.DiGraph() # directional graph to save agents data
        self.opinions = [] # history of orgnization aggregate opinions

        if isinstance(seed, int) or seed is None:
            self.rng = np.random.default_rng(seed)
        elif isinstance(seed, np.random.Generator):
            self.rng = seed
        
        self.initialized: bool = False

    @property
    def all_team_ids(self) -> set:
        """
        Return all non-empty team names.

        Returns
        -------
        set
            Set of team names.
        """
        return set(self.G_teams.nodes())
    
    @property
    def all_team_names(self) -> set:
        """
        Return all agent node IDs.

        Returns
        -------
        set
            Set of integer agent identifiers.
        """
        names = set()
        for _, attrs in self.G_teams.nodes(data=True):
            if attrs.get("name") is not None and attrs.get("name") != "":
                names.add(attrs.get("name"))
        return names
    
    @property
    def all_agent_ids(self) -> set:
        """
        Return all agent node IDs.

        Returns
        -------
        set
            Set of integer agent identifiers.
        """
        return set(self.G_agents.nodes())

    @property
    def all_agent_names(self) -> set:
        """
        Return all non-empty agent names.

        Returns
        -------
        set
            Set of agent names.
        """
        names = set()
        for _, attrs in self.G_agents.nodes(data=True):
            if attrs.get("name") is not None and attrs.get("name") != "":
                names.add(attrs.get("name"))
        return names

    @property
    def all_ids(self) -> set:
        """
        Return all node IDs (agents and teams).

        Returns
        -------
        set
            Union of agent and team IDs.
        """
        return self.all_team_ids.union(self.all_agent_ids)
    
    @property
    def all_names(self) -> set:
        """
        Return all node names (agents and teams).

        Returns
        -------
        set
            Union of agent and team names.
        """
        return self.all_team_names.union(self.all_agent_names)
    
    def agents_from_team(self, team: int | str | None) -> set:
        """
        Return all agents belonging to a given team.

        Parameters
        ----------
        team : int | str | None
            Team identifier or name.

        Returns
        -------
        set
            Set of agent IDs belonging to the team.

        Raises
        ------
        ValueError
            If the team does not exist.
        """
        if team is not None:
            team_id = self.search(team)
            if team_id is None:
                raise ValueError("Team must exist in the organization to get its agents.")
        else:
            team_id = None
        agents = set()
        for node_id, attrs in self.G_agents.nodes(data=True):
            if attrs.get("team") == team_id:
                agents.add(node_id)
        return agents
    
    def agent_team_id(self, agent: int | str) -> int | None:
        """
        Return the team ID an agent belongs to.

        Parameters
        ----------
        agent : int | str
            Agent identifier or name.

        Returns
        -------
        int or None
            Team ID, or None if not assigned.

        Raises
        ------
        ValueError
            If the agent does not exist.
        """
        agent_id = self.search(agent)
        if agent_id is None:
            raise ValueError("Agent must exist in the organization")
        return self.G_agents.nodes[agent_id].get("team", None)
    
    def _is_name_unique(self, name: str | None) -> bool:
        """
        Check whether a node name is unique.

        Parameters
        ----------
        name : str or None
            Proposed name.

        Returns
        -------
        bool
            True if unique or None, False otherwise.
        """
        names = self.all_names
        if name is None:
            return True
        for existing_name in names:
            if existing_name == name:
                return False
        return True
        
    def search(self, input: int | str) -> int | None:
        """
        Resolve agent or team node name or ID to its integer ID.

        Parameters
        ----------
        input : int | str
            Node ID or name.

        Returns
        -------
        int or None
            Resolved node ID, or None if not found.
        """
        if isinstance(input, int):
            return input if input in self.all_ids else None
        elif isinstance(input, str):
            for node_id, attrs in self.G_teams.nodes(data=True):
                if attrs.get("name") == input and attrs.get("name") is not None and attrs.get("name") != "":
                    return node_id
            for node_id, attrs in self.G_agents.nodes(data=True):
                if attrs.get("name") == input and attrs.get("name") is not None and attrs.get("name") != "":
                    return node_id
            return None
        else:
            return None
        
    # =========================
    # Agent-level parameters
    # =========================

    def get_agent_technology_success_impact(self, agent: int | str) -> float:
        agent_id = self.search(agent)
        if agent_id is None:
            raise ValueError("Agent must exist in the organization to get technology_success_impact.")
        return self.G_agents.nodes[agent_id].get("technology_success_impact", 0.0)

    def set_agent_technology_success_impact(self, agent: int | str, value: float) -> None:
        agent_id = self.search(agent)
        if agent_id is None:
            raise ValueError("Agent must exist in the organization to set technology_success_impact.")
        self.G_agents.nodes[agent_id]["technology_success_impact"] = float(value)

    def get_agent_technology_failure_impact(self, agent: int | str) -> float:
        agent_id = self.search(agent)
        if agent_id is None:
            raise ValueError("Agent must exist in the organization to get technology_failure_impact.")
        return self.G_agents.nodes[agent_id].get("technology_failure_impact", 0.0)

    def set_agent_technology_failure_impact(self, agent: int | str, value: float) -> None:
        agent_id = self.search(agent)
        if agent_id is None:
            raise ValueError("Agent must exist in the organization to set technology_failure_impact.")
        self.G_agents.nodes[agent_id]["technology_failure_impact"] = float(value)

    def get_agent_self_trust_learning_rate(self, agent: int | str) -> float:
        agent_id = self.search(agent)
        if agent_id is None:
            raise ValueError("Agent must exist in the organization to get self_trust_learning_rate.")
        return self.G_agents.nodes[agent_id].get("self_trust_learning_rate", 0.0)

    def set_agent_self_trust_learning_rate(self, agent: int | str, value: float) -> None:
        agent_id = self.search(agent)
        if agent_id is None:
            raise ValueError("Agent must exist in the organization to set self_trust_learning_rate.")
        self.G_agents.nodes[agent_id]["self_trust_learning_rate"] = float(value)


    def get_agent_trust_learning_rate(self, agent: int | str) -> float:
        agent_id = self.search(agent)
        if agent_id is None:
            raise ValueError("Agent must exist in the organization to get trust_learning_rate.")
        return self.G_agents.nodes[agent_id].get("trust_learning_rate", 0.0)


    def set_agent_trust_learning_rate(self, agent: int | str, value: float) -> None:
        agent_id = self.search(agent)
        if agent_id is None:
            raise ValueError("Agent must exist in the organization to set trust_learning_rate.")
        self.G_agents.nodes[agent_id]["trust_learning_rate"] = float(value)


    def get_agent_homophily_normative_tradeoff(self, agent: int | str) -> float:
        agent_id = self.search(agent)
        if agent_id is None:
            raise ValueError("Agent must exist in the organization to get homophily_normative_tradeoff.")
        return self.G_agents.nodes[agent_id].get("homophily_normative_tradeoff", 0.5)


    def set_agent_homophily_normative_tradeoff(self, agent: int | str, value: float) -> None:
        agent_id = self.search(agent)
        if agent_id is None:
            raise ValueError("Agent must exist in the organization to set homophily_normative_tradeoff.")
        self.G_agents.nodes[agent_id]["homophily_normative_tradeoff"] = float(value)
        
    # =========================
    # Team-level parameters
    # =========================

    def get_team_self_trust_learning_rate(self, team: int | str) -> float:
        team_id = self.search(team)
        if team_id is None:
            raise ValueError("Team must exist in the organization to get self_trust_learning_rate.")
        return self.G_teams.nodes[team_id].get("self_trust_learning_rate", 0.0)

    def set_team_self_trust_learning_rate(self, team: int | str, value: float) -> None:
        team_id = self.search(team)
        if team_id is None:
            raise ValueError("Team must exist in the organization to set self_trust_learning_rate.")
        self.G_teams.nodes[team_id]["self_trust_learning_rate"] = float(value)

    def get_team_trust_learning_rate(self, team: int | str) -> float:
        team_id = self.search(team)
        if team_id is None:
            raise ValueError("Team must exist in the organization to get trust_learning_rate.")
        return self.G_teams.nodes[team_id].get("trust_learning_rate", 0.0)

    def set_team_trust_learning_rate(self, team: int | str, value: float) -> None:
        team_id = self.search(team)
        if team_id is None:
            raise ValueError("Team must exist in the organization to set trust_learning_rate.")
        self.G_teams.nodes[team_id]["trust_learning_rate"] = float(value)

    def get_team_homophily_normative_tradeoff(self, team: int | str) -> float:
        team_id = self.search(team)
        if team_id is None:
            raise ValueError("Team must exist in the organization to get homophily_normative_tradeoff.")
        return self.G_teams.nodes[team_id].get("homophily_normative_tradeoff", 0.5)

    def set_team_homophily_normative_tradeoff(self, team: int | str, value: float) -> None:
        team_id = self.search(team)
        if team_id is None:
            raise ValueError("Team must exist in the organization to set homophily_normative_tradeoff.")
        self.G_teams.nodes[team_id]["homophily_normative_tradeoff"] = float(value)
    
    def get_agent_opinion(self, agent: int | str, history_index: int = -1) -> float:
        """
        Return an agent's opinion value at a given history index.

        Parameters
        ----------
        agent : int | str
            Agent identifier or name.
        history_index : int, optional
            Index into the stored opinion history. Defaults to ``-1`` (latest).

        Returns
        -------
        float
            Opinion value at the requested history index.

        Raises
        ------
        ValueError
            If the agent does not exist.
        IndexError
            If the agent has no opinion history or the index is out of range.
        """
        opinions = self.get_agent_opinions_history(agent)
        if not opinions:  # empty list
            return None
        try:
            return opinions[history_index]
        except IndexError:
            return None  
    
    def set_agent_opinion(self, agent: int | str, opinion: float):
        """
        Append a new opinion value to an agent's opinion history.

        Parameters
        ----------
        agent : int | str
            Agent identifier or name.
        opinion : float
            Opinion value to append.

        Raises
        ------
        ValueError
            If the agent does not exist.
        """
        opinion = float(opinion)
        agent_id = self.search(agent)
        if agent_id is None:
            raise ValueError("Agent must exist in the organization to set opinion.")
        self.G_agents.nodes[agent_id]["opinions"].append(opinion)

    def get_agent_opinions_history(self, agent: int | str) -> list:
        """
        Return the full opinion history for an agent.

        Parameters
        ----------
        agent : int | str
            Agent identifier or name.

        Returns
        -------
        list
            List of opinion values in chronological order.

        Raises
        ------
        ValueError
            If the agent does not exist.
        """
        agent_id = self.search(agent)
        if agent_id is None:
            raise ValueError("Agent must exist in the organization to get opinion.")
        return self.G_agents.nodes[agent_id].get("opinions", [])

    def get_team_opinion(self, team: int | str, history_index: int = -1) -> float:
        """
        Return a team's opinion value at a given history index.

        Parameters
        ----------
        team : int | str
            Team identifier or name.
        history_index : int, optional
            Index into the stored opinion history. Defaults to ``-1`` (latest).

        Returns
        -------
        float
            Team opinion at the requested history index.

        Raises
        ------
        ValueError
            If the team does not exist.
        IndexError
            If the team has no opinion history or the index is out of range.
        """
        opinions = self.get_team_opinions_history(team)
        if not opinions:  # empty list
            return None
        try:
            return opinions[history_index]
        except IndexError:
            return None
    
    def set_team_opinion(self, team: int | str, opinion: float):
        """
        Append a new opinion value to a team's opinion history.

        Parameters
        ----------
        team : int | str
            Team identifier or name.
        opinion : float
            Opinion value to append.

        Raises
        ------
        ValueError
            If the team does not exist.
        """
        opinion = float(opinion)
        team_id = self.search(team)
        if team_id is None:
            raise ValueError("Team must exist in the organization to set opinion.")
        self.G_teams.nodes[team_id]["opinions"].append(opinion)

    def get_team_opinions_history(self, team: int | str) -> list:
        """
        Return the full opinion history for a team.

        Parameters
        ----------
        team : int | str
            Team identifier or name.

        Returns
        -------
        list
            List of team opinion values in chronological order.

        Raises
        ------
        ValueError
            If the team does not exist.
        """
        team_id = self.search(team)
        if team_id is None:
            raise ValueError("Team must exist in the organization to get opinion.")
        return self.G_teams.nodes[team_id].get("opinions", [])
    
    def get_organization_opinion(self, history_index: int = -1) -> float:
        """
        Return the organization-level opinion at a given history index.

        Parameters
        ----------
        history_index : int, optional
            Index into the organization opinion history. Defaults to ``-1`` (latest).

        Returns
        -------
        float
            Organization opinion value.

        Raises
        ------
        IndexError
            If the organization opinion history is empty or index is out of range.
        """
        return self.opinions[history_index]
    
    def set_organization_opinion(self, opinion: float):
        """
        Append a new organization-level opinion value.

        Parameters
        ----------
        opinion : float
            Organization opinion value to append.
        """
        opinion = float(opinion)
        self.opinions.append(opinion)

    def get_organization_opinion_history(self) -> list:
        """
        Return the full organization opinion history.

        Returns
        -------
        list
            List of organization opinion values in chronological order.
        """
        return self.opinions
    
    def get_agent_exposure_to_technology(self, agent: int | str) -> bool:
        agent_id = self.search(agent)
        if agent_id is None:
            raise ValueError("Agent must exist in the organization to get exposure_to_technology.")
        return self.G_agents.nodes[agent_id].get("exposure_to_technology", [])
    
    def set_agent_exposure_to_technology(self, agent: int | str, value: bool) -> bool:
        agent_id = self.search(agent)
        if agent_id is None:
            raise ValueError("Agent must exist in the organization to set exposure_to_technology.")
        self.G_agents.nodes[agent_id]["exposure_to_technology"] = value
    
    def get_agent_trust(self, agent_1: int | str, agent_2: int | str, history_index: int = -1) -> float:
        """
        Return the trust value from one agent to another at a given history index.

        Parameters
        ----------
        agent_1 : int | str
            Source agent identifier or name (trustor).
        agent_2 : int | str
            Target agent identifier or name (trustee).
        history_index : int, optional
            Index into the trust history. Defaults to ``-1`` (latest).

        Returns
        -------
        float
            Trust value at the requested history index.

        Raises
        ------
        ValueError
            If either agent does not exist or the edge does not exist.
        IndexError
            If the trust history is empty or index is out of range.
        """
        trust_history = self.get_agent_trust_history(agent_1, agent_2)
        if not trust_history:  # empty list
            return None
        try:
            return trust_history[history_index]
        except IndexError:
            return None
    
    def set_agent_trust(self, agent_1: int | str, agent_2: int | str, trust: float):
        """
        Append a trust value from one agent to another.

        Parameters
        ----------
        agent_1 : int | str
            Source agent identifier or name (trustor).
        agent_2 : int | str
            Target agent identifier or name (trustee).
        trust : float
            Trust value to append.

        Raises
        ------
        ValueError
            If either agent does not exist or no edge exists between them.
        """
        trust = float(trust)
        agent_1_id = self.search(agent_1)
        agent_2_id = self.search(agent_2)
        if agent_1_id is None or agent_2_id is None:
            raise ValueError("Both agents must exist in the organization to get trust value.")
        if self.G_agents.has_edge(agent_1_id, agent_2_id):
            self.G_agents.edges[agent_1_id, agent_2_id]["trusts"].append(trust)
        else:
            raise ValueError("No connection exists between the specified agents.")
        
    def get_agent_trust_history(self, agent_1: int | str, agent_2: int | str) -> list:
        """
        Return the full trust history from one agent to another.

        Parameters
        ----------
        agent_1 : int | str
            Source agent identifier or name (trustor).
        agent_2 : int | str
            Target agent identifier or name (trustee).

        Returns
        -------
        list
            List of trust values in chronological order.

        Raises
        ------
        ValueError
            If either agent does not exist or no edge exists between them.
        """
        agent_1_id = self.search(agent_1)
        agent_2_id = self.search(agent_2)
        if agent_1_id is None or agent_2_id is None:
            raise ValueError("Both agents must exist in the organization to get trust value.")
        if self.G_agents.has_edge(agent_1_id, agent_2_id):
            return self.G_agents.edges[agent_1_id, agent_2_id].get("trusts", [])
        else:
            raise ValueError("No connection exists between the specified agents.")

    def get_team_trust(self, team_1: int | str, team_2: int | str, history_index: int = -1) -> list:
        """
        Return the trust value from one team to another at a given history index.

        Parameters
        ----------
        team_1 : int | str
            Source team identifier or name (trustor).
        team_2 : int | str
            Target team identifier or name (trustee).
        history_index : int, optional
            Index into the trust history. Defaults to ``-1`` (latest).

        Returns
        -------
        float
            Trust value at the requested history index.

        Raises
        ------
        ValueError
            If either team does not exist or the edge does not exist.
        IndexError
            If the trust history is empty or index is out of range.
        """
        trust_history = self.get_team_trust_history(team_1, team_2)
        if not trust_history:  # empty list
            return None
        try:
            return trust_history[history_index]
        except IndexError:
            return None
    
    def set_team_trust(self, team_1: int | str, team_2: int | str, trust: float):
        """
        Append a trust value from one team to another.

        Parameters
        ----------
        team_1 : int | str
            Source team identifier or name (trustor).
        team_2 : int | str
            Target team identifier or name (trustee).
        trust : float
            Trust value to append.

        Raises
        ------
        ValueError
            If either team does not exist or no edge exists between them.
        """
        trust = float(trust)
        team_1_id = self.search(team_1)
        team_2_id = self.search(team_2)
        if team_1_id is None or team_2_id is None:
            raise ValueError("Both teams must exist in the organization to get trust value.")
        if self.G_teams.has_edge(team_1_id, team_2_id):
            self.G_teams.edges[team_1_id, team_2_id]["trusts"].append(trust)
        else:
            raise ValueError("No connection exists between the specified teams.")

    def get_team_trust_history(self, team_1: int | str, team_2: int | str) -> list:
        """
        Return the full trust history from one team to another.

        Parameters
        ----------
        team_1 : int | str
            Source team identifier or name (trustor).
        team_2 : int | str
            Target team identifier or name (trustee).

        Returns
        -------
        list
            List of trust values in chronological order.

        Raises
        ------
        ValueError
            If either team does not exist or no edge exists between them.
        """
        team_1_id = self.search(team_1)
        team_2_id = self.search(team_2)
        if team_1_id is None or team_2_id is None:
            raise ValueError("Both teams must exist in the organization to get trust value.")
        if self.G_teams.has_edge(team_1_id, team_2_id):
            return self.G_teams.edges[team_1_id, team_2_id].get("trusts", [])
        else:
            raise ValueError("No connection exists between the specified teams.")
        
    def agent_influence_and_opinions(
            self,
            team: int | str,
            *,
            history_index: int = -1,
        ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Construct the agent-level influence matrix and opinion vector for a team.

        This returns a row-stochastic matrix ``W`` and an aligned opinion vector ``x``,
        both indexed by the same ordered set of agent IDs.

        Definitions
        ----------
        - ``W[i, j]`` is the influence weight agent ``i`` assigns to agent ``j``.
        In this implementation, raw influence weights are taken from trust histories
        stored on directed edges, then row-normalized via :func:`row_stochasticize`.
        - ``x[i]`` is agent ``i``'s opinion at ``history_index``.

        Parameters
        ----------
        team : int | str
            Team identifier or name.
        history_index : int, optional
            Index into trust/opinion histories. Defaults to ``-1`` (latest).

        Returns
        -------
        (pandas.DataFrame, pandas.Series)
            ``W`` (row-stochastic influence matrix) and ``x`` (opinion vector).

        Raises
        ------
        ValueError
            If the team does not exist or an agent has no opinion history.
        IndexError
            If requested history index is out of bounds for any trust/opinion history.

        Notes
        -----
        The returned matrix is guaranteed to be row-stochastic (each row sums to 1),
        with isolated agents assigned ``self_weight_if_isolated=1.0``.
        """
        agent_ids = sorted(self.agents_from_team(team))

        # --- Influence matrix ---
        W = pd.DataFrame(0.0, index=agent_ids, columns=agent_ids)

        for i in agent_ids:
            for j in agent_ids:
                if self.G_agents.has_edge(i, j):
                    trusts = self.get_agent_trust_history(i, j)
                    W.loc[i, j] = float(trusts[history_index]) if len(trusts) > 0 else 0.0

        W = row_stochasticize(W, self_weight_if_isolated=1.0)

        # --- Opinion vector ---
        x_vals = []
        for a in agent_ids:
            hist = self.get_agent_opinions_history(a)
            if len(hist) == 0:
                raise ValueError(f"Agent {a} has no opinion history.")
            x_vals.append(float(hist[history_index]))

        x = pd.Series(x_vals, index=agent_ids, name="opinions")

        # --- Safety invariants ---
        assert W.index.equals(W.columns)
        assert W.index.equals(x.index)

        return W, x
    
    def team_influence_and_opinions(
        self,
        *,
        history_index: int = -1,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Construct the team-level influence matrix and opinion vector for the organization.

        This returns a row-stochastic matrix ``W`` and an aligned opinion vector ``x``,
        both indexed by the same ordered set of team IDs.

        Definitions
        ----------
        - ``W[i, j]`` is the influence weight team ``i`` assigns to team ``j``.
        Raw influence weights are taken from trust histories stored on directed edges,
        then row-normalized via :func:`row_stochasticize`.
        - ``x[i]`` is team ``i``'s opinion at ``history_index``.

        Parameters
        ----------
        history_index : int, optional
            Index into trust/opinion histories. Defaults to ``-1`` (latest).

        Returns
        -------
        (pandas.DataFrame, pandas.Series)
            ``W`` (row-stochastic influence matrix) and ``x`` (opinion vector).

        Raises
        ------
        ValueError
            If any team has no opinion history.
        IndexError
            If requested history index is out of bounds for any trust/opinion history.

        Notes
        -----
        The returned matrix is guaranteed to be row-stochastic (each row sums to 1),
        with isolated teams assigned ``self_weight_if_isolated=1.0``.
        """
        team_ids = sorted(list(self.all_team_ids))

        # --- Influence matrix ---
        W = pd.DataFrame(0.0, index=team_ids, columns=team_ids)

        for i in team_ids:
            for j in team_ids:
                if self.G_teams.has_edge(i, j):
                    trusts = self.get_team_trust_history(i, j)
                    W.loc[i, j] = float(trusts[history_index]) if len(trusts) > 0 else 0.0

        W = row_stochasticize(W, self_weight_if_isolated=1.0)

        # --- Opinion vector ---
        x_vals = []
        for t in team_ids:
            hist = self.get_team_opinions_history(t)
            if len(hist) == 0:
                raise ValueError(f"Team {t} has no opinion history.")
            x_vals.append(float(hist[history_index]))

        x = pd.Series(x_vals, index=team_ids, name="opinions")

        # --- Safety invariants ---
        assert W.index.equals(W.columns)
        assert W.index.equals(x.index)

        return W, x

    def teams_connected_to(self, team: int | str) -> set:
        """
        Return the set of teams directly connected *outgoing* from a given team.

        This uses ``DiGraph.successors`` which returns teams ``j`` such that an edge
        ``team -> j`` exists.

        Parameters
        ----------
        team : int | str
            Team identifier or name.

        Returns
        -------
        set[int]
            Set of team IDs that are direct outgoing neighbors.

        Raises
        ------
        ValueError
            If the team does not exist.

        Notes
        -----
        The team's self-loop (team -> team) is always removed from the returned set.
        """
        team_id = self.search(team)
        if team_id is None:
            raise ValueError("Team must exist in the organization.")

        connected: set[int] = set()
        connected.update(self.G_teams.successors(team_id)) # Outgoing neighbors (team -> other)
        #connected.update(self.G_teams.predecessors(team_id)) # Incoming neighbors (other -> team)
        connected.discard(team_id) # Remove self (you always have a self-loop)
        return connected
    
    def agents_connected_to(self, agent: int | str):
        """
        Return the set of agents directly connected *outgoing* from a given agent.

        This uses ``DiGraph.successors`` which returns agents ``j`` such that an edge
        ``agent -> j`` exists.

        Parameters
        ----------
        agent : int | str
            Agent identifier or name.

        Returns
        -------
        set[int]
            Set of agent IDs that are direct outgoing neighbors.

        Raises
        ------
        ValueError
            If the agent does not exist.

        Notes
        -----
        The agent's self-loop (agent -> agent) is always removed from the returned set.
        """
        agent_id = self.search(agent)
        if agent_id is None:
            raise ValueError("Agent must exist in the organization.")

        connected: set[int] = set()
        connected.update(self.G_agents.successors(agent_id)) # Outgoing neighbors (agent -> other)
        #connected.update(self.G_agents.predecessors(agent_id)) # Incoming neighbors (other -> agent)
        connected.discard(agent_id) # Remove self (you always have a self-loop)
        return connected
    
    def average_opinions(self, agents: list | set | tuple | None = None, history_index: int = -1):
        """
        Compute the mean opinion across a set of agents.

        Parameters
        ----------
        agents : list | set | tuple | None, optional
            Collection of agents (IDs or names). If None, uses all agents.
        history_index : int, optional
            Index into opinion histories. Defaults to ``-1`` (latest).

        Returns
        -------
        float
            Mean of the selected agents' opinions.

        Raises
        ------
        ValueError
            If any specified agent does not exist.
        IndexError
            If any selected agent lacks opinion history or index is out of range.

        Notes
        -----
        Agents are resolved using :meth:`search`.
        """
        if agents is None:
            agent_ids = self.all_agent_ids
        else:
            agent_ids = []
            for agent in agents:
                agent_ids.append(self.search(agent))
        opinions = []
        for agent_id in agent_ids:
            opinion = self.get_agent_opinion(agent=agent_id, history_index=history_index)
            opinions.append(opinion)
        return np.array(opinions).mean()
    
    def average_opinions_history(self, agents: list | set | tuple | None = None):
        """
        Compute the mean opinion trajectory over time for a set of agents.

        Parameters
        ----------
        agents : list | set | tuple | None, optional
            Collection of agents (IDs or names). If None, uses all agents.

        Returns
        -------
        list[float]
            Time series of average opinion values, one per simulation step.

        Raises
        ------
        ValueError
            If any specified agent does not exist.
        IndexError
            If opinion histories are shorter than the organization opinion history.

        Notes
        -----
        The number of time steps is determined by ``len(self)``, which is defined as
        the length of the organization opinion history.
        """
        result = []
        steps = range(self.__len__())
        for step in steps:
            average_opinion = self.average_opinions(agents=agents, history_index=step)
            result.append(average_opinion)
        return result
    
    def __len__(self):
        """
        Return the number of recorded organization-level opinion steps.

        Returns
        -------
        int
            Length of the organization opinion history.
        """
        return len(self.get_organization_opinion_history())
            

if __name__ == "__main__":

    org = Organization()

    org.add_team(name="Team A")
    org.add_team(name="Team B")

    org.add_team_connection("Team A", "Team B")

    org.add_agent(name="Agent 1", team="Team A")
    org.add_agent(name="Agent 2", team="Team B")
    org.add_agent(name="Agent 3", team="Team B")

    org.add_agent_connection("Agent 2", "Agent 3")

    print(org.stat)
    #org.show_agents()
    #org.show_teams()