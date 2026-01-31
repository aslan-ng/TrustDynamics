import networkx as nx

from trustdynamics.organization.graphics import Graphics
from trustdynamics.utils.new_id import new_unique_id


class Organization(Graphics):

    def __init__(self, name: str = "Organization"):
        self.name = name
        self.G_teams = nx.DiGraph()
        self.G_agents = nx.DiGraph()
        self.opinions = [] # History of orgnization aggregate opinions

    @property
    def all_team_ids(self) -> set:
        return set(self.G_teams.nodes())
    
    @property
    def all_team_names(self) -> set:
        names = set()
        for _, attrs in self.G_teams.nodes(data=True):
            if attrs.get("name") is not None and attrs.get("name") != "":
                names.add(attrs.get("name"))
        return names
    
    @property
    def all_agent_ids(self) -> set:
        return set(self.G_agents.nodes())

    @property
    def all_agent_names(self) -> set:
        names = set()
        for _, attrs in self.G_agents.nodes(data=True):
            if attrs.get("name") is not None and attrs.get("name") != "":
                names.add(attrs.get("name"))
        return names

    @property
    def all_ids(self) -> set:
        return self.all_team_ids.union(self.all_agent_ids)
    
    @property
    def all_names(self) -> set:
        return self.all_team_names.union(self.all_agent_names)
    
    def agents_from_team(self, team: int | str | None) -> set:
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
        agent_id = self.search(agent)
        if agent_id is None:
            raise ValueError("Agent must exist in the organization")
        return self.G_agents.nodes[agent_id].get("team", None)
    
    def _is_name_unique(self, name: str | None) -> bool:
        names = self.all_names
        if name is None:
            return True
        for existing_name in names:
            if existing_name == name:
                return False
        return True

    def add_team(self, name: str | None = None):
        if not self._is_name_unique(name):
            raise ValueError(f"Team name must be unique in the organization. '{name}' already exists.")
        team_id = new_unique_id(existing_values=self.all_ids)
        self.G_teams.add_node(
            team_id,
            name=name,
            opinions=[], # History of team opinions
        )
        self.G_teams.add_edge(
            team_id,
            team_id,
            trusts=[], # History of self-trust values
        )

    def add_agent(self, name: str | None = None, team: int | str = None):
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
        self.G_agents.add_node(
            agent_id,
            name=name,
            team=team_id,
            opinions=[], # History of agent opinions
        )
        self.G_agents.add_edge(
            agent_id,
            agent_id,
            trusts=[], # History of self-trust values
        )

    def add_agent_connection(self, agent_1: int | str, agent_2: int | str):
        agent_1_id = self.search(agent_1)
        agent_2_id = self.search(agent_2)
        agent_1_team_id = self.agent_team_id(agent_1_id)
        agent_2_team_id = self.agent_team_id(agent_2_id)
        if agent_1_team_id is None or agent_2_team_id is None or agent_1_team_id != agent_2_team_id:
            raise ValueError("Both agents must belong to the same team.")
        if agent_1_id is not None and agent_2_id is not None:
            self.G_agents.add_edge(
                agent_1_id,
                agent_2_id,
                trusts=[], # History of trust values
            )
            self.G_agents.add_edge(
                agent_2_id,
                agent_1_id,
                trusts=[], # History of trust values
            )
        else:
            raise ValueError("Both agents must exist in the organization to add a connection.")

    def add_team_connection(self, team_1: int | str, team_2: int | str):
        team_1_id = self.search(team_1)
        team_2_id = self.search(team_2)
        if team_1_id is not None and team_2_id is not None:
            self.G_teams.add_edge(
                team_1_id,
                team_2_id,
                trusts=[], # History of trust values
            )
            self.G_teams.add_edge(
                team_2_id,
                team_1_id,
                trusts=[], # History of trust values
            )
        else:
            raise ValueError("Both teams must exist in the organization to add a connection.")
        
    def search(self, input: int | str) -> int | None:
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

    @property
    def stat(self) -> dict:
        return {
            "total_teams": self.G_teams.number_of_nodes(),
            "total_agents": self.G_agents.number_of_nodes(),
            "total_team_connections": int((self.G_teams.number_of_edges() - self.G_teams.number_of_nodes()) / 2), # exclude self-loop
            "total_agent_connections": int((self.G_agents.number_of_edges() - self.G_agents.number_of_nodes()) / 2), # exclude self-loop
        }
    
    def get_agent_opinion(self, agent: int | str) -> float:
        """
        Get agent latest opinion.
        """
        opinions = self.get_agent_opinions_history(agent)
        return opinions[-1]
    
    def set_agent_opinion(self, agent: int | str, opinion: float):
        """
        Set agent latest opinion.
        """
        agent_id = self.search(agent)
        if agent_id is None:
            raise ValueError("Agent must exist in the organization to set opinion.")
        self.G_agents.nodes[agent_id]["opinions"].append(opinion)

    def get_agent_opinions_history(self, agent: int | str) -> list:
        """
        Get agent opinions history.
        """
        agent_id = self.search(agent)
        if agent_id is None:
            raise ValueError("Agent must exist in the organization to get opinion.")
        return self.G_agents.nodes[agent_id].get("opinions", [])

    def get_team_opinion(self, team: int | str) -> float:
        """
        Get team latest opinion.
        """
        opinions = self.get_team_opinions_history(team)
        return opinions[-1]
    
    def set_team_opinion(self, team: int | str, opinion: float):
        """
        Set team latest opinion.
        """
        team_id = self.search(team)
        if team_id is None:
            raise ValueError("Team must exist in the organization to set opinion.")
        self.G_teams.nodes[team_id]["opinions"].append(opinion)

    def get_team_opinions_history(self, team: int | str) -> list:
        """
        Get team opinions history.
        """
        team_id = self.search(team)
        if team_id is None:
            raise ValueError("Team must exist in the organization to get opinion.")
        return self.G_teams.nodes[team_id].get("opinions", [])
    
    def get_organization_opinion(self) -> float:
        """
        Get organization latest opinion.
        """
        return self.opinions[-1]
    
    def set_organization_opinion(self, opinion: float):
        """
        Set organization latest opinion.
        """
        self.opinions.append(opinion)

    def get_organization_opinion_history(self) -> list:
        """
        Get organization opinions history.
        """
        return self.opinions
    
    def get_agent_trust(self, agent_1: int | str, agent_2: int | str) -> float:
        """
        Get latest trust value from agent_1 to agent_2.
        """
        trust_history = self.get_agent_trust_history(agent_1, agent_2)
        return trust_history[-1]
    
    def set_agent_trust(self, agent_1: int | str, agent_2: int | str, influece: float):
        """
        Set latest trust value from agent_1 to agent_2.
        """
        agent_1_id = self.search(agent_1)
        agent_2_id = self.search(agent_2)
        if agent_1_id is None or agent_2_id is None:
            raise ValueError("Both agents must exist in the organization to get trust value.")
        if self.G_agents.has_edge(agent_1_id, agent_2_id):
            self.G_agents.edges[agent_1_id, agent_2_id]["trusts"].append(influece)
        else:
            raise ValueError("No connection exists between the specified agents.")
        
    def get_agent_trust_history(self, agent_1: int | str, agent_2: int | str) -> list:
        """
        Get trust history from agent_1 to agent_2.
        """
        agent_1_id = self.search(agent_1)
        agent_2_id = self.search(agent_2)
        if agent_1_id is None or agent_2_id is None:
            raise ValueError("Both agents must exist in the organization to get trust value.")
        if self.G_agents.has_edge(agent_1_id, agent_2_id):
            return self.G_agents.edges[agent_1_id, agent_2_id].get("trusts", [])
        else:
            raise ValueError("No connection exists between the specified agents.")

    def get_team_trust(self, team_1: int | str, team_2: int | str) -> list:
        """
        Get latest trust from team_1 to team_2.
        """
        trust_history = self.get_team_trust_history(team_1, team_2)
        return trust_history[-1]
    
    def set_team_trust(self, team_1: int | str, team_2: int | str, trust: float):
        """
        Set latest trust from team_1 to team_2.
        """
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
        Get trust history from team_1 to team_2.
        """
        team_1_id = self.search(team_1)
        team_2_id = self.search(team_2)
        if team_1_id is None or team_2_id is None:
            raise ValueError("Both teams must exist in the organization to get trust value.")
        if self.G_teams.has_edge(team_1_id, team_2_id):
            return self.G_teams.edges[team_1_id, team_2_id].get("trusts", [])
        else:
            raise ValueError("No connection exists between the specified teams.")


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
    org.show_agents()
    #org.show_teams()