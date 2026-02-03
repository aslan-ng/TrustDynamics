import numpy as np


class Stat:

    @property
    def stat(self) -> dict:
        """
        Return structural and opinion statistics for the organization graphs.
        """
        # --- basic counts ---
        total_teams = self.G_teams.number_of_nodes()
        total_agents = self.G_agents.number_of_nodes()

        # exclude self-loops; assume bidirectional edges
        total_team_connections = int(
            (self.G_teams.number_of_edges() - total_teams) / 2
        )
        total_agent_connections = int(
            (self.G_agents.number_of_edges() - total_agents) / 2
        )

        # --- agents per team ---
        agents_per_team = {
            team_id: len(self.agents_from_team(team_id))
            for team_id in self.all_team_ids
        }

        agent_counts = list(agents_per_team.values())

        # --- isolation ---
        isolated_agents = [
            a for a in self.all_agent_ids
            if len(self.agents_connected_to(a)) == 0
        ]
        isolated_teams = [
            t for t in self.all_team_ids
            if len(self.teams_connected_to(t)) == 0
        ]

        # --- latest opinions ---
        agent_latest = [
            self.get_agent_opinion(a)
            for a in self.all_agent_ids
            if self.get_agent_opinion(a) is not None
        ]
        team_latest = [
            self.get_team_opinion(t)
            for t in self.all_team_ids
            if self.get_team_opinion(t) is not None
        ]

        return {
            # existing
            "total_teams": total_teams,
            "total_agents": total_agents,
            "total_team_connections": total_team_connections,
            "total_agent_connections": total_agent_connections,

            # added: isolation
            "isolated_agents": len(isolated_agents),
            "isolated_teams": len(isolated_teams),

            # added: agents per team
            "max_agents_in_team": max(agent_counts) if agent_counts else 0,
            "min_agents_in_team": min(agent_counts) if agent_counts else 0,
            "avg_agents_per_team": float(np.mean(agent_counts)) if agent_counts else 0.0,

            # added: opinion statistics (latest step)
            "agent_opinion_mean": float(np.mean(agent_latest)) if agent_latest else None,
            "agent_opinion_std": float(np.std(agent_latest)) if agent_latest else None,
            "team_opinion_mean": float(np.mean(team_latest)) if team_latest else None,
            "team_opinion_std": float(np.std(team_latest)) if team_latest else None,
        }