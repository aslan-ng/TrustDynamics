import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import colorsys


class Graphics:

    def agents_to_plt(
        self,
        ax=None,
        *,
        pos: dict | None = None,
        seed: int = 0,
        iterations: int = 200,
        k: float | None = None,
        agent_node_size: int = 420,
        edge_alpha: float = 0.25,
        edge_width: float = 1.4,
        with_labels: bool = True,
    ):
        """
        Plot ONLY agents and agent-agent connections.

        - Agents are colored by their team (one random-ish color per team).
        - Teams are NOT drawn.
        - Returns (fig, ax, pos) so pos can be reused for animation frames.
        """

        # --- choose / create axis ---
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 7))
        else:
            fig = ax.figure

        ax.set_aspect("equal")
        ax.axis("off")

        G = self.G_agents

        # --- positions (reusable) ---
        if pos is None:
            pos = nx.spring_layout(G, seed=seed, iterations=iterations, k=k)

        # --- team -> stable random color ---
        def _team_color(team_id):
            # stable across calls for same team_id and seed
            rng = np.random.default_rng(abs(hash((int(team_id), seed))) % (2**32))
            h = rng.uniform(0.0, 1.0)
            s = rng.uniform(0.45, 0.85)
            v = rng.uniform(0.70, 0.95)
            return colorsys.hsv_to_rgb(h, s, v)

        default_color = (0.55, 0.55, 0.55)

        # Build node colors by team
        node_colors = []
        for node_id, attrs in G.nodes(data=True):
            team_id = attrs.get("team")
            if team_id is None:
                node_colors.append(default_color)
            else:
                node_colors.append(_team_color(team_id))

        # --- draw edges ---
        if G.number_of_edges() > 0:
            nx.draw_networkx_edges(
                G, pos, ax=ax,
                alpha=edge_alpha,
                width=edge_width,
            )

        # --- draw nodes ---
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_size=agent_node_size,
            node_color=node_colors,
            edgecolors="white",
            linewidths=1.0,
        )

        # --- labels ---
        if with_labels:
            labels = {}
            for node_id, attrs in G.nodes(data=True):
                name = attrs.get("name")
                labels[node_id] = name if (name is not None and name != "") else str(node_id)

            if labels:
                nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)

        ax.set_title(f"{self.name} (Agents)")

        return fig, ax, pos
    
    def show_agents(self, **kwargs):
        fig, ax, _ = self.agents_to_plt(**kwargs)
        import matplotlib.pyplot as plt
        plt.show()
        return fig, ax
    
    def teams_to_plt(
        self,
        ax=None,
        *,
        pos: dict | None = None,
        seed: int = 0,
        iterations: int = 200,
        k: float | None = None,
        team_node_size: int = 900,
        edge_alpha: float = 0.35,
        edge_width: float = 2.0,
        with_labels: bool = True,
        use_random_colors: bool = True,
    ):
        """
        Plot ONLY teams and team-team connections from self.G_teams.
        Ignores agents and agent connections completely.

        Returns:
            (fig, ax, pos) where pos can be reused for animations.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import colorsys

        # --- choose / create axis ---
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 7))
        else:
            fig = ax.figure

        ax.set_aspect("equal")
        ax.axis("off")

        G = self.G_teams

        # --- positions (reusable) ---
        if pos is None:
            pos = nx.spring_layout(G, seed=seed, iterations=iterations, k=k)

        # --- stable random color per team node (optional) ---
        def _team_color(team_id):
            rng = np.random.default_rng(abs(hash((int(team_id), seed))) % (2**32))
            h = rng.uniform(0.0, 1.0)
            s = rng.uniform(0.45, 0.85)
            v = rng.uniform(0.70, 0.95)
            return colorsys.hsv_to_rgb(h, s, v)

        default_color = (0.55, 0.55, 0.55)

        nodelist = list(G.nodes())
        if use_random_colors:
            node_colors = [_team_color(tid) for tid in nodelist]
        else:
            node_colors = [default_color for _ in nodelist]

        # --- draw edges ---
        if G.number_of_edges() > 0:
            nx.draw_networkx_edges(
                G, pos, ax=ax,
                alpha=edge_alpha,
                width=edge_width,
            )

        # --- draw nodes ---
        if G.number_of_nodes() > 0:
            nx.draw_networkx_nodes(
                G, pos, ax=ax,
                nodelist=nodelist,
                node_size=team_node_size,
                node_color=node_colors,
                edgecolors="black",
                linewidths=1.0,
            )

        # --- labels ---
        if with_labels and G.number_of_nodes() > 0:
            labels = {}
            for team_id, attrs in G.nodes(data=True):
                name = attrs.get("name")
                labels[team_id] = name if (name is not None and name != "") else str(team_id)

            nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight="bold", ax=ax)

        # Title (if Organization has .name)
        ax.set_title(f"{getattr(self, 'name', 'Organization')} (Teams)")

        return fig, ax, pos


    def show_teams(self, **kwargs):
        """
        Convenience snapshot. Uses teams_to_plt and calls plt.show().
        """
        fig, ax, _ = self.teams_to_plt(**kwargs)
        import matplotlib.pyplot as plt
        plt.show()
        return fig, ax