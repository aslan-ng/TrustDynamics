from __future__ import annotations

import json
from pathlib import Path
from typing import Self

from networkx.readwrite import json_graph


class Serialization:
    """
    Mixin providing versioned JSON serialization for an organization object.

    This mixin assumes the consuming class defines at least:

    Required attributes
    -------------------
    SERIALIZATION_VERSION : int
        Integer schema version used for compatibility checks.
    name : str
        Human-readable organization name.
    opinions : list[float]
        Organization-level opinion history.
    G_teams : networkx.Graph
        Team graph (typically a directed graph) storing node and edge attributes.
    G_agents : networkx.Graph
        Agent graph (typically a directed graph) storing node and edge attributes.

    Notes
    -----
    - Graph serialization uses NetworkX node-link format via
      :func:`networkx.readwrite.json_graph.node_link_data`.
    - Deserialization restores graphs using :func:`networkx.readwrite.json_graph.node_link_graph`.
    - A schema version is embedded to guard against incompatible structural changes.
    """

    SERIALIZATION_VERSION = 1
    SERIALIZATION_NAME = "trustdynamics.organization.Organization"

    def to_dict(self) -> dict:
        return {
            "schema": {
                "name": self.SERIALIZATION_NAME,
                "version": self.SERIALIZATION_VERSION,
            },
            "name": self.name,
            "opinions": list(self.opinions),
            "G_teams": json_graph.node_link_data(self.G_teams),
            "G_agents": json_graph.node_link_data(self.G_agents),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        schema = data.get("schema", {})
        version = schema.get("version", None)
        name = schema.get("name", None)

        if name not in (None, cls.SERIALIZATION_NAME):
            raise ValueError(f"Unsupported schema name: {name}")

        if version != cls.SERIALIZATION_VERSION:
            raise ValueError(
                f"Unsupported serialization version: {version}. "
                f"Expected {cls.SERIALIZATION_VERSION}."
            )

        org = cls(name=data.get("name", "Organization"))
        org.opinions = list(data.get("opinions", []))

        org.G_teams = json_graph.node_link_graph(data["G_teams"], directed=True)
        org.G_agents = json_graph.node_link_graph(data["G_agents"], directed=True)

        # -----------------------
        # Invariants & migrations
        # -----------------------

        def _ensure_self_loop_with_trusts(G):
            for n in G.nodes():
                if not G.has_edge(n, n):
                    G.add_edge(n, n, trusts=[])
                else:
                    G.edges[n, n].setdefault("trusts", [])

        def _ensure_edge_trusts_list(G):
            for u, v, attrs in G.edges(data=True):
                attrs.setdefault("trusts", [])
                # coerce to list (handles tuples / numpy arrays / None)
                attrs["trusts"] = list(attrs["trusts"]) if attrs["trusts"] is not None else []

        def _ensure_node_opinions_list(G):
            for n, attrs in G.nodes(data=True):
                attrs.setdefault("opinions", [])
                attrs["opinions"] = list(attrs["opinions"]) if attrs["opinions"] is not None else []

        # Team node defaults (add as you grow your API)
        TEAM_DEFAULTS = {
            "name": None,
            "opinions": [],
            "self_trust_learning_rate": 0.0,
            "trust_learning_rate": 0.0,
            "homophily_normative_tradeoff": 0.5,
        }

        # Agent node defaults
        AGENT_DEFAULTS = {
            "name": None,
            "team": None,
            "opinions": [],
            "technology_success_impact": 0.0,
            "technology_failure_impact": 0.0,
            "self_trust_learning_rate": 0.0,
            "trust_learning_rate": 0.0,
            "homophily_normative_tradeoff": 0.5,
        }

        def _apply_defaults(G, defaults: dict):
            for n, attrs in G.nodes(data=True):
                for k, v in defaults.items():
                    attrs.setdefault(k, v)

        def _coerce_types_team(attrs: dict):
            # Keep these as floats
            for k in ("self_trust_learning_rate", "trust_learning_rate", "homophily_normative_tradeoff"):
                attrs[k] = float(attrs.get(k, 0.0))
            # opinions already list, leave contents as-is (you can coerce if needed)

        def _coerce_types_agent(attrs: dict):
            for k in (
                "technology_success_impact",
                "technology_failure_impact",
                "self_trust_learning_rate",
                "trust_learning_rate",
                "homophily_normative_tradeoff",
            ):
                attrs[k] = float(attrs.get(k, 0.0))

        # Apply defaults
        _apply_defaults(org.G_teams, TEAM_DEFAULTS)
        _apply_defaults(org.G_agents, AGENT_DEFAULTS)

        # Ensure opinions lists exist
        _ensure_node_opinions_list(org.G_teams)
        _ensure_node_opinions_list(org.G_agents)

        # Ensure self-loops + trusts histories exist
        _ensure_self_loop_with_trusts(org.G_teams)
        _ensure_self_loop_with_trusts(org.G_agents)

        # Ensure every edge has trusts list
        _ensure_edge_trusts_list(org.G_teams)
        _ensure_edge_trusts_list(org.G_agents)

        # Coerce node attribute types
        for _, attrs in org.G_teams.nodes(data=True):
            _coerce_types_team(attrs)

        for _, attrs in org.G_agents.nodes(data=True):
            _coerce_types_agent(attrs)

        return org

    def save(self, path: str | Path) -> None:
        path = Path(path)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)