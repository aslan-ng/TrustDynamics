from __future__ import annotations
import json
from pathlib import Path
from typing import Self
from networkx.readwrite import json_graph


class Serialization:

    def to_dict(self) -> dict:
        """
        Serialize the organization to a JSON-safe Python dict.

        Notes
        -----
        - Uses NetworkX node-link format for graphs.
        - Preserves all node/edge attributes (including opinion/trust histories).
        - Includes a schema version for forward compatibility.
        """
        data = {
            "schema": {
                "name": "trustdynamics.organization.Organization",
                "version": self.SERIALIZATION_VERSION,
            },
            "name": self.name,
            "opinions": list(self.opinions),
            "G_teams": json_graph.node_link_data(self.G_teams),
            "G_agents": json_graph.node_link_data(self.G_agents),
        }
        return data

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """
        Deserialize an organization from a dict produced by `to_dict()`.
        """
        schema = data.get("schema", {})
        version = schema.get("version", None)
        if version != cls.SERIALIZATION_VERSION:
            raise ValueError(
                f"Unsupported serialization version: {version}. "
                f"Expected {cls.SERIALIZATION_VERSION}."
            )

        org = cls(name=data.get("name", "Organization"))
        org.opinions = list(data.get("opinions", []))

        org.G_teams = json_graph.node_link_graph(data["G_teams"], directed=True)
        org.G_agents = json_graph.node_link_graph(data["G_agents"], directed=True)

        # --- optional safety checks / normalization ---
        # Ensure self-loops exist (if your model assumes them)
        for t in org.G_teams.nodes():
            if not org.G_teams.has_edge(t, t):
                org.G_teams.add_edge(t, t, trusts=[])
            else:
                org.G_teams.edges[t, t].setdefault("trusts", [])

        for a in org.G_agents.nodes():
            if not org.G_agents.has_edge(a, a):
                org.G_agents.add_edge(a, a, trusts=[])
            else:
                org.G_agents.edges[a, a].setdefault("trusts", [])

        # Ensure expected node attrs exist
        for t, attrs in org.G_teams.nodes(data=True):
            attrs.setdefault("name", None)
            attrs.setdefault("opinions", [])

        for a, attrs in org.G_agents.nodes(data=True):
            attrs.setdefault("name", None)
            attrs.setdefault("team", None)
            attrs.setdefault("opinions", [])

        return org

    def save(self, path: str | Path) -> None:
        """
        Write the organization to a JSON file.
        """
        path = Path(path)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """
        Read the organization from a JSON file.
        """
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)