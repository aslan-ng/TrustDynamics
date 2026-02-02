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

    def to_dict(self) -> dict:
        """
        Serialize the organization to a JSON-safe Python dictionary.

        The output is intended to be round-trippable via :meth:`from_dict` and
        safe to store using :func:`json.dump`.

        Returns
        -------
        dict
            JSON-safe dictionary containing:
            - schema metadata (name + version),
            - organization name,
            - organization-level opinion history,
            - team graph in node-link format,
            - agent graph in node-link format.

        Notes
        -----
        - Uses NetworkX node-link format for graphs.
        - Preserves node and edge attributes (including opinion/trust histories).
        - Includes a schema version for forward compatibility checks.
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
        Deserialize an organization from a dictionary produced by :meth:`to_dict`.

        Parameters
        ----------
        data : dict
            Dictionary created by :meth:`to_dict`.

        Returns
        -------
        Self
            A newly constructed organization instance with restored graphs and histories.

        Raises
        ------
        ValueError
            If the embedded schema version does not match ``cls.SERIALIZATION_VERSION``.
        KeyError
            If required keys (e.g., ``"G_teams"``, ``"G_agents"``) are missing.

        Notes
        -----
        After loading graphs, this method applies normalization/safety defaults:
        - Ensures self-loops exist for every team/agent node (with ``trusts`` history).
        - Ensures expected node attributes exist:
          - teams: ``name``, ``opinions``
          - agents: ``name``, ``team``, ``opinions``

        If your model assumes additional invariants, this is the correct place to
        enforce them.
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
        Write the organization to disk as a JSON file.

        Parameters
        ----------
        path : str | pathlib.Path
            Destination file path.

        Returns
        -------
        None

        Notes
        -----
        The file content is generated by :meth:`to_dict` and written with:
        - UTF-8 encoding
        - ``ensure_ascii=False`` (preserves unicode)
        - ``indent=2`` (human-readable formatting)

        Raises
        ------
        OSError
            If the file cannot be written (e.g., permissions, invalid path).
        TypeError
            If the object contains non-JSON-serializable content.
        """
        path = Path(path)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """
        Read an organization from a JSON file on disk.

        Parameters
        ----------
        path : str | pathlib.Path
            Path to the JSON file.

        Returns
        -------
        Self
            Deserialized organization instance.

        Raises
        ------
        OSError
            If the file cannot be read (e.g., missing file, permissions).
        json.JSONDecodeError
            If the file is not valid JSON.
        ValueError
            If the schema version is unsupported (raised by :meth:`from_dict`).
        KeyError
            If required keys are missing in the JSON payload (raised by :meth:`from_dict`).

        See Also
        --------
        from_dict : Constructs an instance from an in-memory dictionary.
        save : Writes an instance to disk.
        """
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)