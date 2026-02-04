from __future__ import annotations
import json
from pathlib import Path
from typing import Self

from trustdynamics.organization.organization import Organization
from trustdynamics.technology.technology import Technology


class Serialization:
    """
    Mixin class providing JSON serialization/deserialization for `Model`.

    Responsibilities
    ----------------
    - Serialize *all* model hyperparameters required to reproduce dynamics.
    - Serialize the full organization state (delegated to `Organization`).
    - Persist and restore the numpy RNG state exactly, enabling deterministic
      continuation of simulations from saved checkpoints.

    Design principles
    -----------------
    - Explicit is better than implicit: all non-derived configuration parameters
      are stored verbatim.
    - Determinism is achieved via RNG *state* serialization, not seeds.
    - Backward compatibility is maintained via schema versioning and defaults.

    Notes
    -----
    - The `Organization` object owns serialization of graphs, trust values,
      opinions, and opinion histories; this class simply delegates to it.
    - The model "step" counter is *not* stored here, because `Model.step` is
      derived from the organization’s stored opinion history length.
    """


    SERIALIZATION_VERSION = 1
    SERIALIZATION_NAME = "trustdynamics.model.Model"

    def to_dict(self) -> dict:
        return {
            "schema": {
                "name": self.SERIALIZATION_NAME,
                "version": self.SERIALIZATION_VERSION,
            },
            "organization": self.organization.to_dict(),
            "technology": self.technology.to_dict(),
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

        # --- Deserialize components ---
        org_data = data.get("organization", None)
        tech_data = data.get("technology", None)

        if org_data is None:
            raise KeyError("Missing required key: 'organization'")
        if tech_data is None:
            raise KeyError("Missing required key: 'technology'")

        organization = Organization.from_dict(org_data)
        technology = Technology.from_dict(tech_data)

        # --- Construct model ---
        model = cls(organization=organization, technology=technology)

        # --- Re-bind runtime backreferences (CRITICAL) ---
        model.technology.model = model

        return model

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