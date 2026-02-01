from __future__ import annotations
import json
from pathlib import Path
from typing import Self
import numpy as np

from trustdynamics.organization.organization import Organization


class Serialization:

    @staticmethod
    def _rng_state_to_jsonable(state: dict) -> dict:
        """
        Convert numpy RNG state to JSON-safe structures.
        """
        out = {}
        for k, v in state.items():
            if isinstance(v, np.ndarray):
                out[k] = v.tolist()
            elif isinstance(v, (np.integer, np.floating)):
                out[k] = v.item()
            else:
                out[k] = v
        return out

    @staticmethod
    def _rng_state_from_jsonable(state: dict) -> dict:
        """
        Convert JSON-safe RNG state back to numpy-compatible structures.
        """
        out = {}
        for k, v in state.items():
            # numpy bit_generator.state expects certain fields as arrays
            if isinstance(v, list):
                out[k] = np.array(v, dtype=np.uint64)
            else:
                out[k] = v
        return out

    def to_dict(self) -> dict:
        return {
            "schema": {
                "name": "trustdynamics.model.Model",
                "version": self.SERIALIZATION_VERSION,
            },
            "state": {
                "initialized": bool(self.initialized),
                # Optional, but strongly recommended so the loaded model’s config matches:
                "average_initial_opinion": float(self.average_initial_opinion),
                # Optional future-proofing (only if you add step later):
                # "step": int(getattr(self, "step", 0)),
            },
            "config": {
                "technology_success_rate": float(self.technology_success_rate),
                "tech_successful_delta": float(self.tech_successful_delta),
                "tech_failure_delta": float(self.tech_failure_delta),
                "agents_self_trust_learning_rate": float(self.agents_self_trust_learning_rate),
                "agents_neighbor_trust_learning_rate": float(self.agents_neighbor_trust_learning_rate),
                "agents_homophily_normative_tradeoff": float(self.agents_homophily_normative_tradeoff),
                "teams_self_trust_learning_rate": float(self.teams_self_trust_learning_rate),
                "teams_neighbor_trust_learning_rate": float(self.teams_neighbor_trust_learning_rate),
                "teams_homophily_normative_tradeoff": float(self.teams_homophily_normative_tradeoff),
            },
            "rng": {
                "bit_generator": self.rng.bit_generator.__class__.__name__,
                "state": self._rng_state_to_jsonable(self.rng.bit_generator.state),
            },
            "organization": self.organization.to_dict(),
        }


    @classmethod
    def from_dict(cls, data: dict) -> Self:
        schema = data.get("schema", {})
        version = schema.get("version", None)
        if version != cls.SERIALIZATION_VERSION:
            raise ValueError(
                f"Unsupported serialization version: {version}. "
                f"Expected {cls.SERIALIZATION_VERSION}."
            )

        cfg = data["config"]
        st = data.get("state", {})

        org = Organization.from_dict(data["organization"])

        model = cls(
            organization=org,
            technology_success_rate=cfg["technology_success_rate"],
            tech_successful_delta=cfg["tech_successful_delta"],
            tech_failure_delta=cfg["tech_failure_delta"],
            average_initial_opinion=st.get("average_initial_opinion", 0.0),
            agents_self_trust_learning_rate=cfg["agents_self_trust_learning_rate"],
            agents_neighbor_trust_learning_rate=cfg["agents_neighbor_trust_learning_rate"],
            agents_homophily_normative_tradeoff=cfg["agents_homophily_normative_tradeoff"],
            teams_self_trust_learning_rate=cfg["teams_self_trust_learning_rate"],
            teams_neighbor_trust_learning_rate=cfg["teams_neighbor_trust_learning_rate"],
            teams_homophily_normative_tradeoff=cfg["teams_homophily_normative_tradeoff"],
            seed=0,
        )

        # Restore state
        model.initialized = bool(st.get("initialized", False))
        # Optional:
        # model.step = int(st.get("step", 0))

        # Restore RNG state (exact continuation)
        rng_blob = data.get("rng", None)
        if rng_blob is not None and "state" in rng_blob:
            model.rng.bit_generator.state = cls._rng_state_from_jsonable(rng_blob["state"])

        return model
    
    def save(self, path: str | Path) -> None:
        """
        Write the model to a JSON file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """
        Read the model from a JSON file.
        """
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)