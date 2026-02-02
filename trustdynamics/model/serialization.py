from __future__ import annotations
import json
from pathlib import Path
from typing import Self
import numpy as np

from trustdynamics.organization.organization import Organization


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

    @staticmethod
    def _rng_state_to_jsonable(state: dict) -> dict:
        """
        Convert a numpy RNG state dictionary into JSON-safe Python objects.

        Numpy bit generator states may contain:
        - numpy arrays (e.g., uint64 state vectors)
        - numpy scalar types (np.integer, np.floating)

        These are converted to built-in Python types (lists, ints, floats)
        so the result can be serialized with `json.dump`.
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
        Convert a JSON-safe RNG state back into numpy-compatible structures.

        This reverses `_rng_state_to_jsonable`:
        - lists are converted back into uint64 numpy arrays
        - scalar values are passed through unchanged

        The resulting dictionary can be assigned directly to
        `rng.bit_generator.state` to resume the RNG exactly.
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
        """
        Serialize the full model into a JSON-compatible dictionary.

        The output includes:
        - schema metadata (name + version)
        - model state flags (e.g., initialized)
        - full model configuration (hyperparameters)
        - RNG state for deterministic continuation
        - organization state (delegated)

        This dictionary is guaranteed to be consumable by `from_dict`.
        """
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

                # --- trust initialization bounds ---
                "agents_initial_trust_min": float(self.agents_initial_trust_min),
                "agents_initial_trust_max": float(self.agents_initial_trust_max),
                "teams_initial_trust_min": float(self.teams_initial_trust_min),
                "teams_initial_trust_max": float(self.teams_initial_trust_max),

                # --- trust adaptation parameters ---
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
        """
        Reconstruct a model from a dictionary produced by `to_dict`.

        Guarantees
        ----------
        - All model hyperparameters are restored exactly.
        - The organization state (graphs, opinions, trust, history) is restored.
        - If an RNG state is present, the simulation can continue deterministically
          from the exact point at which it was saved.

        Raises
        ------
        ValueError
            If the serialization version is incompatible.
        """
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

            # --- trust initialization bounds ---
            agents_initial_trust_min=cfg.get("agents_initial_trust_min", 0.01),
            agents_initial_trust_max=cfg.get("agents_initial_trust_max", 0.99),
            teams_initial_trust_min=cfg.get("teams_initial_trust_min", 0.01),
            teams_initial_trust_max=cfg.get("teams_initial_trust_max", 0.99),

            # --- trust adaptation parameters ---
            agents_self_trust_learning_rate=cfg["agents_self_trust_learning_rate"],
            agents_neighbor_trust_learning_rate=cfg["agents_neighbor_trust_learning_rate"],
            agents_homophily_normative_tradeoff=cfg["agents_homophily_normative_tradeoff"],
            teams_self_trust_learning_rate=cfg["teams_self_trust_learning_rate"],
            teams_neighbor_trust_learning_rate=cfg["teams_neighbor_trust_learning_rate"],
            teams_homophily_normative_tradeoff=cfg["teams_homophily_normative_tradeoff"],

            seed=None,
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
        Write the serialized model to a JSON file.

        Parent directories are created automatically if needed.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """
        Load a serialized model from a JSON file.
        """
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)