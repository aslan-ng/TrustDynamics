import numpy as np


class Technology:

    def __init__(
        self,
        name: str = "Technology",
        success_rate: float = 1.0,
        seed: int | None | np.random.Generator = None,
    ):
        """
        Models the technology behavior.
        
        Parameters
        ----------

        success_rate : float, optional
            Probability an agent's technology interaction succeeds at each step.
            Must lie in [0, 1].
        """
        self.name = name

        if success_rate < 0.0 or success_rate > 1.0:
            raise ValueError("success_rate must be between 0.0 and 1.0")
        self.success_rate = success_rate

        # Random generator
        if isinstance(seed, int) or seed is None:
            self.rng = np.random.default_rng(seed)
        elif isinstance(seed, np.random.Generator):
            self.rng = seed

        self.model = None

    def use(self, agent_id: int | None = None) -> bool | None:
        """
        Use technology to see the result
        """
        def generate() -> bool:
            tech_successful: bool = self.rng.random() < self.success_rate
            return tech_successful
        if agent_id is None:
            return generate()
        access = self.model.organization.get_agent_exposure_to_technology(agent_id)
        if access is True:
            tech_successful: bool = self.rng.random() < self.success_rate
            return tech_successful
        else:
            return None
        
    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    SERIALIZATION_VERSION = 1

    def to_dict(self) -> dict:
        """
        Serialize technology to a JSON-safe dict.

        Notes
        -----
        - Does NOT serialize the attached model.
        - Preserves RNG state for reproducibility.
        """
        return {
            "schema": {
                "name": "trustdynamics.technology.Technology",
                "version": self.SERIALIZATION_VERSION,
            },
            "name": self.name,
            "success_rate": self.success_rate,
            "rng_state": self.rng.bit_generator.state,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Technology":
        """
        Deserialize technology from a dict produced by `to_dict()`.
        """
        schema = data.get("schema", {})
        version = schema.get("version")

        if version != cls.SERIALIZATION_VERSION:
            raise ValueError(
                f"Unsupported Technology serialization version: {version}"
            )

        tech = cls(
            name=data["name"],
            success_rate=data["success_rate"],
            seed=None,  # overwritten by RNG state
        )

        tech.rng.bit_generator.state = data["rng_state"]
        return tech
    

if __name__ == "__main__":
    tech = Technology(success_rate=0.5)
    print(tech.use())
    print(tech.use())