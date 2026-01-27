import os
import json
import numpy as np
import pandas as pd

from trustdynamics.organization import Organization, OrganizationalTrust
from trustdynamics.degroot import Degroot


class Model:

    def __init__(
        self,
        organization: Organization,
        seed: int | None = None,
    ):
        self.rng = np.random.default_rng(seed)
        self.org = organization
        organizational_trust = OrganizationalTrust(organization=self.org, rng=self.rng)
        self.trusts = [
            self.row_stochastic(organizational_trust.adjacency_dataframe()),
        ]
        self.agents = self.trusts[-1].index
        self.opinions = [
            pd.Series(
                self.rng.uniform(-1.0, 1.0, size=len(self.agents)),
                index=self.agents,
                name="opinions"
            ),
        ]

    #def random(self, low: float = 0.0, high: float = 1.0):
    #    return self.rng.uniform(low, high)

    def row_stochastic(self, T: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
        """
        Return a row-stochastic version of matrix T (rows sum to 1).
        Does not modify T.
        """
        row_sums = T.sum(axis=1)
        row_sums = row_sums.where(row_sums > eps, 1.0)  # avoid divide-by-zero
        W = T.div(row_sums, axis=0)
        return W

    def update(self):
        # Aggregate opinions
        W = self.row_stochastic(self.trusts[-1])
        #print(W)
        degroot = Degroot(W)
        initial_opinions = self.opinions[-1]
        #print(initial_opinions)
        final_opinions = degroot.run_steps(initial_opinions)["final_opinions"]
        #print(final_opinions)
        self.opinions.append(final_opinions)

    def serialize(self):
        return {
            "organization": self.org.serialize(),

            "trusts": [
                {
                    "index": list(W.index),
                    "columns": list(W.columns),
                    "data": W.to_numpy().tolist(),
                }
                for W in self.trusts
            ],

            "opinions": [
                {
                    "index": list(s.index),
                    "name": s.name,
                    "data": s.to_numpy().tolist(),
                }
                for s in self.opinions
            ],

            "rng_state": self.rng.bit_generator.state,
        }
    
    def save(self, path):
        folderpath = os.path.join(path, "result")
        os.makedirs(folderpath, exist_ok=True)

        data = self.serialize()
        filepath = os.path.join(folderpath, f"{self.org.name}.json")

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        return filepath
    
    @classmethod
    def deserialize(cls, data: dict) -> "Model":
        """
        Reconstruct a Model from a serialized dictionary.
        """
        # Restore organization
        org = Organization()
        org.deserialize(data["organization"])

        # Create instance without __init__
        model = cls.__new__(cls)
        model.org = org

        # Restore RNG
        model.rng = np.random.default_rng()
        model.rng.bit_generator.state = data["rng_state"]

        # Restore trust matrices
        model.trusts = [
            pd.DataFrame(
                t["data"],
                index=t["index"],
                columns=t["columns"],
            )
            for t in data["trusts"]
        ]

        model.agents = model.trusts[-1].index

        # Restore opinions
        model.opinions = [
            pd.Series(
                o["data"],
                index=o["index"],
                name=o["name"],
            )
            for o in data["opinions"]
        ]

        return model

    @classmethod
    def load(cls, filepath: str) -> "Model":
        """
        Load a Model from a JSON file.
        """
        import json

        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"No such file: {filepath}")

        with open(filepath, "r") as f:
            data = json.load(f)

        return cls.deserialize(data)
    

if __name__ == "__main__":
    
    from trustdynamics.organization.generate import generate_organization
    
    seed = 42

    organization = generate_organization(
        n_departments=3,
        n_people=6,
        max_depth=2,
        seed=seed
    )
    model = Model(organization=organization, seed=seed)
    model.update()
    print(model.opinions)