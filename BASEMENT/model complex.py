import numpy as np
import pandas as pd

from trustdynamics.organization import Organization, OrganizationalTrust
from trustdynamics.trust.degroot import Degroot


class Model:

    def __init__(
        self,
        organization: Organization,
        seed: int | None = None,
    ):
        self.rng = np.random.default_rng(seed)
        self.org = organization
        organizational_trust = OrganizationalTrust(organization=self.org, seed=seed)
        self.personal_trust = organizational_trust.personal_adjacency_dataframe()
        self.departmental_trust = organizational_trust.departmental_adjacency_dataframe()
        self.agents = self.personal_trust.index
        self.departments = self.departmental_trust.index
        self.opinions = pd.Series(
            self.rng.uniform(-1.0, 1.0, size=len(self.agents)),
            index=self.agents,
            name="opinion"
        )

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
        # Aggregate opinion within a department
        department_opinions = {}
        for department in self.departments:
            agents = self.org.agents(department)
            print(agents)
            opinions = # Extract the agents opinions from self.opinions
            trust_matrix = # Extract the agents trusts from self.personal_trust
            department_opinion = # Use degroot for update
            department_opinions[department] = department_opinion

        # Update the trust values within department

        # Aggregate opinion within all departments of the organization

        # Update the trust values within organizaion


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
    #print(model.personal_trust)
    #print(model.departmental_trust)
    #print(model.opinions)
    model.update()