from trustdynamics.organization import Organization, OrganizationTrust


class Model:

    def __init__(
        self,
        organization: Organization,
        seed: int | None = None,
    ):
        self.org = organization
        self.trust = OrganizationTrust(organization=self.org, seed=seed).adjacency_dataframe()


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
    print(model.trust)