import numpy as np

from trustdynamics.model.update import Update
from trustdynamics.model.serialization import Serialization
from trustdynamics.organization import Organization
from trustdynamics.technology import Technology


class Model(
    Update,
    Serialization
):
    """
    Core simulation model for trust and opinion dynamics in a multi-level organization.

    The model is hierarchical with three coupled layers:
    1) Agents within teams (micro layer)
    2) Teams within an organization (meso layer)
    3) Organization-wide aggregate opinion (macro layer)

    At each step:
    - Opinions propagate via consensus influence dynamics within teams and across teams.
    - Trust adapts via a convex combination of:
        * homophily (agreement with peers), and
        * normative alignment (agreement with the higher-level reference belief).
    - Agents experience an exogenous stochastic "technology shock" that shifts opinions.
    """

    def __init__(
        self,
        organization: Organization,
        technology: Technology,
    ):
        """
        Initialize the trust–opinion dynamics model.

        Parameters
        ----------
        organization : Organization
            Organizational state and topology. Contains:
            - agent graph and team graph (directed)
            - opinions and trust values stored on nodes/edges (via setters/getters)
            - methods to compute influence matrices for DeGroot updates

        technology : Technology
            Technology that agents use.
            
        seed : int or None, optional
            Seed used to initialize the model RNG. If None, randomness is not seeded.
            The same RNG is passed into initialization routines to ensure end-to-end
            reproducibility across opinion sampling and stochastic technology outcomes.
        """
        # Organization
        self.organization = organization

        # Technology
        self.technology = technology
        technology.model = self # binding
        
    @property
    def step(self):
        """
        Return the number of recorded organization-level opinion steps.

        Returns
        -------
        int
            Length of the organization opinion history.
        """
        return self.organization.__len__()


if __name__ == "__main__":
    
    from trustdynamics.organization.samples import organization_0 as organization
    from trustdynamics.technology import Technology

    """
    organization.add_team(name="Team C")
    organization.add_team(name="Team D")
    organization.add_team_connection("Team A", "Team C")
    organization.add_team_connection("Team B", "Team C")
    organization.add_team_connection("Team A", "Team D")
    """
    organization.initialize(seed=42)
    technology = Technology(success_rate=0.9, seed=42)
    
    model = Model(
        organization=organization,
        technology=technology,
    )
    """
    print(model.organization.get_team_trust(team_1="Team A", team_2="Team A"))
    print(model.organization.get_team_trust(team_1="Team B", team_2="Team B"))
    print(model.organization.get_team_trust(team_1="Team D", team_2="Team D"))
    """
    #organization.show_teams()
    #model.update()
    model.run(10)
    print(model.organization.get_organization_opinion_history())