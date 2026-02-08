from copy import deepcopy
from pathlib import Path
from trustdynamics import Model, Organization, Technology


seed = 42
agents_average_initial_opinions = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]


if __name__ == "__main__":
    org = Organization()
    org.add_team(name="Team A")
    org.add_team(name="Team B")
    org.add_team_connection("Team A", "Team B")
    org.add_agent(name="Agent 1", team="Team A")
    org.add_agent(name="Agent 2", team="Team B")
    org.add_agent(name="Agent 3", team="Team B")
    org.add_agent(name="Agent 4", team="Team B")
    org.add_agent(name="Agent 5", team="Team B")
    org.add_agent_connection("Agent 2", "Agent 3")
    org.add_agent_connection("Agent 3", "Agent 4")
    org.add_agent_connection("Agent 2", "Agent 4")
    org.add_agent_connection("Agent 2", "Agent 5")

    BASE_DIR = Path(__file__).resolve().parent
    models_dir = BASE_DIR / "models"

    for agents_average_initial_opinion in agents_average_initial_opinions:
        organization = deepcopy(org)
        organization.initialize(
            agents_average_initial_opinion=agents_average_initial_opinion,
            seed=seed
        )
        technoloty = Technology(success_rate=0.8, seed=seed)
        model = Model(
            organization=organization,
            technology=technoloty
        )
        model.run(steps=200)
        model.save(models_dir / f"model_{agents_average_initial_opinion}.json")