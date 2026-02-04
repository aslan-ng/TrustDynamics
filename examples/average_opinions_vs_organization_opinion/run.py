from copy import deepcopy
from pathlib import Path
from trustdynamics import Model, Organization, Technology


seed = 42
technology_success_rates = [0.6, 0.7, 0.8, 0.9]


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
    org.initialize(seed=seed)

    BASE_DIR = Path(__file__).resolve().parent

    for i, technology_success_rate in enumerate(technology_success_rates):
        tech = Technology(success_rate=technology_success_rate, seed=seed)
        model = Model(
            organization=deepcopy(org),
            technology=tech,
        )
        model.run(steps=100)        
        model.save(BASE_DIR / "models" / f"model_{i}.json")