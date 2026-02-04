from pathlib import Path
from trustdynamics import Model, Organization, Technology


seed = 42

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
org.initialize(seed=42)

tech = Technology(success_rate=0.8, seed=seed)

model = Model(organization=org, technology=tech)
model.run(steps=100)

BASE_DIR = Path(__file__).resolve().parent
model.save(BASE_DIR / "model.json")