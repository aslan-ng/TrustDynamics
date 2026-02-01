import numpy as np
from trustdynamics.organization import Organization


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


if __name__ == "__main__":
    print(org.stat)
    org.show_agents()
    #org.show_teams()