from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from trustdynamics import Model


BASE_DIR = Path(__file__).resolve().parent
model = Model.load(BASE_DIR / "model.json")

# Organization opinion history
organization_opinion_history = model.organization.get_organization_opinion_history()

# Teams opinion history
teams_opinion_history = []
team_ids = list(model.organization.all_team_ids)
for team_id in team_ids:
    team_opinion_history = model.organization.get_team_opinions_history(team=team_id)
    teams_opinion_history.append(team_opinion_history)

steps = range(model.step)

plt.figure()
plt.plot(
    steps,
    organization_opinion_history,
    color="black",
    linewidth=3,
    label="Organization"
)

# Teams (same color family)
cmap = plt.cm.Blues
colors = cmap(np.linspace(0.4, 0.9, len(team_ids)))

for team_id, history, color in zip(team_ids, teams_opinion_history, colors):
    plt.plot(
        steps,
        history,
        color=color,
        alpha=0.8,
        linewidth=1.5,
        label=f"Team {team_id}"
    )

plt.xlabel("Time step")
plt.ylabel("Opinion")
plt.title("Organization and Team Opinions Over Time")
plt.grid(True, alpha=0.3)

# Optional: comment this out if too crowded
plt.legend(fontsize=8, ncol=2)

plt.tight_layout()
out_path = BASE_DIR / "figure.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()