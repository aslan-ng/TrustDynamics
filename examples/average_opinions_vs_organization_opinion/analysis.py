from pathlib import Path
import matplotlib.pyplot as plt
from trustdynamics import Model


BASE_DIR = Path(__file__).resolve().parent
model = Model.load(BASE_DIR / "model.json")

organization_opinion_history = model.organization.get_organization_opinion_history()
average_opinions_history = model.organization.average_opinions_history()
steps = range(model.step)

# Plot both on the same figure
plt.figure()
plt.plot(steps, organization_opinion_history, label="Organization opinion")
plt.plot(steps, average_opinions_history, label="Average agent opinion")
plt.xlabel("Time step")
plt.ylabel("Opinion")
plt.title("Organization vs. Average Agent Opinion Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
out_path = BASE_DIR / "figure.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()
