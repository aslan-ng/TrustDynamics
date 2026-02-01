from pathlib import Path
import matplotlib.pyplot as plt
from trustdynamics import Model


BASE_DIR = Path(__file__).resolve().parent
model = Model.load(BASE_DIR / "model.json")
organization_opinion_history = model.organization.get_organization_opinion_history()

plt.figure()
plt.plot(organization_opinion_history)
plt.xlabel("Time step")
plt.ylabel("Organization opinion")
plt.title("Organization Opinion Over Time")
plt.grid(True)
plt.tight_layout()
plt.show()