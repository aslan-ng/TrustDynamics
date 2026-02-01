from pathlib import Path
import matplotlib.pyplot as plt
from trustdynamics import Model
from run import technology_success_rates  # wherever this is defined


BASE_DIR = Path(__file__).resolve().parent

plt.figure()

for i, technology_success_rate in enumerate(technology_success_rates):
    model = Model.load(BASE_DIR / "models" / f"model_{i}.json")
    history = model.organization.get_organization_opinion_history()
    plt.plot(
        history,
        label=f"tech success rate = {technology_success_rate}"
    )

plt.xlabel("Time step")
plt.ylabel("Organization opinion")
plt.title("Organization Opinion vs. Technology Success Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
out_path = BASE_DIR / "figure.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()