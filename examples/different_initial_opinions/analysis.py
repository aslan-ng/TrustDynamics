from pathlib import Path
import matplotlib.pyplot as plt
from trustdynamics import Model

from run import average_initial_opinions


BASE_DIR = Path(__file__).resolve().parent
models_dir = BASE_DIR / "models"

plt.figure()

for average_initial_opinion in average_initial_opinions:
    model = Model.load(models_dir / f"model_{average_initial_opinion}.json")
    organization_opinion_history = model.organization.get_organization_opinion_history()
    steps = range(model.step)

    plt.plot(
        steps,
        organization_opinion_history,
        label=f"init = {average_initial_opinion:.2f}",
    )

plt.xlabel("Time step")
plt.ylabel("Opinion")
plt.title("Organization Opinion Over Time")
plt.legend(title="Average initial opinion")
plt.grid(True)
plt.tight_layout()

out_path = BASE_DIR / "figure.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()