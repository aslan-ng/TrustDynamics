from pathlib import Path
import matplotlib.pyplot as plt

from trustdynamics import Model
from config import BASE_DIR

MODELS_DIR = BASE_DIR / "models"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def load_model(model_name: str) -> Model:
    model_path = MODELS_DIR / f"{model_name}.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return Model.load(model_path)


def analyze_trust(model_name: str):
    model = load_model(model_name)
    org = model.organization

    trust_to_homophilic = org.get_agent_trust_history(
        "Main Agent",
        "Homophilic Agent",
    )

    trust_to_normative = org.get_agent_trust_history(
        "Main Agent",
        "Normative Agent",
    )

    main_self_trust = org.get_agent_trust_history(
        "Main Agent",
        "Main Agent",
    )

    main_opinion = org.get_agent_opinions_history("Main Agent")
    homophilic_opinion = org.get_agent_opinions_history("Homophilic Agent")
    normative_opinion = org.get_agent_opinions_history("Normative Agent")

    trust_min_len = min(
        len(trust_to_homophilic),
        len(trust_to_normative),
        len(main_self_trust),
    )

    opinion_min_len = min(
        len(main_opinion),
        len(homophilic_opinion),
        len(normative_opinion),
    )

    trust_to_homophilic = trust_to_homophilic[:trust_min_len]
    trust_to_normative = trust_to_normative[:trust_min_len]
    main_self_trust = main_self_trust[:trust_min_len]

    main_opinion = main_opinion[:opinion_min_len]
    homophilic_opinion = homophilic_opinion[:opinion_min_len]
    normative_opinion = normative_opinion[:opinion_min_len]

    trust_steps = range(trust_min_len)
    opinion_steps = range(opinion_min_len)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=False)
    axes[0].plot(trust_steps, trust_to_homophilic, label="Trust toward Homophilic Agent")
    axes[0].plot(trust_steps, trust_to_normative, label="Trust toward Normative Agent")
    axes[0].plot(trust_steps, main_self_trust, label="Main Agent Self-Trust")
    axes[0].set_ylabel("Trust")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Main Agent Trust Dynamics")
    axes[0].legend()
    axes[1].plot(opinion_steps, main_opinion, label="Main Agent Opinion")
    axes[1].plot(opinion_steps, homophilic_opinion, label="Homophilic Agent Opinion")
    axes[1].plot(opinion_steps, normative_opinion, label="Normative Agent Opinion")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Opinion")
    axes[1].set_ylim(-1, 1)
    axes[1].set_title("Agent Opinion Dynamics")
    axes[1].legend()

    fig.suptitle(f"Model Dynamics: {model_name}")
    fig.tight_layout()
    figure_path = FIGURES_DIR / f"{model_name}_trust_opinion_comparison.png"
    fig.savefig(figure_path, dpi=300)
    plt.show()


if __name__ == "__main__":
    #analyze_trust("0_0_0_0_0_0")
    #analyze_trust("0_0_0_0_0_1")
    analyze_trust("0_0_0_0_1_0")
    analyze_trust("0_0_0_0_1_1")