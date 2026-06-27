import csv
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
SETTINGS_PATH = BASE_DIR / "settings.csv"

def read_settings(settings_path: Path = SETTINGS_PATH) -> list[dict]:
    settings = []
    with open(settings_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            settings.append({
                "name": row["name"],
                "seed": int(row["seed"]),
                "normative_agent_opinion": float(row["normative_agent_opinion"]),
                "main_agent_opinion": float(row["main_agent_opinion"]),
                "main_agent_self_trust": float(row["main_agent_self_trust"]),
                "homophily_normative_tradeoff": float(row["homophily_normative_tradeoff"]),
                "technology_success_rate": float(row["technology_success_rate"]),
            })
    return settings


if __name__ == "__main__":

    # Master seed
    master_seed = 42
    rng = np.random.default_rng(master_seed)
    seeds_num = 3
    seeds = rng.integers(low=0, high=2**32 - 1, size=seeds_num).tolist()

    normative_agent_opinions = np.linspace(-1, 1, 6)
    main_agent_opinions = np.linspace(-1, 1, 6)
    main_agent_self_trusts = np.linspace(0, 1, 6)
    homophily_normative_tradeoffs = np.linspace(0, 1, 6)
    technology_success_rates = np.linspace(0, 1, 6)

    settings: list[dict] = []
    for s, seed in enumerate(seeds):
        for n, normative_agent_opinion in enumerate(normative_agent_opinions):
            for m, main_agent_opinion in enumerate(main_agent_opinions):
                for st, main_agent_self_trust in enumerate(main_agent_self_trusts):
                    for h, homophily_normative_tradeoff in enumerate(homophily_normative_tradeoffs):
                        for t, technology_success_rate in enumerate(technology_success_rates):
                            setting = {
                                "name": f"{s}_{n}_{m}_{st}_{h}_{t}",
                                "seed": seed,
                                "normative_agent_opinion": float(normative_agent_opinion),
                                "main_agent_opinion": float(main_agent_opinion),
                                "main_agent_self_trust": float(main_agent_self_trust),
                                "homophily_normative_tradeoff": float(homophily_normative_tradeoff),
                                "technology_success_rate": float(technology_success_rate),
                            }

                            settings.append(setting)

    fieldnames = settings[0].keys()

    with open(SETTINGS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(settings)

    print("Total runs:", len(settings))