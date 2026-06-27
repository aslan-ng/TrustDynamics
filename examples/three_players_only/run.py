import csv
from pathlib import Path
from trustdynamics import Model, Organization, Technology
from config import read_settings, BASE_DIR
import numpy as np

MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

steps = 100

def run_model(
        name: str = "model",
        seed: int = 42,
        normative_agent_opinion: float = -0.1,
        main_agent_opinion: float = 0.5,
        main_agent_self_trust: float = 0.5,
        homophily_normative_tradeoff: float = 0.8,
        technology_success_rate: float = 0.5,
        self_trust_learning_rate: float = 0.01,
        trust_learning_rate: float = 0.01,
        technology_success_impact: float = 0.05,
        technology_failure_impact: float = -0.15,
        add_normative: bool = True,
        add_homophilic: bool = True,
    ):
    org = Organization()
    org.add_team(name="Team")
    org.add_agent(
        name="Main Agent",
        team="Team",
        initial_opinion=main_agent_opinion,
        initial_self_trust=main_agent_self_trust,
        technology_success_impact=technology_success_impact,
        technology_failure_impact=technology_failure_impact,
        self_trust_learning_rate=self_trust_learning_rate,
        trust_learning_rate=trust_learning_rate,
        homophily_normative_tradeoff=homophily_normative_tradeoff,
        technology_use_cutoff_opinion=-1.0,
    )
    if add_homophilic is True:
        homophilic_agent_self_trust = 1.0
        org.add_agent(
            name="Homophilic Agent",
            team="Team",
            initial_opinion=main_agent_opinion, # Mimics the main agent's opinion
            initial_self_trust=homophilic_agent_self_trust,
            self_trust_learning_rate=0,
            trust_learning_rate=0,
            technology_use_cutoff_opinion=1.0,
            technology_success_impact=0,
            technology_failure_impact=0,
            )
        org.add_agent_connection(
            agent_1="Main Agent",
            agent_2="Homophilic Agent",
            initial_trust_1_to_2=0,
            initial_trust_2_to_1=0,
            )
        
    if add_normative is True:
        normative_agent_self_trust = 1.0
        org.add_agent(
            name="Normative Agent",
            team="Team",
            initial_opinion=normative_agent_opinion,
            initial_self_trust=normative_agent_self_trust,
            self_trust_learning_rate=0,
            trust_learning_rate=0,
            technology_use_cutoff_opinion=1.0,
            technology_success_impact=0,
            technology_failure_impact=0,
        )
        org.add_agent_connection(
            agent_1="Main Agent",
            agent_2="Normative Agent",
            initial_trust_1_to_2=0,
            initial_trust_2_to_1=0,
            )

    org.initialize(seed=seed)

    tech = Technology(success_rate=technology_success_rate, seed=seed)

    model = Model(organization=org, technology=tech)

    for _ in range(steps):
        model.update_teams_opinion()
        
        if add_normative is True:
            # Normative agent's opinion remains fixed
            model.organization.set_agent_opinion(
                "Normative Agent",
                opinion=normative_agent_opinion,
                mode="overwrite"
            )
            model.organization.set_agent_trust(
                "Normative Agent",
                "Normative Agent",
                trust=normative_agent_self_trust,
                mode="overwrite"
            )

        if add_homophilic is True:
        # Homophilic agent's opinion mimics the main agent's opinion
            main_agent_opinion = model.organization.get_agent_opinion("Main Agent")
            noise = np.random.uniform(-0.02, 0.02)
            homophilic_agent_opinion = np.clip(main_agent_opinion + noise, -1.0, 1.0)
            model.organization.set_agent_opinion(
                "Homophilic Agent",
                opinion=homophilic_agent_opinion,
                mode="overwrite"
            )
            model.organization.set_agent_trust(
                "Homophilic Agent",
                "Homophilic Agent",
                trust=homophilic_agent_self_trust,
                mode="overwrite"
            )

        #Recompute team opinion after role enforcement
        team_agents = list(model.organization.agents_from_team("Team"))
        team_opinion = float(
            np.mean([
                model.organization.get_agent_opinion(agent)
                for agent in team_agents
            ])
        )
        model.organization.set_team_opinion(
            "Team",
            team_opinion,
            mode="overwrite",
        )

        model.update_organization_opinion()
        model.update_teams_trust()
        model.update_agents_trust()
        model.agents_use_technology()

    model.save(MODELS_DIR / f"{name}.json")
    return model

def run_all(run_name: str | None = None):
    settings = read_settings()

    if run_name is not None:
        settings = [s for s in settings if s["name"] == run_name]

        if len(settings) == 0:
            raise ValueError(f"No setting found with name={run_name!r}")

    print("Runs to execute:", len(settings))

    for i, setting in enumerate(settings, start=1):
        print(f"[{i}/{len(settings)}] Running {setting['name']}")

        run_model(
            name=setting["name"],
            seed=setting["seed"],
            normative_agent_opinion=setting["normative_agent_opinion"],
            main_agent_opinion=setting["main_agent_opinion"],
            main_agent_self_trust=setting["main_agent_self_trust"],
            homophily_normative_tradeoff=setting["homophily_normative_tradeoff"],
            technology_success_rate=setting["technology_success_rate"],
        )


if __name__ == "__main__":
    # Run all settings
    #run_all()

    # Or debug only one run:
    #run_all(run_name="0_0_0_0_0_0")

    run_model(
            name="0_0_0_0_0_0",
            seed=42,
            normative_agent_opinion=-1,
            main_agent_opinion=1.0,
            main_agent_self_trust=0.8,
            homophily_normative_tradeoff=0.9,
            technology_success_rate=0.8,
            self_trust_learning_rate=0.05,
            trust_learning_rate=0.05,
        )
    
    run_model(
            name="0_0_0_0_0_1",
            seed=42,
            normative_agent_opinion=-1,
            main_agent_opinion=1.0,
            main_agent_self_trust=0.8,
            homophily_normative_tradeoff=0.1,
            technology_success_rate=0.8,
            self_trust_learning_rate=0.05,
            trust_learning_rate=0.05,
        )
    
    run_model(
            name="0_0_0_0_1_0",
            seed=42,
            normative_agent_opinion=-1,
            main_agent_opinion=1.0,
            main_agent_self_trust=0.8,
            homophily_normative_tradeoff=0.9,
            technology_success_rate=0.8,
            self_trust_learning_rate=0.05,
            trust_learning_rate=0.05,
            technology_success_impact=0.2,
            technology_failure_impact=-0.6,
        )
    
    run_model(
            name="0_0_0_0_1_1",
            seed=42,
            normative_agent_opinion=-1,
            main_agent_opinion=1.0,
            main_agent_self_trust=0.8,
            homophily_normative_tradeoff=0.1,
            technology_success_rate=0.8,
            self_trust_learning_rate=0.05,
            trust_learning_rate=0.05,
            technology_success_impact=0.2,
            technology_failure_impact=-0.6,
        )