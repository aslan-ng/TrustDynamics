import numpy as np
from itertools import combinations

from trustdynamics.organization import Organization


def generate(
    teams_count: int,
    agents_count: int,
    agents_connectoion_probability_inside_team: float,
    teams_connection_probability: float,
    seed: int | np.random.Generator | None = None,
) -> Organization:
    # Validate inputs
    if teams_count <= 0:
        raise ValueError("teams_count must be > 0.")
    if agents_count < 0:
        raise ValueError("agents_count must be >= 0.")
    for p, name in [
        (agents_connectoion_probability_inside_team, "agents_connectoion_probability_inside_team"),
        (teams_connection_probability, "teams_connection_probability"),
    ]:
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"{name} must be between 0 and 1.")

    # RNG handling
    if isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)

    org = Organization()

    # --- create teams ---
    team_names = np.array([f"Team {i+1}" for i in range(teams_count)], dtype=object)
    for name in team_names:
        org.add_team(name=str(name))

    # Resolve team IDs deterministically (same order as team_names)
    team_ids = np.array([org.search(str(name)) for name in team_names], dtype=object)
    if np.any(team_ids == None):  # noqa: E711
        raise RuntimeError("Failed to resolve some team IDs after creation.")

    # --- connect teams ONCE (using IDs) ---
    for i, j in combinations(range(teams_count), 2):
        if rng.random() < teams_connection_probability:
            org.add_team_connection(team_ids[i], team_ids[j])

    # --- balanced team assignment for agents ---
    base = agents_count // teams_count
    rem = agents_count % teams_count

    counts = np.full(teams_count, base, dtype=int)
    counts[:rem] += 1

    team_idx = np.repeat(np.arange(teams_count), counts)
    rng.shuffle(team_idx)

    agent_names = np.array([f"Agent {k+1}" for k in range(agents_count)], dtype=object)

    # Keep IDs as Python ints (object dtype)
    agent_ids = np.empty(agents_count, dtype=object)
    assigned_team_ids = team_ids[team_idx]  # dtype=object, no astype

    # --- add agents ---
    for k in range(agents_count):
        org.add_agent(name=str(agent_names[k]), team=assigned_team_ids[k])
        agent_id = org.search(str(agent_names[k]))
        if agent_id is None:
            raise RuntimeError(f"Failed to resolve agent id for {agent_names[k]}.")
        agent_ids[k] = agent_id

    # --- connect agents inside each team ---
    for t in team_ids:
        members = list(org.agents_from_team(t))
        for a, b in combinations(members, 2):
            if rng.random() < agents_connectoion_probability_inside_team:
                org.add_agent_connection(a, b)

    return org


if __name__ == "__main__":
    org = generate(
        seed=42,
        teams_count=4,
        agents_count=20,
        agents_connectoion_probability_inside_team=0.5,
        teams_connection_probability=0.3,
    )
    print(org.stat)
    org.show_agents()