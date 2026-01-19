import numpy as np
from trustdynamics.organization.organization import Organization


def generate_organization(
        n_departments: int,
        n_people: int,
        max_depth: int,
        name: str = "Organization",
        seed: int | None = None,
    ) -> Organization:
    """Generate a random organization with guaranteed departments.

    Construction:
    1) Allocate headcount per department (sum = n_people - 1, excluding CEO).
    2) Create one department head per department as a direct report of the CEO.
    3) For each department, generate a random tree with a HARD depth cap.

    Depth semantics (hard cap):
    - Let CEO have id 0.
    - Each department head is a direct report of the CEO.
    - Within each department branch, we enforce that the maximum number of edges
      from the department head to any node is <= (d-1), where d is sampled in
      [1, max_depth].

    Notes:
    - If n_people - 1 < n_departments, it is impossible to have all departments present.
      In that case, only (n_people - 1) departments will be present (each with 1 person).
    """
    if n_departments < 1:
        raise ValueError("n_departments must be at least 1.")
    if n_people < 1:
        raise ValueError("n_people must be at least 1 (CEO).")
    if max_depth < 1:
        raise ValueError("max_depth must be at least 1.")

    rng = np.random.default_rng(seed)
    org = Organization(name=name)  # CEO exists with id 0

    # Employees excluding CEO
    n_emp = n_people - 1
    if n_emp == 0:
        return org

    departments = [f"department_{i+1}" for i in range(n_departments)]

    # --- Step 1: Decide how many agents per department (excluding CEO) ---
    # Guarantee at least 1 per department if feasible.
    if n_emp >= n_departments:
        base = np.ones(n_departments, dtype=int)
        remainder = n_emp - n_departments
        extra = rng.multinomial(remainder, np.ones(n_departments) / n_departments) if remainder > 0 else np.zeros(n_departments, dtype=int)
        dept_sizes = base + extra
    else:
        dept_sizes = np.zeros(n_departments, dtype=int)
        chosen = rng.choice(n_departments, size=n_emp, replace=False)
        dept_sizes[chosen] = 1

    # --- Step 2: Create department heads (direct reports of CEO) ---
    dept_heads: dict[int, int] = {}
    for d_i, size in enumerate(dept_sizes):
        if size <= 0:
            continue
        head_id = org.add_agent(name=None, parent=0, department=departments[d_i])
        dept_heads[d_i] = head_id

    # --- Step 3: For each department, generate an internal tree with HARD bounded depth ---
    for d_i, size in enumerate(dept_sizes):
        if size <= 0:
            continue

        remaining_in_dept = int(size - 1)  # head already created
        if remaining_in_dept <= 0:
            continue

        dept = departments[d_i]
        head = dept_heads[d_i]

        # Sample actual depth for this department branch
        # depth=1 means only head; depth>=2 allows levels below head.
        depth = int(rng.integers(1, max_depth + 1))

        # Levels under the department head
        levels: list[list[int]] = [[] for _ in range(depth)]
        levels[0].append(head)

        # Track depth-from-head for a HARD cap
        depth_from_head: dict[int, int] = {head: 0}
        max_allowed = depth - 1  # maximum edges from head

        # Seed level-by-level
        for lvl in range(1, depth):
            if remaining_in_dept <= 0:
                break

            n_this_level = int(rng.integers(1, remaining_in_dept + 1))
            parents = rng.choice(np.array(levels[lvl - 1], dtype=int), size=n_this_level, replace=True)

            for p in parents.tolist():
                new_id = org.add_agent(name=None, parent=int(p), department=dept)
                levels[lvl].append(new_id)
                depth_from_head[new_id] = depth_from_head[int(p)] + 1
                remaining_in_dept -= 1
                if remaining_in_dept <= 0:
                    break

        # Attach remaining nodes ONLY to eligible parents so we never exceed max_allowed
        all_dept_nodes = [n for level in levels for n in level]
        while remaining_in_dept > 0:
            eligible = [n for n in all_dept_nodes if depth_from_head[n] < max_allowed]
            if not eligible:
                # No place to attach without violating max depth.
                break

            p = int(rng.choice(np.array(eligible, dtype=int)))
            new_id = org.add_agent(name=None, parent=p, department=dept)
            depth_from_head[new_id] = depth_from_head[p] + 1
            all_dept_nodes.append(new_id)
            remaining_in_dept -= 1

    return org


if __name__ == "__main__":
    org = generate_organization(n_departments=3, n_people=30, max_depth=5, seed=42)
    org.draw()