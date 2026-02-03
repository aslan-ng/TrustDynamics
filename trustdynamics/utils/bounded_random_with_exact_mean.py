import numpy as np


def bounded_random_with_exact_mean(
    n_total: int,
    target_mean: float,
    fixed_values: tuple[float] | list[float] | set[float] | np.ndarray = [],
    *,
    min_value: float = 0.0,
    max_value: float = 1.0,
    seed: np.random.Generator | int | None = None,
    tol: float = 1e-12,
    max_iter: int = 10_000,
) -> np.ndarray:
    """
    Generate n random values in [min_value, max_value] with (numerically) exact target mean.
    """
    fixed = np.asarray(list(fixed_values), dtype=float)
    k = fixed.size
    n_free = n_total - k

    if n_free == 0:
        return fixed

    if n_total < k:
        raise ValueError(
            "n_total must be >= len(fixed_values). "
            f"Got n_total={n_total}, len(fixed_values)={k}."
        )

    if min_value >= max_value:
        raise ValueError("min_value must be < max_value.")

    if not (min_value <= target_mean <= max_value):
        raise ValueError("target_mean must be within [min_value, max_value].")

    if np.any(fixed < min_value) or np.any(fixed > max_value):
        raise ValueError("All fixed_values must lie within [min_value, max_value].")

    # RNG handling
    if isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)
    
    target_sum_total = n_total * target_mean
    fixed_sum = float(fixed.sum())

    # If no free values, fixed must already satisfy the mean
    if n_free == 0:
        if abs(fixed_sum - target_sum_total) > tol:
            raise ValueError(
                "No degrees of freedom: fixed_values do not match the target mean."
            )
        out = fixed.copy()
        return out

    # Required mean for the free part
    free_target_sum = target_sum_total - fixed_sum
    free_target_mean = free_target_sum / n_free

    # Feasibility check
    if free_target_mean < min_value - tol or free_target_mean > max_value + tol:
        raise ValueError(
            "Infeasible constraints: cannot reach target_mean given fixed_values and bounds.\n"
            f"Required mean for remaining {n_free} values would be {free_target_mean}."
        )

    free_target_mean = float(np.clip(free_target_mean, min_value, max_value))

    # ---- generate the free values (your original algorithm) ----
    x = rng.uniform(min_value, max_value, size=n_free)

    # Shift to target mean
    x = x - x.mean() + free_target_mean
    x = np.clip(x, min_value, max_value)

    target_sum = n_free * free_target_mean
    delta = target_sum - float(x.sum())

    for _ in range(max_iter):
        if abs(delta) < tol:
            break

        if delta > 0:
            movable = x < max_value
            if not np.any(movable):
                break
            room = max_value - x[movable]
            step = min(delta / room.size, float(room.max()))
            inc = np.minimum(room, step)
            x[movable] += inc
            delta -= float(inc.sum())
        else:
            movable = x > min_value
            if not np.any(movable):
                break
            room = x[movable] - min_value
            step = min((-delta) / room.size, float(room.max()))
            dec = np.minimum(room, step)
            x[movable] -= dec
            delta += float(dec.sum())

    # Final numerical cleanup for free part
    x = np.clip(x, min_value, max_value)
    residual = target_sum - float(x.sum())

    if abs(residual) > tol:
        if residual > 0:
            idxs = np.where(x < max_value)[0]
            if idxs.size:
                i = idxs[0]
                x[i] = min(max_value, x[i] + residual)
        else:
            idxs = np.where(x > min_value)[0]
            if idxs.size:
                i = idxs[0]
                x[i] = max(min_value, x[i] + residual)

    # Combine fixed + free
    out = np.concatenate([fixed, x])

    # Final sanity check
    if abs(out.mean() - target_mean) > 10 * tol:
        raise RuntimeError("Internal error: overall mean constraint not met within tolerance.")

    return out


if __name__ == "__main__":
    samples = bounded_random_with_exact_mean(
        n_total=10,
        fixed_values=[0, 4],
        target_mean=0.3,
        min_value=-2.0,
        max_value=5.0,
        seed=42,
    )
    print(samples)
    print(samples.mean())