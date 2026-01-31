def normalize_01(values: dict) -> dict:
    """
    Normalize dict of {node: value} to [0, 1].
    If all values are identical, map all to 1.0.
    """
    if not values:
        return {}
    vmin = min(values.values())
    vmax = max(values.values())
    if abs(vmax - vmin) < 1e-12:
        return {k: 1.0 for k in values}
    return {k: (v - vmin) / (vmax - vmin) for k, v in values.items()}