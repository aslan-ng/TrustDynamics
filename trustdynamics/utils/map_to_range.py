def map_to_range(x01: float, vmin: float, vmax: float) -> float:
    x = max(0.0, min(1.0, float(x01)))
    return vmin + (vmax - vmin) * x
