
import math, numpy as np
EARTH_RADIUS_MI = 3958.8

def haversine(lon1, lat1, lon2, lat2):
    """Great‑circle distance in miles. Supports scalars or NumPy arrays."""
    lon1 = np.asanyarray(lon1, dtype=float)
    lat1 = np.asanyarray(lat1, dtype=float)
    lon2 = np.asanyarray(lon2, dtype=float)
    lat2 = np.asanyarray(lat2, dtype=float)

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return EARTH_RADIUS_MI * 2.0 * np.arcsin(np.sqrt(a))

def warehousing_cost(demand_lbs, sqft_per_lb, cost_per_sqft, fixed_cost):
    return fixed_cost + demand_lbs * sqft_per_lb * cost_per_sqft
