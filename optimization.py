
import numpy as np
from sklearn.cluster import KMeans
from utils import haversine, warehousing_cost

# Multiplier to approximate road miles from great-circle distance
ROAD_FACTOR = 1.3

def _distance_matrix(lon, lat, centers):
    d = np.empty((len(lon), len(centers)))
    for j, (clon, clat) in enumerate(centers):
        d[:, j] = haversine(lon, lat, clon, clat) * ROAD_FACTOR
    return d

def _assign(df, centers):
    lon = df["Longitude"].values
    lat = df["Latitude"].values
    dmat = _distance_matrix(lon, lat, centers)
    idx = dmat.argmin(axis=1)
    dmin = dmat[np.arange(len(df)), idx]
    return idx, dmin

def _greedy_select(df, k, fixed, sites, rate_out):
    fixed_uniq = []
    seen = set()
    for lon, lat in fixed:
        key = (round(lon, 6), round(lat, 6))
        if key not in seen:
            seen.add(key)
            fixed_uniq.append([lon, lat])
    chosen = fixed_uniq.copy()
    pool = [s for s in sites if (round(s[0],6), round(s[1],6)) not in {(round(x[0],6),round(x[1],6)) for x in chosen}]
    while len(chosen) < k and pool:
        best_site, best_cost = None, None
        for cand in pool:
            cost, _, _ = _outbound(df, chosen + [cand], rate_out)
            if best_cost is None or cost < best_cost:
                best_site, best_cost = cand, cost
        chosen.append(best_site)
        pool.remove(best_site)
    return chosen

def _outbound(df, centers, rate_out):
    idx, dmin = _assign(df, centers)
    return (df["DemandLbs"] * dmin * rate_out).sum(), idx, dmin

def _service_levels(dmin, weights):
    import numpy as _np
    wtot = float(_np.sum(weights)) if _np.sum(weights) > 0 else 1.0
    d = _np.asarray(dmin, dtype=float)
    w = _np.asarray(weights, dtype=float)
    by7  = _np.sum(w[d <= 350.0]) / wtot
    by10 = _np.sum(w[(d > 350.0) & (d <= 500.0)]) / wtot
    eod  = _np.sum(w[(d > 500.0) & (d <= 700.0)]) / wtot
    d2p  = _np.sum(w[d > 700.0]) / wtot
    return {"by7": float(by7), "by10": float(by10), "eod": float(eod), "2day": float(d2p)}

def optimize(
    df, k_vals, rate_out,
    sqft_per_lb, cost_sqft, fixed_cost,
    consider_inbound=False, inbound_rate_mile=0.0, inbound_pts=None,
    fixed_centers=None, rdc_list=None, transfer_rate_mile=0.0,
    rdc_sqft_per_lb=None, rdc_cost_per_sqft=None,
    candidate_sites=None, restrict_cand=False, candidate_costs=None,
    service_level_targets=None, enforce_service_levels=False,
    current_state=False,
    brand_col="Brand",
    curr_wh_lon_col="CurrWH_Lon",
    curr_wh_lat_col="CurrWH_Lat",
    brand_allowed_sites=None,
    country_col="Country",
    canada_enabled=False,
    canada_threshold_lon=-105.0,
    canada_wh=None,
    brand_can_thresholds=None,
    brand_overrides_canada=False,
    warehouse_brand_allowed=None
):
    inbound_pts = inbound_pts or []
    fixed_centers = fixed_centers or []
    rdc_list = rdc_list or []
    candidate_costs = candidate_costs or {}
    service_level_targets = service_level_targets or {}
    brand_allowed_sites = brand_allowed_sites or {}
    brand_can_thresholds = brand_can_thresholds or {}
    warehouse_brand_allowed = warehouse_brand_allowed or {}

    tier1_nodes = [dict(coords=r["coords"], is_sdc=bool(r.get("is_sdc"))) for r in rdc_list]
    sdc_coords = [r["coords"] for r in tier1_nodes if r["is_sdc"]]

    def _cost_sqft(lon, lat):
        if restrict_cand:
            return candidate_costs.get((round(lon, 6), round(lat, 6)), cost_sqft)
        return cost_sqft

    if "DemandLbs" not in df.columns or "Longitude" not in df.columns or "Latitude" not in df.columns:
        raise ValueError("Demand file must include Longitude, Latitude, and DemandLbs columns.")
    if brand_col not in df.columns:
        df = df.copy(); df[brand_col] = "ALL"
    if country_col not in df.columns:
        df = df.copy(); df[country_col] = "USA"

    brand_allowed_keysets = {}
    for b, pairs in (brand_allowed_sites or {}).items():
        keyset = {(round(float(lon),6), round(float(lat),6)) for lon, lat in (pairs or [])}
        brand_allowed_keysets[str(b)] = keyset

    if current_state:
        if (curr_wh_lon_col not in df.columns) or (curr_wh_lat_col not in df.columns):
            raise ValueError("Current-state mode requires columns CurrWH_Lon and CurrWH_Lat.")
        pairs = df[[curr_wh_lon_col, curr_wh_lat_col]].dropna().drop_duplicates()
        centers = pairs[[curr_wh_lon_col, curr_wh_lat_col]].values.tolist()
        if not centers:
            raise ValueError("No current-state warehouse coordinates found in the data.")
        key_to_idx = {(round(lon,6), round(lat,6)): i for i,(lon,lat) in enumerate(centers)}
        lon = df["Longitude"].values; lat = df["Latitude"].values
        dmat = _distance_matrix(lon, lat, centers)
        forced_idx = np.empty(len(df), dtype=int)
        for i, r in enumerate(df.itertuples(index=False)):
            lon_wh = getattr(r, curr_wh_lon_col); lat_wh = getattr(r, curr_wh_lat_col)
            if (lon_wh == lon_wh) and (lat_wh == lat_wh):
                key = (round(lon_wh,6), round(lat_wh,6))
                forced_idx[i] = key_to_idx.get(key, int(np.argmin(dmat[i,:])))
            else:
                forced_idx[i] = int(np.argmin(dmat[i,:]))
        dmin = dmat[np.arange(len(df)), forced_idx]
        assigned = df.copy(); assigned["Warehouse"] = forced_idx; assigned["DistMi"] = dmin
        out_cost = float(np.sum(assigned["DemandLbs"].values * dmin * rate_out))

        demand_per_wh = [float(assigned.loc[assigned["Warehouse"] == j, "DemandLbs"].sum()) for j in range(len(centers))]
        demand_per_wh = np.asarray(demand_per_wh, dtype=float)

        trans_cost = 0.0; in_cost = 0.0
        if rdc_list:
            t1_coords = [t["coords"] for t in rdc_list]
            t1_dists = np.zeros((len(centers), len(t1_coords)), dtype=float)
            for j, (wx, wy) in enumerate(centers):
                t1_dists[j, :] = [haversine(wx, wy, tx, ty) * ROAD_FACTOR for tx, ty in t1_coords]
            center_to_t1_idx = t1_dists.argmin(axis=1)
            center_to_t1_dist = t1_dists[np.arange(len(centers)), center_to_t1_idx]
            trans_cost = float(np.sum(demand_per_wh * center_to_t1_dist) * transfer_rate_mile)
            t1_downstream_dem = np.zeros(len(t1_coords), dtype=float)
            for j in range(len(centers)):
                t1_downstream_dem[center_to_t1_idx[j]] += demand_per_wh[j]
            if consider_inbound and inbound_pts:
                for slon, slat, pct in inbound_pts:
                    d_to_t1 = np.array([haversine(slon, slat, tx, ty) * ROAD_FACTOR for tx, ty in t1_coords])
                    in_cost += float(np.sum(d_to_t1 * t1_downstream_dem) * pct * inbound_rate_mile)
        else:
            center_to_t1_idx = None; center_to_t1_dist = None; t1_coords = []; t1_downstream_dem = np.array([], dtype=float)
            if consider_inbound and inbound_pts:
                for lon_s, lat_s, pct in inbound_pts:
                    dists = np.array([haversine(lon_s, lat_s, cx, cy) * ROAD_FACTOR for cx, cy in centers])
                    in_cost += float((dists * demand_per_wh * pct * inbound_rate_mile).sum())

        wh_cost_centers = 0.0
        for (clon, clat), dem in zip(centers, demand_per_wh):
            wh_cost_centers += warehousing_cost(dem, sqft_per_lb, cost_sqft, fixed_cost)

        wh_cost_tier1 = 0.0
        if rdc_list:
            _sqft = (rdc_sqft_per_lb if rdc_sqft_per_lb is not None else sqft_per_lb)
            _csqft = (rdc_cost_per_sqft if rdc_cost_per_sqft is not None else cost_sqft)
            for handled in (t1_downstream_dem if len(t1_downstream_dem) else []):
                wh_cost_tier1 += warehousing_cost(handled, _sqft, _csqft, fixed_cost)

        wh_cost = wh_cost_centers + wh_cost_tier1
        total = out_cost + trans_cost + in_cost + wh_cost
        sl = _service_levels(dmin, assigned["DemandLbs"].values)
        return dict(
            centers=centers, assigned=assigned, demand_per_wh=demand_per_wh.tolist(),
            total_cost=total, out_cost=out_cost, in_cost=in_cost, trans_cost=trans_cost, wh_cost=wh_cost,
            rdc_list=rdc_list, tier1_coords=t1_coords,
            center_to_t1_idx=(center_to_t1_idx.tolist() if center_to_t1_idx is not None else None),
            center_to_t1_dist=(center_to_t1_dist.tolist() if center_to_t1_dist is not None else None),
            tier1_downstream_dem=t1_downstream_dem.tolist() if len(t1_downstream_dem) else [],
            service_levels=sl, sl_targets=service_level_targets or {}, sl_penalty=0.0, score=float(total),
        )

    best = None
    for k in k_vals:
        fixed_all = (fixed_centers + sdc_coords).copy()

        canada_idx_in_centers = None
        if canada_enabled and canada_wh and len(canada_wh) == 2:
            fixed_all.append(list(canada_wh))

        seen = set(); fixed_all_uniq = []
        for lon, lat in fixed_all:
            key = (round(lon,6), round(lat,6))
            if key not in seen:
                seen.add(key); fixed_all_uniq.append([lon, lat])

        k_eff = max(k, len(fixed_all_uniq))

        if candidate_sites and len(candidate_sites) >= k_eff:
            centers = _greedy_select(df, k_eff, fixed_all_uniq, candidate_sites, rate_out)
        else:
            km = KMeans(n_clusters=k_eff, n_init=10, random_state=42).fit(df[["Longitude", "Latitude"]])
            centers = km.cluster_centers_.tolist()
            for i, fc in enumerate(fixed_all_uniq[:k_eff]):
                centers[i] = fc

        if canada_enabled and canada_wh and len(canada_wh) == 2:
            key_can = (round(canada_wh[0],6), round(canada_wh[1],6))
            canada_idx_in_centers = None
            for j,(cx,cy) in enumerate(centers):
                if (round(cx,6), round(cy,6)) == key_can:
                    canada_idx_in_centers = j; break

        brand_to_mask = {}
        center_keys = [(round(cx,6), round(cy,6)) for cx, cy in centers]
        allowed_brands_by_center = []
        for (cx, cy) in centers:
            key_str = f"{round(cx,6)},{round(cy,6)}"
            brands_list = warehouse_brand_allowed.get(key_str)
            if brands_list:
                allowed_brands_by_center.append(set([str(x) for x in brands_list]))
            else:
                allowed_brands_by_center.append(None)

        for b in df[brand_col].astype(str).unique():
            allowed = brand_allowed_sites.get(str(b), [])
            if not allowed:
                brand_to_mask[b] = np.ones(len(centers), dtype=bool)
            else:
                allowed_set = {(round(float(lon),6), round(float(lat),6)) for lon, lat in allowed}
                mask = np.array([((k in allowed_set)) for k in center_keys], dtype=bool)
                brand_to_mask[b] = mask

        lon = df["Longitude"].values; lat = df["Latitude"].values
        dmat = _distance_matrix(lon, lat, centers)

        infeasible_weight = 0.0
        for i, r in enumerate(df.itertuples(index=False)):
            b = str(getattr(r, brand_col))

# Per-warehouse brand restrictions
if 'allowed_brands_by_center' in locals():
    for j in range(len(centers)):
        allowed = allowed_brands_by_center[j]
        if allowed is not None and (b not in allowed):
            mask[j] = False
            mask = brand_to_mask.get(b, np.ones(len(centers), dtype=bool)).copy()
            brand_has_allowed = (str(b) in brand_allowed_keysets) and (len(brand_allowed_keysets[str(b)]) > 0)

            if canada_enabled and (canada_idx_in_centers is not None):
                country = str(getattr(r, country_col, "USA")).upper()
                if country == "USA":
                    mask[canada_idx_in_centers] = False
                elif country == "CAN":
                    if not (brand_overrides_canada and brand_has_allowed):
                        thr = float(brand_can_thresholds.get(str(b), canada_threshold_lon))
                        lon_cust = float(getattr(r, "Longitude"))
                        if lon_cust >= thr:
                            mask[:] = False
                            mask[canada_idx_in_centers] = True
                        else:
                            mask[canada_idx_in_centers] = False

            disallowed = ~mask
            dmat[i, disallowed] = np.inf
            if not np.any(mask):
                infeasible_weight += float(getattr(r, "DemandLbs"))

        idx = np.argmin(dmat, axis=1)
        dmin = dmat[np.arange(len(df)), idx]
        infeasible_rows = np.isinf(dmin)
        if np.any(infeasible_rows):
            infeasible_weight += float(df.loc[infeasible_rows, "DemandLbs"].sum())
            dmin = dmin.copy(); dmin[infeasible_rows] = 1e6

        assigned = df.copy(); assigned["Warehouse"] = idx; assigned["DistMi"] = dmin
        out_cost = float(np.sum(assigned["DemandLbs"].values * dmin * rate_out))

        demand_per_wh = [float(assigned.loc[assigned["Warehouse"] == j, "DemandLbs"].sum()) for j in range(len(centers))]
        demand_per_wh = np.asarray(demand_per_wh, dtype=float)

        trans_cost = 0.0; in_cost = 0.0
        if rdc_list:
            t1_coords = [t["coords"] for t in rdc_list]
            t1_dists = np.zeros((len(centers), len(t1_coords)), dtype=float)
            for j, (wx, wy) in enumerate(centers):
                t1_dists[j, :] = [haversine(wx, wy, tx, ty) * ROAD_FACTOR for tx, ty in t1_coords]
            center_to_t1_idx = t1_dists.argmin(axis=1)
            center_to_t1_dist = t1_dists[np.arange(len(centers)), center_to_t1_idx]
            trans_cost = float(np.sum(demand_per_wh * center_to_t1_dist) * transfer_rate_mile)
            t1_downstream_dem = np.zeros(len(t1_coords), dtype=float)
            for j in range(len(centers)):
                t1_downstream_dem[center_to_t1_idx[j]] += demand_per_wh[j]
            if consider_inbound and inbound_pts:
                for slon, slat, pct in inbound_pts:
                    d_to_t1 = np.array([haversine(slon, slat, tx, ty) * ROAD_FACTOR for tx, ty in t1_coords])
                    in_cost += float(np.sum(d_to_t1 * t1_downstream_dem) * pct * inbound_rate_mile)
        else:
            center_to_t1_idx = None; center_to_t1_dist = None; t1_coords = []; t1_downstream_dem = np.array([], dtype=float)
            if consider_inbound and inbound_pts:
                for lon_s, lat_s, pct in inbound_pts:
                    dists = np.array([haversine(lon_s, lat_s, cx, cy) * ROAD_FACTOR for cx, cy in centers])
                    in_cost += float((dists * demand_per_wh * pct * inbound_rate_mile).sum())

        def _cost_sqft_local(lon, lat):
            if restrict_cand:
                return candidate_costs.get((round(lon, 6), round(lat, 6)), cost_sqft)
            return cost_sqft

        wh_cost_centers = 0.0
        for (clon, clat), dem in zip(centers, demand_per_wh):
            wh_cost_centers += warehousing_cost(dem, sqft_per_lb, _cost_sqft_local(clon, clat), fixed_cost)

        wh_cost_tier1 = 0.0
        if rdc_list:
            _sqft = (rdc_sqft_per_lb if rdc_sqft_per_lb is not None else sqft_per_lb)
            _csqft = (rdc_cost_per_sqft if rdc_cost_per_sqft is not None else cost_sqft)
            for handled in (t1_downstream_dem if len(t1_downstream_dem) else []):
                wh_cost_tier1 += warehousing_cost(handled, _sqft, _csqft, fixed_cost)

        wh_cost = wh_cost_centers + wh_cost_tier1
        total = out_cost + trans_cost + in_cost + wh_cost

        sl = _service_levels(dmin, assigned["DemandLbs"].values)
        shortfall_sum = 0.0
        if enforce_service_levels and service_level_targets:
            for key in ("by7", "by10", "eod", "2day"):
                tgt = float(service_level_targets.get(key, 0.0))
                ach = float(sl.get(key, 0.0))
                if tgt > ach:
                    shortfall_sum += (tgt - ach)

        infeas_penalty = float(infeasible_weight) * (rate_out + 1.0) * 1e6
        sl_penalty = shortfall_sum * (total + 1.0) * 1000.0
        score = total + sl_penalty + infeas_penalty

        if (best is None) or (score < best["score"]):
            best = dict(
                centers=centers,
                assigned=assigned,
                demand_per_wh=demand_per_wh.tolist(),
                total_cost=total,
                out_cost=out_cost,
                in_cost=in_cost,
                trans_cost=trans_cost,
                wh_cost=wh_cost,
                rdc_list=rdc_list,
                tier1_coords=t1_coords,
                center_to_t1_idx=(center_to_t1_idx.tolist() if center_to_t1_idx is not None else None),
                center_to_t1_dist=(center_to_t1_dist.tolist() if center_to_t1_dist is not None else None),
                tier1_downstream_dem=t1_downstream_dem.tolist() if len(t1_downstream_dem) else [],
                service_levels=sl,
                sl_targets=service_level_targets,
                sl_penalty=float(sl_penalty + infeas_penalty),
                score=float(score),
                canada_idx=canada_idx_in_centers
            )

    return best
