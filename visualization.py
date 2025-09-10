
import os
import streamlit as st
import pandas as pd
import pydeck as pdk

# ──────────────────────────────────────────────────────────
# Map provider / token handling
# If a Mapbox token is available (via env var or Streamlit secrets),
# use Mapbox. Otherwise fall back to the free Carto basemap so that
# a background map is always rendered without requiring credentials.
# ──────────────────────────────────────────────────────────
_MAPBOX_TOKEN = os.getenv("MAPBOX_API_KEY")
if not _MAPBOX_TOKEN:
    try:
        _MAPBOX_TOKEN = st.secrets["MAPBOX_API_KEY"]  # type: ignore[attr-defined]
    except Exception:
        _MAPBOX_TOKEN = None

if _MAPBOX_TOKEN:
    pdk.settings.mapbox_api_key = _MAPBOX_TOKEN

# Colour palette
_COL = [
    [31, 119, 180],
    [255, 127, 14],
    [44, 160, 44],
    [214, 39, 40],
    [148, 103, 189],
    [140, 86, 75],
    [227, 119, 194],
    [127, 127, 127],
    [188, 189, 34],
    [23, 190, 207],
]
def _c(i): return _COL[i % len(_COL)]


def _build_deck(layers):
    """Return a pydeck.Deck object with the appropriate basemap."""
    view_state = pdk.ViewState(latitude=39, longitude=-98, zoom=3.5)
    if _MAPBOX_TOKEN:
        # Mapbox basemap — token already set above
        return pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            map_style="mapbox://styles/mapbox/light-v10"
        )
    else:
        # Free Carto basemap (no token required)
        return pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            map_provider="carto",
            map_style="light"
        )


def plot_network(stores: pd.DataFrame, centers):
    """Render the outbound network on an interactive map."""
    st.subheader("Network Map")

    # Warehouses dataframe
    cen_df = pd.DataFrame(centers, columns=["Lon", "Lat"])

    # Lines from store → assigned warehouse
    edges = [
        {
            "f": [r.Longitude, r.Latitude],
            "t": [cen_df.iloc[int(r.Warehouse)].Lon, cen_df.iloc[int(r.Warehouse)].Lat],
            "col": _c(int(r.Warehouse)) + [120],
        }
        for r in stores.itertuples()
    ]
    line_layer = pdk.Layer(
        "LineLayer",
        edges,
        get_source_position="f",
        get_target_position="t",
        get_color="col",
        get_width=2,
    )

    # Warehouse (center) layer
    cen_df[["r", "g", "b"]] = [_c(i) for i in range(len(cen_df))]
    wh_layer = pdk.Layer(
        "ScatterplotLayer",
        cen_df,
        get_position="[Lon,Lat]",
        get_fill_color="[r,g,b]",
        get_radius=35000,
        opacity=0.9,
    )

    # Store layer
    store_layer = pdk.Layer(
        "ScatterplotLayer",
        stores,
        get_position="[Longitude,Latitude]",
        get_fill_color="[0,128,255]",
        get_radius=12000,
        opacity=0.6,
    )

    deck = _build_deck([line_layer, store_layer, wh_layer])
    st.pydeck_chart(deck)


def summary(
    stores,
    total,
    out,
    in_,
    trans,
    wh,
    centers,
    demand,
    sqft_per_lb,
    rdc_on,
    consider_in,
    show_trans,
):
    """Display cost breakdown and warehouse details."""
    st.subheader("Cost Summary")
    st.metric("Total annual cost", f"${total:,.0f}")
    cols = st.columns(4 if (consider_in or show_trans) else 2)
    i = 0
    cols[i].metric("Outbound", f"${out:,.0f}"); i += 1
    if consider_in:
        cols[i].metric("Inbound", f"${in_:,.0f}"); i += 1
    if show_trans:
        cols[i].metric("Transfers", f"${trans:,.0f}"); i += 1
    cols[i].metric("Warehousing", f"${wh:,.0f}")

    df = pd.DataFrame(centers, columns=["Lon", "Lat"])
    df["DemandLbs"] = demand
    df["SqFt"] = df["DemandLbs"] * sqft_per_lb
    st.subheader("Warehouse Demand & Size")
    st.dataframe(
        df[["DemandLbs", "SqFt", "Lat", "Lon"]].style.format(
            {"DemandLbs": "{:,}", "SqFt": "{:,}"}
        )
    )


def plot_flows(lanes_df: pd.DataFrame, centers, flow_types=("outbound","transfer","inbound"), brand_filter="__ALL__"):
    """Render lane flows with toggles and brand filter.
    - flow_types: iterable subset of {"outbound","transfer","inbound"}
    - brand_filter: "__ALL__" for all brands or specific brand name
    """
    st.subheader("Flow Map")
    if lanes_df is None or lanes_df.empty:
        st.info("No lanes to display.")
        return

    # Apply brand filter
    if brand_filter != "__ALL__":
        lanes_df = lanes_df[lanes_df["brand"].astype(str) == str(brand_filter)]

    # Apply flow type filter
    lanes_df = lanes_df[lanes_df["lane_type"].isin(list(flow_types))]

    layers = []

    # Lines per flow type with distinct widths/opacities
    def _line_layer(df_sub, width, rgba):
        return pdk.Layer(
            "LineLayer",
            df_sub.to_dict("records"),
            get_source_position="origin_lon_lat",
            get_target_position="dest_lon_lat",
            get_color=rgba,
            get_width=width,
            pickable=True,
            auto_highlight=True,
        )

    # Prepare columns for pydeck positions
    lanes_df = lanes_df.copy()
    lanes_df["origin_lon_lat"] = lanes_df[["origin_lon","origin_lat"]].values.tolist()
    lanes_df["dest_lon_lat"] = lanes_df[["dest_lon","dest_lat"]].values.tolist()

    # Outbound
    out_df = lanes_df[lanes_df["lane_type"]=="outbound"]
    if not out_df.empty and "outbound" in flow_types:
        layers.append(_line_layer(out_df, 2, [0,128,255,160]))

    # Transfer
    tr_df = lanes_df[lanes_df["lane_type"]=="transfer"]
    if not tr_df.empty and "transfer" in flow_types:
        layers.append(_line_layer(tr_df, 3, [255,127,14,180]))

    # Inbound
    in_df = lanes_df[lanes_df["lane_type"]=="inbound"]
    if not in_df.empty and "inbound" in flow_types:
        layers.append(_line_layer(in_df, 1, [44,160,44,140]))

    # Centers
    cen_df = pd.DataFrame(centers, columns=["Lon", "Lat"])
    cen_df[["r","g","b"]] = [_c(i) for i in range(len(cen_df))]
    layers.append(pdk.Layer(
        "ScatterplotLayer",
        cen_df,
        get_position="[Lon,Lat]",
        get_fill_color="[r,g,b]",
        get_radius=35000,
        opacity=0.9,
    ))

    deck = _build_deck(layers)
    st.pydeck_chart(deck)
