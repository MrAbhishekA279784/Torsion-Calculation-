"""ShaftLab — Torsion of Circular Shaft
Simplified two-section dashboard:
    Section 1: Shaft Simulator (live inputs, KPI cards, 3D shaft, graphs)
    Section 2: Engineering Analysis (formulas, graphs, safety, data table)

Results are recalculated from the LATEST input widgets on every rerun:
the root cause of the old "not updating" bug was that results were stored
in session state once and only refreshed inside the Calculate button.
"""

import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st

from calculations import (
    MATERIAL_NAMES,
    TorsionInputs,
    TorsionResults,
    calculate_torsion,
    get_material,
    shear_stress_profile,
    torque_twist_curve,
    validate_inputs,
)


# ---------------------------------------------------------------------------
# Page config + dark navy CAD theme (blue / orange accents)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ShaftLab — Torsion of Circular Shaft",
    page_icon="⚙",
    layout="wide",
    initial_sidebar_state="expanded",
)

CAD_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Outfit:wght@400;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background-color: #080d1a !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}
.block-container { padding-top: 1rem !important; max-width: 100% !important; }

[data-testid="stSidebar"] {
    background-color: #0b1325 !important;
    border-right: 1px solid rgba(56, 189, 248, 0.12) !important;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #0f192e !important;
    border: 1px solid rgba(56, 189, 248, 0.15) !important;
    border-radius: 10px !important;
    padding: 14px 16px !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    background: #0d172b; border: 1px solid rgba(56, 189, 248, 0.15);
    border-radius: 8px 8px 0 0; color: #94a3b8 !important;
    font-weight: 600; letter-spacing: 0.03em;
}
.stTabs [data-baseweb="tab"] p { margin: 4px 12px; }
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #0f2340, #132a52) !important;
    border-color: #38bdf8 !important; color: #38bdf8 !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none; }

/* KPI cards */
.kpi-container { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 6px 0 18px 0; }
.kpi-card {
    background: #0d172b; border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px; padding: 14px 16px; position: relative; overflow: hidden;
    transition: border-color 0.2s ease;
}
.kpi-card:hover { border-color: rgba(56, 189, 248, 0.35); }
.kpi-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: #38bdf8;
}
.kpi-card.orange::before { background: linear-gradient(90deg, #f97316, #fb923c); }
.kpi-card.blue::before { background: linear-gradient(90deg, #0284c7, #38bdf8); }
.kpi-card.emerald::before { background: linear-gradient(90deg, #059669, #10b981); }
.kpi-card.green::before { background: linear-gradient(90deg, #10b981, #34d399); }
.kpi-card.amber::before { background: linear-gradient(90deg, #d97706, #f59e0b); }
.kpi-card.red::before { background: linear-gradient(90deg, #dc2626, #ef4444); }
.kpi-title { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; color: #94a3b8; }
.kpi-val { font-family: 'JetBrains Mono', monospace; font-size: 1.6rem; font-weight: 700; color: #f8fafc; letter-spacing: -0.02em; }
.kpi-unit { font-size: 0.78rem; color: #64748b; font-weight: 500; margin-left: 6px; }
.kpi-footer { display: flex; justify-content: space-between; margin-top: 6px; font-size: 0.7rem; color: #64748b; }
.kpi-trend { padding: 2px 6px; border-radius: 4px; font-weight: 600; }
.kpi-trend.safe { background: rgba(16, 185, 129, 0.15); color: #34d399; }
.kpi-trend.warn { background: rgba(245, 158, 11, 0.15); color: #fbbf24; }
.kpi-trend.danger { background: rgba(239, 68, 68, 0.15); color: #f87171; }

/* Status badge */
.status-badge {
    display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px;
    border-radius: 20px; font-size: 0.72rem; font-weight: 700;
    letter-spacing: 0.04em; text-transform: uppercase;
}
.badge-safe { background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.35); }
.badge-marginal { background: rgba(245, 158, 11, 0.15); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.35); }
.badge-unsafe { background: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.35); }

/* Inputs */
[data-testid="stSlider"] label, [data-testid="stNumberInput"] label,
[data-testid="stSelectbox"] label, [data-testid="stRadio"] label {
    font-size: 0.75rem !important; font-weight: 600 !important;
    color: #cbd5e1 !important; text-transform: uppercase !important; letter-spacing: 0.03em !important;
}
div[data-baseweb="input"] {
    background-color: #0c1424 !important; border-color: rgba(56, 189, 248, 0.2) !important;
}
div[data-baseweb="select"] > div {
    background-color: #0c1424 !important; border-color: rgba(56, 189, 248, 0.2) !important;
}

/* Buttons */
.stButton > button {
    background: #131f37 !important; border: 1px solid rgba(56, 189, 248, 0.25) !important;
    color: #e2e8f0 !important; border-radius: 8px !important; font-weight: 500 !important;
}
.stButton > button:hover { border-color: #38bdf8 !important; color: #ffffff !important; }
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #f97316 0%, #ea580c 100%) !important;
    color: #ffffff !important; font-weight: 700 !important;
    border: 1px solid #ff7849 !important;
    box-shadow: 0 4px 14px rgba(249, 115, 22, 0.35) !important;
}

/* Markdown tables (st.table) */
table {
    background: #0d172b !important; border: 1px solid rgba(56, 189, 248, 0.12) !important;
    border-radius: 8px !important; overflow: hidden !important;
}
th {
    background: #13213d !important; color: #94a3b8 !important;
    font-size: 0.74rem !important; text-transform: uppercase !important;
    padding: 9px 13px !important;
}
td {
    padding: 8px 13px !important; font-size: 0.82rem !important; color: #e2e8f0 !important;
    font-family: 'JetBrains Mono', monospace !important;
}
tr:hover td { background: rgba(56, 189, 248, 0.05) !important; }

/* Sidebar quick guide */
.quick-guide-box {
    background: #0d172b; border: 1px solid rgba(56, 189, 248, 0.12);
    border-radius: 8px; padding: 12px 14px; margin-top: 16px;
    font-size: 0.78rem; color: #94a3b8; line-height: 1.9;
}
.quick-guide-title { font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; }
.quick-guide-box b { color: #e2e8f0; }

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
"""
st.markdown(CAD_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state defaults (single source of truth)
# ---------------------------------------------------------------------------
DEFAULTS = {
    "shaft_type": "Solid",
    "material_select": "Alloy Steel (AISI 4140)",
    "do_slider": 50.0, "do_num": 50.0,
    "di_slider": 30.0, "di_num": 30.0,
    "len_slider": 1000.0, "len_num": 1000.0,
    "torque_slider": 2000.0, "torque_num": 2000.0,
    "g_slider": 80.0, "g_num": 80.0,
    "tau_allow_slider": 210.0, "tau_allow_num": 210.0,
    "twist_allow_slider": 3.0, "twist_allow_num": 3.0,
}


def init_session_state() -> None:
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)


init_session_state()


# ---------------------------------------------------------------------------
# Slider <-> number-input synchronization helpers
# ---------------------------------------------------------------------------
def _sync_to_num(key):
    def callback():
        st.session_state[f"{key}_num"] = float(st.session_state[f"{key}_slider"])
    return callback


def _sync_to_slider(key):
    def callback():
        st.session_state[f"{key}_slider"] = float(st.session_state[f"{key}_num"])
    return callback


def control_pair(label: str, key: str, min_v: float, max_v: float, step: float, fmt: str = "%0.1f") -> float:
    """A slider + exact number input that always stay in sync."""
    col_slider, col_num = st.columns([1.85, 1.0])
    with col_slider:
        st.slider(
            label, min_value=float(min_v), max_value=float(max_v), step=float(step),
            key=f"{key}_slider", format=fmt, on_change=_sync_to_num(key),
        )
    with col_num:
        st.number_input(
            "Exact value", min_value=float(min_v), max_value=float(max_v), step=float(step),
            key=f"{key}_num", format=fmt, label_visibility="collapsed",
            on_change=_sync_to_slider(key),
        )
    return float(st.session_state[f"{key}_slider"])


def on_shaft_type_change() -> None:
    """When switching to Hollow, give a sensible default bore if none exists."""
    if st.session_state["shaft_type"] == "Hollow":
        di = float(st.session_state.get("di_slider", 0.0) or 0.0)
        if di <= 0.5:
            do = float(st.session_state["do_slider"])
            di = float(min(max(0.6 * do, 1.0), do - 2.0))
            st.session_state["di_slider"] = di
            st.session_state["di_num"] = di


def on_material_change() -> None:
    """A selected material instantly rewrites G and tau_allow (live update)."""
    mat = get_material(st.session_state["material_select"])
    st.session_state["g_slider"] = float(mat["G"])
    st.session_state["g_num"] = float(mat["G"])
    st.session_state["tau_allow_slider"] = float(mat["tau_allow"])
    st.session_state["tau_allow_num"] = float(mat["tau_allow"])


# ---------------------------------------------------------------------------
# Read LATEST values from the widgets -> build engine inputs
# ---------------------------------------------------------------------------
hollow = st.session_state["shaft_type"] == "Hollow"
material = get_material(st.session_state["material_select"])

# Clamp the bore so a shrunken outer diameter can never leave Di in an invalid range
if hollow:
    di_max = max(1.0, float(st.session_state["do_slider"]) - 2.0)
    for di_key in ("di_slider", "di_num"):
        st.session_state[di_key] = float(
            min(max(float(st.session_state[di_key]), 1.0), di_max)
        )

engine_inputs = TorsionInputs(
    shaft_type=st.session_state["shaft_type"],
    outer_diameter_mm=float(st.session_state["do_slider"]),
    inner_diameter_mm=float(st.session_state["di_slider"]) if hollow else 0.0,
    length_mm=float(st.session_state["len_slider"]),
    applied_torque_nm=float(st.session_state["torque_slider"]),
    shear_modulus_gpa=float(st.session_state["g_slider"]),
    allowable_shear_stress_mpa=float(st.session_state["tau_allow_slider"]),
    max_twist_deg=float(st.session_state["twist_allow_slider"]),
    density_kg_m3=float(material["rho"]),
    material=st.session_state["material_select"],
)

is_valid, error_msg = validate_inputs(engine_inputs)

# Recalculate on EVERY rerun with the latest widget values - no caching.
results = calculate_torsion(engine_inputs) if is_valid else None


# ---------------------------------------------------------------------------
# Visualization builders (all data comes from the current inputs)
# ---------------------------------------------------------------------------
def build_3d_shaft_figure(inp: TorsionInputs, res: TorsionResults) -> go.Figure:
    """Interactive 3D CAD shaft whose geometry follows the live inputs."""
    do, di, length = inp.outer_diameter_mm, inp.inner_diameter_mm, inp.length_mm
    ro, ri = res.radius_mm, res.inner_radius_mm

    theta = np.linspace(0, 2 * np.pi, 64)
    x = np.linspace(0, length, 40)
    theta_g, x_g = np.meshgrid(theta, x)

    tau_max = max(1e-9, res.max_shear_stress_mpa)
    tau_scale = max(inp.allowable_shear_stress_mpa, tau_max * 1.05)
    stress_colorscale = [
        [0.0, "#1e3a5f"],   # low stress - deep blue
        [0.4, "#3b82f6"],   # safe - electric blue
        [0.65, "#0ea5e9"],
        [0.85, "#f97316"],  # approaching limit - orange
        [1.0, "#ef4444"],   # over limit - red
    ]

    fig = go.Figure()

    # Fixed wall flange at x = 0
    flange_r, flange_thick = ro * 1.6, max(10.0, length * 0.03)
    xf = np.linspace(-flange_thick, 0, 8)
    tf, xf_g = np.meshgrid(theta, xf)
    fig.add_trace(go.Surface(
        x=xf_g, y=flange_r * np.cos(tf), z=flange_r * np.sin(tf),
        colorscale=[[0, "#334155"], [1, "#475569"]], showscale=False,
        hoverinfo="text", text="Fixed wall support (reaction torque)",
        lighting=dict(ambient=0.6, diffuse=0.8, specular=0.5),
    ))
    rr = np.linspace(0, flange_r, 12)
    rd, td = np.meshgrid(rr, theta)
    fig.add_trace(go.Surface(
        x=np.full_like(rd, -flange_thick), y=rd * np.cos(td), z=rd * np.sin(td),
        colorscale=[[0, "#1e293b"], [1, "#334155"]], showscale=False, hoverinfo="skip",
    ))

    # Outer cylinder, colored by surface shear stress level
    stress_surface = np.full_like(x_g, tau_max)
    fig.add_trace(go.Surface(
        x=x_g, y=ro * np.cos(theta_g), z=ro * np.sin(theta_g),
        surfacecolor=stress_surface, colorscale=stress_colorscale, cmin=0, cmax=tau_scale,
        colorbar=dict(title=dict(text="τ (MPa)", font=dict(color="#94a3b8", size=11)),
                      tickfont=dict(color="#cbd5e1", size=10), x=1.02, len=0.7, thickness=14),
        lighting=dict(ambient=0.45, diffuse=0.85, specular=0.6, fresnel=0.3),
        hovertemplate=("Shaft surface<br>x: %{x:.0f} mm<br>Radius: " + f"{ro:.1f} mm"
                       + "<br>τ: " + f"{res.max_shear_stress_mpa:.1f} MPa<extra></extra>"),
    ))

    # End faces colored by the radial stress field tau(r) = T*r/J
    r_dom = np.linspace(ri if inp.shaft_type == "Hollow" else 0.0, ro, 16)
    rd2, td2 = np.meshgrid(r_dom, theta)
    fig.add_trace(go.Surface(
        x=np.full_like(rd2, length), y=rd2 * np.cos(td2), z=rd2 * np.sin(td2),
        surfacecolor=(rd2 / ro) * tau_max if tau_max > 0 else rd2 * 0.0,
        colorscale=stress_colorscale, cmin=0, cmax=tau_scale, showscale=False,
        lighting=dict(ambient=0.5, diffuse=0.8, specular=0.5),
        hovertemplate="End face<br>r: %{y:.1f} mm<br>τ: %{surfacecolor:.1f} MPa<extra></extra>",
    ))

    # Hollow bore (dark cavity)
    if inp.shaft_type == "Hollow" and ri > 0:
        fig.add_trace(go.Surface(
            x=x_g, y=ri * np.cos(theta_g), z=ri * np.sin(theta_g),
            colorscale=[[0, "#090d16"], [1, "#0f172a"]], showscale=False,
            hoverinfo="text", text=f"Inner bore Di = {di:.1f} mm",
            lighting=dict(ambient=0.2, diffuse=0.4),
        ))

    # Applied torque: orange circular arrow at the free end
    if inp.applied_torque_nm > 0:
        arc_r = ro * 1.35
        phi = np.linspace(0.15 * np.pi, 1.45 * np.pi, 45)
        ax_x = np.full_like(phi, length + length * 0.04 + 8.0)
        ax_y, ax_z = arc_r * np.cos(phi), arc_r * np.sin(phi)
        fig.add_trace(go.Scatter3d(
            x=ax_x, y=ax_y, z=ax_z, mode="lines",
            line=dict(color="#f97316", width=9),
            hoverinfo="text", text=f"Applied torque T = {inp.applied_torque_nm:.1f} N·m",
        ))
        tip_y, tip_z = ax_y[-1], ax_z[-1]
        fig.add_trace(go.Cone(
            x=[ax_x[-1]], y=[tip_y], z=[tip_z],
            u=[0], v=[-arc_r * np.sin(phi[-1])], w=[arc_r * np.cos(phi[-1])],
            colorscale=[[0, "#f97316"], [1, "#fb923c"]], showscale=False,
            sizemode="absolute", sizeref=max(2.0, ro * 0.4), anchor="tip", hoverinfo="skip",
        ))

    # Length callout + labels
    dim_z = ro * 1.7
    fig.add_trace(go.Scatter3d(
        x=[0, length], y=[0, 0], z=[dim_z, dim_z], mode="lines+markers",
        line=dict(color="#94a3b8", width=3, dash="dash"),
        marker=dict(size=4, color="#38bdf8"),
        hoverinfo="text", text=f"Length L = {length:.0f} mm",
    ))
    fig.add_trace(go.Scatter3d(
        x=[length * 0.5, length + length * 0.04 + 8.0], y=[0, 0],
        z=[dim_z * 1.12, arc_r * 1.18 if inp.applied_torque_nm > 0 else dim_z],
        mode="text",
        text=[f"L = {length:.0f} mm", f"T = {inp.applied_torque_nm:.0f} N·m"],
        textfont=dict(color=["#38bdf8", "#f97316"], size=[12, 13], family="JetBrains Mono"),
        hoverinfo="skip", showlegend=False,
    ))

    # Dynamic proportions: play off the real slenderness L/Do but capped for visibility
    slenderness = do / max(length, 1e-6)
    cross_scale = float(min(2.1, max(0.45, slenderness * 28.0)))
    # The three scene axes share one style; only the title differs.
    _axis_common = dict(
        backgroundcolor="#080d1a",
        gridcolor="rgba(255,255,255,0.06)",
        showbackground=True,
        tickfont=dict(color="#64748b", size=9),
    )
    scene = {
        axis_key: {**_axis_common, "title": dict(text=title, font=dict(color="#64748b", size=10))}
        for axis_key, title in (("xaxis", "Length (mm)"), ("yaxis", "Y (mm)"), ("zaxis", "Z (mm)"))
    }
    fig.update_layout(
        paper_bgcolor="#0b1325", plot_bgcolor="#0b1325",
        margin=dict(l=0, r=0, t=8, b=0), height=400, showlegend=False,
        scene={**scene,
               "aspectratio": dict(x=2.6, y=cross_scale, z=cross_scale),
               "camera": dict(eye=dict(x=1.8, y=-2.0, z=1.3), up=dict(x=0, y=0, z=1))},
    )
    return fig


def build_cross_section_figure(inp: TorsionInputs, res: TorsionResults) -> go.Figure:
    """2D circular cross-section with the shear stress field tau(r) = T*r/J."""
    ro, ri = res.radius_mm, res.inner_radius_mm
    grid = np.linspace(-ro * 1.25, ro * 1.25, 140)
    xm, ym = np.meshgrid(grid, grid)
    r = np.hypot(xm, ym)

    tau_max = max(1e-9, res.max_shear_stress_mpa)
    stress = np.full_like(r, np.nan)
    inside = (r <= ro) & (r >= ri)
    stress[inside] = (r[inside] / ro) * res.max_shear_stress_mpa

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=stress, x=grid, y=grid, showscale=False,
        colorscale=[[0.0, "#080d1a"], [0.1, "#0369a1"], [0.4, "#0284c7"],
                    [0.75, "#f97316"], [1.0, "#ef4444"]],
        hovertemplate="x: %{x:.1f} mm<br>y: %{y:.1f} mm<br>τ: %{z:.1f} MPa<extra></extra>",
    ))

    circle = np.linspace(0, 2 * np.pi, 120)
    fig.add_trace(go.Scatter(
        x=ro * np.cos(circle), y=ro * np.sin(circle), mode="lines",
        line=dict(color="#38bdf8", width=2.5),
        hoverinfo="text", text=f"Outer boundary Do = {inp.outer_diameter_mm:.1f} mm",
    ))
    if inp.shaft_type == "Hollow" and ri > 0:
        fig.add_trace(go.Scatter(
            x=ri * np.cos(circle), y=ri * np.sin(circle), mode="lines",
            line=dict(color="#f87171", width=2, dash="dash"),
            hoverinfo="text", text=f"Inner bore Di = {inp.inner_diameter_mm:.1f} mm",
        ))

    ang = np.pi / 4
    fig.add_annotation(
        x=ro * np.cos(ang), y=ro * np.sin(ang),
        ax=ro * 1.4 * np.cos(ang), ay=ro * 1.4 * np.sin(ang),
        xref="x", yref="y", axref="x", ayref="y",
        text=f"τ_max = {res.max_shear_stress_mpa:.1f} MPa",
        showarrow=True, arrowhead=2, arrowwidth=2, arrowcolor="#f97316",
        font=dict(color="#f97316", size=11, family="JetBrains Mono"),
        bgcolor="rgba(15, 25, 46, 0.85)", bordercolor="rgba(249, 115, 22, 0.4)",
        borderwidth=1, borderpad=4,
    )
    fig.add_annotation(
        x=0, y=-ro * 1.15,
        text=f"Do = {inp.outer_diameter_mm:.1f} mm"
             + (f" | Di = {inp.inner_diameter_mm:.1f} mm" if inp.shaft_type == "Hollow" else ""),
        showarrow=False, font=dict(color="#38bdf8", size=11, family="JetBrains Mono"),
        bgcolor="rgba(15, 25, 46, 0.8)",
    )

    fig.update_layout(
        paper_bgcolor="#0b1325", plot_bgcolor="#0b1325",
        margin=dict(l=10, r=10, t=26, b=10), height=400,
        title=dict(text="Cross-section τ field", font=dict(color="#94a3b8", size=13), x=0.5, xanchor="center"),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-ro * 1.45, ro * 1.45]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[-ro * 1.45, ro * 1.45], scaleanchor="x", scaleratio=1),
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Matplotlib engineering graphs (dark theme)
# ---------------------------------------------------------------------------
FIG_BG, AX_BG, GRID_C, TEXT_C = "#0f192e", "#0b1424", "#263245", "#cbd5e1"


def _style_axes(ax) -> None:
    ax.set_facecolor(AX_BG)
    ax.grid(True, color=GRID_C, linewidth=0.7, alpha=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color("#3b4b6b")
    ax.tick_params(colors="#94a3b8", labelsize=9)
    ax.xaxis.label.set_color(TEXT_C)
    ax.yaxis.label.set_color(TEXT_C)
    ax.title.set_color("#e2e8f0")


def _finalize_chart(fig, ax, legend_loc: str) -> None:
    """Shared legend styling and final polish for the dark engineering charts."""
    ax.legend(framealpha=0.25, facecolor="#1a2a4b", edgecolor="#3b4b6b", fontsize=8.5,
              loc=legend_loc, labelcolor=TEXT_C)
    _style_axes(ax)
    fig.patch.set_facecolor(FIG_BG)
    fig.tight_layout()


def fig_shear_vs_radius(inp: TorsionInputs, res: TorsionResults) -> plt.Figure:
    """Graph A: tau(r) - zero at the core, maximum at the surface."""
    r_mm, tau_mpa = shear_stress_profile(inp, res)
    fig, ax = plt.subplots(figsize=(4.8, 3.5), dpi=110)
    ax.plot(r_mm, tau_mpa, color="#38bdf8", lw=2.6, label="τ(r) = T·r / J", zorder=3)
    ax.fill_between(r_mm, tau_mpa, color="#0ea5e9", alpha=0.16, zorder=2)
    if res.min_shear_stress_mpa > 0:
        ax.annotate(
            f"τ_min = {res.min_shear_stress_mpa:.1f} MPa (bore)",
            xy=(r_mm[0], res.min_shear_stress_mpa), xytext=(5, 8),
            textcoords="offset points", color="#fbbf24", fontsize=8.5,
        )
    ax.axhline(inp.allowable_shear_stress_mpa, color="#ef4444", lw=1.4, ls="--",
               label=f"τ_allow = {inp.allowable_shear_stress_mpa:.0f} MPa", zorder=2)
    ax.scatter([res.radius_mm], [res.max_shear_stress_mpa], color="#f97316", zorder=4, s=42,
               edgecolor="#0b1424", linewidth=1.2)
    ax.annotate(
        f"τ_max = {res.max_shear_stress_mpa:.1f} MPa",
        xy=(res.radius_mm, res.max_shear_stress_mpa), xytext=(-10, 10),
        textcoords="offset points", color="#f97316", fontsize=9, fontweight="bold",
        ha="right",
    )
    ax.set_xlabel("Radial position r (mm)")
    ax.set_ylabel("Shear stress τ (MPa)")
    ax.set_title("Shear Stress vs Radius", fontsize=11, fontweight="bold")
    _finalize_chart(fig, ax, "lower right")
    return fig


def fig_torque_vs_twist(inp: TorsionInputs, res: TorsionResults) -> plt.Figure:
    """Graph B: T-theta - a straight line, slope = torsional stiffness kt = G*J/L."""
    theta_deg, torque_nm, kt = torque_twist_curve(inp, res)
    fig, ax = plt.subplots(figsize=(4.8, 3.5), dpi=110)
    ax.plot(theta_deg, torque_nm, color="#38bdf8", lw=2.6,
            label=f"T = kt·θ   (kt = {kt / 1000:.2f} kN·m/rad)", zorder=3)
    ax.fill_between(theta_deg, torque_nm, color="#0ea5e9", alpha=0.12, zorder=2)
    if len(theta_deg) > 1:
        ax.axvline(inp.max_twist_deg, color="#fb923c", lw=1.4, ls="--",
                   label=f"θ_allow = {inp.max_twist_deg:.2f}°", zorder=2)
    # operating point (current torque)
    ax.scatter([res.angle_of_twist_deg], [inp.applied_torque_nm], color="#f97316", zorder=4,
               s=52, edgecolor="#0b1424", linewidth=1.2)
    ax.annotate(
        f"Operating point\nT = {inp.applied_torque_nm:.0f} N·m\nθ = {res.angle_of_twist_deg:.2f}°",
        xy=(res.angle_of_twist_deg, inp.applied_torque_nm), xytext=(10, -12),
        textcoords="offset points", color="#f97316", fontsize=8.5,
    )
    ax.set_xlabel("Angle of twist θ (degrees)")
    ax.set_ylabel("Applied torque T (N·m)")
    ax.set_title("Torque vs Angle of Twist", fontsize=11, fontweight="bold")
    _finalize_chart(fig, ax, "upper left")
    return fig


# ---------------------------------------------------------------------------
# KPI cards
# ---------------------------------------------------------------------------
def kpi_card_html(title: str, value: str, unit: str, accent: str,
                  foot_left: str, foot_right: str, trend: str) -> str:
    return (
        f"<div class='kpi-card {accent}'>"
        f"<div><span class='kpi-title'>{title}</span></div>"
        f"<div style='margin:4px 0;'><span class='kpi-val'>{value}</span>"
        f"<span class='kpi-unit'>{unit}</span></div>"
        f"<div class='kpi-footer'><span>{foot_left}</span>"
        f"<span class='kpi-trend {trend}'>{foot_right}</span></div>"
        f"</div>"
    )


def render_kpi_cards(inp: TorsionInputs, res: TorsionResults) -> str:
    j_val = res.polar_moment_mm4
    j_str = f"{j_val / 1e6:.2f}×10⁶" if j_val >= 1e6 else f"{j_val / 1e5:.2f}×10⁵"
    stress_ok = res.max_shear_stress_mpa <= inp.allowable_shear_stress_mpa
    twist_ok = res.angle_of_twist_deg <= inp.max_twist_deg
    if math.isinf(res.factor_of_safety):
        fos_str = "∞"
    else:
        fos_str = f"{res.factor_of_safety:.2f}"
    if res.status == "SAFE":
        fos_accent, fos_trend, fos_label = "green", "safe", "SAFE"
    elif res.status == "MARGINAL":
        fos_accent, fos_trend, fos_label = "amber", "warn", "CAUTION"
    else:
        fos_accent, fos_trend, fos_label = "red", "danger", "CRITICAL"
    cards = "".join([
        kpi_card_html("Max Shear Stress", f"{res.max_shear_stress_mpa:.1f}", "MPa", "orange",
                      f"Limit: {inp.allowable_shear_stress_mpa:.0f} MPa",
                      "✓ In Limit" if stress_ok else "✕ Exceeded",
                      "safe" if stress_ok else "danger"),
        kpi_card_html("Angle of Twist", f"{res.angle_of_twist_deg:.3f}", "deg", "blue",
                      f"Limit: {inp.max_twist_deg:.2f}°",
                      "✓ Permissible" if twist_ok else "⚠ Excessive",
                      "safe" if twist_ok else "danger"),
        kpi_card_html("Polar Moment J", j_str, "mm⁴", "emerald",
                      "Torsional resistance", f"{'Hollow' if inp.shaft_type == 'Hollow' else 'Solid'}", "safe"),
        kpi_card_html("Factor of Safety", fos_str, "FOS", fos_accent,
                      f"τ_allow / τ_max = {inp.allowable_shear_stress_mpa:.0f}/{res.max_shear_stress_mpa:.1f}",
                      fos_label, fos_trend),
    ])
    return f"<div class='kpi-container'>{cards}</div>"


def status_badge_html(status: str) -> str:
    label, cls = {
        "SAFE": ("● SAFE", "badge-safe"),
        "MARGINAL": ("● MARGINAL", "badge-marginal"),
        "UNSAFE": ("● UNSAFE", "badge-unsafe"),
    }.get(status, ("● —", "badge-unsafe"))
    return f"<span class='status-badge {cls}'>{label}</span>"


# ---------------------------------------------------------------------------
# Summary table data
# ---------------------------------------------------------------------------
def build_summary_rows(inp: TorsionInputs, res: TorsionResults) -> list:
    rows = [
        ("Input", "Shaft Type", "—", inp.shaft_type, "—"),
        ("Input", "Outer Diameter", "Do", f"{inp.outer_diameter_mm:.1f}", "mm"),
        ("Input", "Inner Diameter", "Di", f"{inp.inner_diameter_mm:.1f}" if inp.shaft_type == "Hollow" else "—", "mm"),
        ("Input", "Shaft Length", "L", f"{inp.length_mm:.0f}", "mm"),
        ("Input", "Applied Torque", "T", f"{inp.applied_torque_nm:.0f}", "N·m"),
        ("Input", "Shear Modulus", "G", f"{inp.shear_modulus_gpa:.1f}", "GPa"),
        ("Input", "Allowable Shear Stress", "τ_allow", f"{inp.allowable_shear_stress_mpa:.0f}", "MPa"),
        ("Input", "Max Permissible Twist", "θ_allow", f"{inp.max_twist_deg:.2f}", "deg"),
        ("Input", "Material", "—", inp.material, "—"),
        ("Result", "Polar Moment of Inertia", "J", f"{res.polar_moment_mm4:,.0f}", "mm⁴"),
        ("Result", "Max Shear Stress", "τ_max", f"{res.max_shear_stress_mpa:.2f}", "MPa"),
        ("Result", "Angle of Twist", "θ", f"{res.angle_of_twist_deg:.4f}", "deg"),
        ("Result", "Factor of Safety", "FOS",
         "∞ (no load)" if math.isinf(res.factor_of_safety) else f"{res.factor_of_safety:.2f}", "—"),
        ("Result", "Torsional Stiffness", "kt", f"{res.torsional_stiffness_nm_rad:,.1f}", "N·m/rad"),
        ("Result", "Polar Section Modulus", "Zp", f"{res.polar_section_modulus_mm3:,.0f}", "mm³"),
        ("Result", "Strain Energy Stored", "U", f"{res.strain_energy_j:.3f}", "J"),
        ("Result", "Shaft Mass", "m", f"{res.mass_kg:.2f}", "kg"),
    ]
    return [dict(zip(("Type", "Parameter", "Symbol", "Value", "Unit"), r)) for r in rows]


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
        <div style="display:flex; align-items:center; gap:12px;">
            <div style="background:linear-gradient(135deg,#0284c7,#38bdf8); width:36px; height:36px;
                 border-radius:8px; display:flex; align-items:center; justify-content:center;
                 box-shadow:0 0 12px rgba(56,189,248,0.35);"><span style="font-size:1.1rem;">⚙</span></div>
            <div>
                <div style="font-family:'Outfit'; font-size:1.2rem; font-weight:800; color:#fff; line-height:1.1;">
                    SHAFT<span style="color:#38bdf8;">LAB</span></div>
                <div style="font-size:0.65rem; color:#64748b; letter-spacing:0.04em; text-transform:uppercase;">
                    Torsion of Circular Shaft</div>
            </div>
        </div>
    <div style="display:flex; align-items:center; gap:12px; align-self:flex-end;">
            <span class="status-badge badge-safe">● Analysis Online</span>
        </div>
    </div>
    <div style="border-bottom:1px solid rgba(56,189,248,0.12); margin-bottom:12px;"></div>
    """,
    unsafe_allow_html=True,
)

if not is_valid:
    st.error(f"Input Error: {error_msg}")

# ---------------------------------------------------------------------------
# Sidebar — all inputs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:14px;">
            <span style="font-size:1.2rem; color:#38bdf8;">⚙</span>
            <div>
                <div style="font-family:'Outfit'; font-size:1.05rem; font-weight:800; color:#fff; line-height:1.1;">
                    SHAFT<span style="color:#38bdf8;">LAB</span></div>
                <div style="font-size:0.62rem; color:#64748b; text-transform:uppercase; letter-spacing:0.04em;">
                    Input Parameters</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.radio("Shaft Type", ["Solid", "Hollow"], key="shaft_type",
             on_change=on_shaft_type_change, horizontal=True)

    st.selectbox("Material", MATERIAL_NAMES, key="material_select", on_change=on_material_change)
    st.caption(
        f"G = {material['G']:.1f} GPa · τ_allow = {material['tau_allow']:.0f} MPa · "
        f"ρ = {material['rho']:.0f} kg/m³ — applied to the model automatically"
    )

    control_pair("Outer Diameter Do (mm)", "do", 10.0, 200.0, 1.0, "%0.0f")
    if st.session_state["shaft_type"] == "Hollow":
        di_max = max(1.0, float(st.session_state["do_slider"]) - 2.0)
        control_pair("Inner Diameter Di (mm)", "di", 1.0, di_max, 0.5, "%0.1f")
    control_pair("Shaft Length L (mm)", "len", 100.0, 5000.0, 50.0, "%0.0f")
    control_pair("Applied Torque T (N·m)", "torque", 0.0, 20000.0, 50.0, "%0.0f")
    control_pair("Shear Modulus G (GPa)", "g", 1.0, 200.0, 0.5, "%0.1f")
    control_pair("Allowable Shear Stress τ_allow (MPa)", "tau_allow", 10.0, 600.0, 5.0, "%0.0f")
    control_pair("Max Permissible Twist θ_allow (deg)", "twist_allow", 0.25, 25.0, 0.25, "%0.2f")

    col_calc, col_reset = st.columns(2)
    calculate_clicked = col_calc.button("⚙ Calculate", type="primary", width='stretch')
    reset_clicked = col_reset.button("↺ Reset", width='stretch')

    if reset_clicked:
        for key, value in DEFAULTS.items():
            st.session_state[key] = value
        st.rerun()

    if calculate_clicked and is_valid:
        st.toast(f"Recalculated with the latest inputs (T = {engine_inputs.applied_torque_nm:.0f} N·m)", icon="🔄")
        st.rerun()

    st.markdown(
        """
        <div class="quick-guide-box">
            <div class="quick-guide-title">Quick Guide</div>
            <b>1 — Set parameters</b> on the left<br>
            <b>2 — Shaft Simulator</b> updates live<br>
            <b>3 — Engineering Analysis</b> for formulas, graphs & safety
        </div>
        """,
        unsafe_allow_html=True,
    )

if not is_valid:
    st.stop()
if results is None:
    st.stop()


# ---------------------------------------------------------------------------
# SECTION 1 — SHAFT SIMULATOR
# ---------------------------------------------------------------------------
# Build each engineering chart ONCE per rerun. Both tabs then render the same
# finished matplotlib figures instead of regenerating identical copies.
fig_tau_r = fig_shear_vs_radius(engine_inputs, results)
fig_t_theta = fig_torque_vs_twist(engine_inputs, results)

tab_sim, tab_analysis = st.tabs(["Shaft Simulator", "Engineering Analysis"])

with tab_sim:
    st.markdown(render_kpi_cards(engine_inputs, results), unsafe_allow_html=True)

    col_3d, col_cs = st.columns([1.7, 1.05])
    with col_3d:
        st.markdown(
            "<div style='font-size:0.8rem; font-weight:600; letter-spacing:0.05em; "
            "text-transform:uppercase; color:#94a3b8; margin-bottom:6px;'>"
            "3D Shaft Model — fixed wall + live torque arrow</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(build_3d_shaft_figure(engine_inputs, results),
                        width='stretch', config={"displayModeBar": False})
        st.caption("Proportions slightly exaggerated for clarity — the geometry, arrow and "
                   "colors follow your exact Do / Di / L / T values.")
    with col_cs:
        st.plotly_chart(build_cross_section_figure(engine_inputs, results),
                        width='stretch', config={"displayModeBar": False})

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.pyplot(fig_tau_r, width='stretch')
    with col_g2:
        st.pyplot(fig_t_theta, width='stretch')

    st.caption(
        "Everything on this page is recomputed from the current slider values on every "
        "change — move a slider and watch the shaft, KPI cards and both graphs update instantly."
    )


# ---------------------------------------------------------------------------
# SECTION 2 — ENGINEERING ANALYSIS
# ---------------------------------------------------------------------------
with tab_analysis:
    st.markdown("### Formula Guide (plain language)")

    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        st.markdown("**Polar Moment of Inertia — J**")
        st.latex(r"J = \frac{\pi D^4}{32} \quad\text{(solid)}")
        st.latex(r"J = \frac{\pi\left(D_o^4 - D_i^4\right)}{32} \quad\text{(hollow)}")
        st.caption("How strongly the cross-section resists twisting. Radius matters "
                   "to the 4th power — doubling D raises J sixteen-fold.")
    with f_col2:
        st.markdown("**Maximum Shear Stress — τ**")
        st.latex(r"\tau_{max} = \frac{T \cdot R}{J}")
        st.caption("Stress grows linearly from zero at the center to τ_max at the "
                   "surface, so τ(r) = T·r/J at any radius r.")
    with f_col3:
        st.markdown("**Angle of Twist — θ**")
        st.latex(r"\theta = \frac{T \cdot L}{J \cdot G}")
        st.caption("One end of the shaft rotates relative to the other by θ. Longer L "
                   "and softer G twist it more; larger J resists.")
    with f_col4:
        st.markdown("**Factor of Safety — FOS**")
        st.latex(r"\text{FOS} = \frac{\tau_{allow}}{\tau_{max}}")
        st.caption("How far the shaft is from yielding in shear. FOS = 1 is at the "
                   "limit; the design target here is FOS ≥ 1.5.")

    st.markdown("### Key Results")
    m1, m2, m3 = st.columns(3)
    m1.metric("Maximum Shear Stress", f"{results.max_shear_stress_mpa:.2f} MPa",
              f"{results.stress_margin_pct:+.0f}% vs τ_allow")
    m2.metric("Angle of Twist", f"{results.angle_of_twist_deg:.4f}°",
              f"{results.twist_margin_pct:+.0f}% vs θ_allow")
    m3.metric("Factor of Safety", "∞" if math.isinf(results.factor_of_safety) else f"{results.factor_of_safety:.2f}",
              results.status)

    st.markdown("### Safety Assessment")
    st.markdown(status_badge_html(results.status), unsafe_allow_html=True)
    st.markdown(f"- **Stress check:** τ_max = {results.max_shear_stress_mpa:.2f} MPa vs "
                f"τ_allow = {engine_inputs.allowable_shear_stress_mpa:.2f} MPa → "
                f"{'PASS' if results.is_stress_safe else 'FAIL'} "
                f"({results.stress_margin_pct:+.0f}% margin)")
    st.markdown(f"- **Twist check:** θ = {results.angle_of_twist_deg:.4f}° vs "
                f"θ_allow = {engine_inputs.max_twist_deg:.2f}° → "
                f"{'PASS' if results.is_twist_safe else 'FAIL'} "
                f"({results.twist_margin_pct:+.0f}% margin)")
    if results.status == "SAFE":
        st.success(results.status_detail)
    elif results.status == "MARGINAL":
        st.warning(results.status_detail)
    else:
        st.error(results.status_detail)

    st.markdown("### Engineering Graphs")
    an_col1, an_col2 = st.columns(2)
    with an_col1:
        st.pyplot(fig_tau_r, width='stretch')
        st.caption("Zero at the shaft center, maximum τ at the surface — exactly the "
                   "τ(r) = T·r/J straight line, cut at the bore for a hollow shaft.")
    with an_col2:
        st.pyplot(fig_t_theta, width='stretch')
        st.caption("In the elastic range twist is proportional to torque: the orange dot "
                   "is your current operating point T and θ.")

    st.markdown("### Input & Result Summary")
    st.table(pd.DataFrame(build_summary_rows(engine_inputs, results)), hide_index=True)
