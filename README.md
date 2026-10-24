# ShaftLab — Torsion of Circular Shaft

**ShaftLab** is a simplified, college-friendly engineering dashboard for the torsion of
circular shafts. It is built with just **two Python files**:

| File | Purpose |
| --- | --- |
| [`calculations.py`](calculations.py) | Complete calculation engine: polar moment of inertia, max shear stress, angle of twist, factor of safety, stiffness, input validation, and graph data generators |
| [`app.py`](app.py) | The Streamlit UI: sidebar inputs, KPI cards, 3D shaft model, cross-section, graphs, formulas and safety assessment |

The project is intentionally minimal — no multi-page navigation, no reporting engine,
no optimiser. A professor can understand it from five things: the shaft and applied
torque, the input parameters, the calculated values, the stress graph, and the twist graph.

---

## 1. Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`.

**Verification:** the formulas were validated against independent SI-unit calculations
(9 test scenarios covering solid and hollow shafts, torque/diameter/length scaling,
material substitution, graph profiles and invalid-input messages). The dedicated
`tests/` folder was removed during project cleanup to keep the deliverable at the
minimal two-file structure. A quick sanity check against closed-form theory:

| Inputs | Expected result |
| --- | --- |
| Solid: Do=50 mm, L=1000 mm, T=2000 N·m, G=80 GPa, τ_allow=200 MPa | J ≈ 613592 mm⁴, τ_max ≈ 81.5 MPa, θ ≈ 2.334°, FOS ≈ 2.45 |
| Hollow: Do=60 mm, Di=40 mm, L=1500 mm, T=1800 N·m, G=80 GPa | J ≈ 1021018 mm⁴, τ_max ≈ 52.9 MPa, τ_bore ≈ 35.3 MPa, θ ≈ 1.89° |

---

## 2. The Two Sections

### Section 1 — Shaft Simulator
* All inputs in the sidebar: Solid/Hollow, material, Do, Di (hollow only), L, T, G,
  τ_allow, θ_allow — each as a slider **plus** an exact numeric input that stay in sync.
* Four live KPI cards: Max Shear Stress (MPa), Angle of Twist (°), Polar Moment of
  Inertia (mm⁴), Factor of Safety.
* Dynamic 3D shaft (Plotly): wall support, stress-coloured cylinder and end face, and an
  orange torque arc whose position/magnitude follow the current T.
* Cross-section heatmap showing the τ(r) field (with the correct bore for hollow shafts).
* Two live graphs: **Shear Stress vs Radius** and **Torque vs Angle of Twist**.

### Section 2 — Engineering Analysis
* Every formula explained in plain language (so a student can explain it to a professor).
* τ_max, θ and FOS metrics with percentage margins.
* A colour-coded safety assessment (stress check + twist check + verdict).
* Both engineering graphs, plus a summary table of all inputs and results.

---

## 3. Engineering Equations

Everything is computed in SI units internally (m, N, m⁴, Pa, rad) and converted to
display units (mm, MPa, degrees, GPa) for the UI.

* **Polar moment of inertia**
  $$J = \frac{\pi D^4}{32} \;(\text{solid}), \qquad J = \frac{\pi (D_o^4 - D_i^4)}{32} \;(\text{hollow})$$
* **Max shear stress**  $\tau_{max} = \dfrac{T \cdot R}{J}$ (zero at the centre, max at the surface)
* **Angle of twist**  $\theta = \dfrac{T \cdot L}{J \cdot G}$
* **Factor of safety**  $\text{FOS} = \dfrac{\tau_{allow}}{\tau_{max}}$
* **Torsional stiffness**  $k_t = \dfrac{G \cdot J}{L}$ (slope of the T–θ line)

---

## 4. Why Updates Are Live

The original version cached `st.session_state.results` and only refreshed it inside the
Calculate button, so graphs and the 3D model kept showing stale numbers. In this build,
`calculate_torsion()` runs **on every rerun** with the latest widget values; the Calculate
button only forces an extra rerun, and material selection instantly writes new G and
τ_allow values into the inputs. No caching is used for anything input-dependent.
