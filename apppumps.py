import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

# =========================
# Config & Estilo
# =========================
st.set_page_config(
    page_title="Análise de Bombeamento",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

HLINE = "—" * 60
G = 9.80665  # m/s²

# =========================
# Utils de engenharia
# =========================
def reynolds(rho: float, mu: float, v: float, D: float) -> float:
    return max(1e-12, rho * v * D / mu)

def swamee_jain_f(Re: float, eps_rel: float) -> float:
    """Fator de atrito (turbulento). Para laminar usamos 64/Re."""
    if Re < 2300:
        return 64.0 / max(Re, 1e-9)
    # Swamee-Jain explícita
    A = eps_rel / 3.7 + 5.74 / (Re ** 0.9)
    return 0.25 / (np.log10(A) ** 2)

def area(D: float) -> float:
    return np.pi * (D ** 2) / 4.0

def velocity(Q_m3_s: float, D: float) -> float:
    return Q_m3_s / area(D)

def hazen_williams_hf(L: float, Q_m3_s: float, D: float, C: float) -> float:
    """
    Hazen-Williams (água) — D em m, Q m³/s, L m, h_f em mca.
    Fórmula SI: h_f = 10.67 * L * Q^1.852 / (C^1.852 * D^4.87)
    """
    return 10.67 * L * (Q_m3_s ** 1.852) / ((C ** 1.852) * (D ** 4.870))

def darcy_weisbach_hf(L: float, Q_m3_s: float, D: float, rho: float, mu: float, eps: float, K_minor: float = 0.0) -> Tuple[float, float, float, float]:
    """
    Retorna h_major, h_minor, f, Re.
    """
    v = velocity(Q_m3_s, D)
    Re = reynolds(rho, mu, v, D)
    f = swamee_jain_f(Re, eps / D)
    h_major = f * (L / D) * (v ** 2) / (2 * G)
    h_minor = K_minor * (v ** 2) / (2 * G)
    return h_major, h_minor, f, Re

def total_dynamic_head(H_static: float, h_major: float, h_minor: float) -> float:
    return max(0.0, H_static + h_major + h_minor)

def pump_power_kW(Q_m3_s: float, TDH_m: float, eta_pump: float, eta_motor: float, rho: float) -> Tuple[float, float, float]:
    """
    Retorna (P_hidráulica kW, P_eixo kW, P_elétrica kW)
    """
    P_hid_W = rho * G * Q_m3_s * TDH_m
    eta_p = max(1e-6, eta_pump)
    eta_m = max(1e-6, eta_motor)
    P_eixo_W = P_hid_W / eta_p
    P_ele_W = P_eixo_W / eta_m
    return P_hid_W / 1000.0, P_eixo_W / 1000.0, P_ele_W / 1000.0

def annual_energy_cost(P_kW: float, hours_per_year: float, tariff_R_per_kWh: float) -> Tuple[float, float]:
    kWh = P_kW * hours_per_year
    cost = kWh * tariff_R_per_kWh
    return kWh, cost

def water_vapor_pressure_kPa(T_C: float) -> float:
    """
    Aproximação para água (Antoine simplificada; 1–100 °C).
    Retorna kPa.
    """
    # Antoine (A,B,C) para faixa 1–100C (pressão em mmHg):
    A, B, C = 8.14019, 1810.94, 244.485
    Pv_mmHg = 10 ** (A - B / (T_C + C))
    return Pv_mmHg * 0.133322  # kPa

def npsh_available(Patm_kPa: float, T_C: float, rho: float,
                   H_suction_static_m: float, hf_suction_m: float, v_suction_mps: float) -> float:
    """
    NPSHa (m) = (Patm/ρg) + Hsuc - (Pvap/ρg) - hf_suc - v²/(2g)
    """
    Pv_kPa = water_vapor_pressure_kPa(T_C)
    Patm_Pa = Patm_kPa * 1000.0
    Pv_Pa = Pv_kPa * 1000.0
    return (Patm_Pa - Pv_Pa) / (rho * G) + H_suction_static_m - hf_suction_m - (v_suction_mps ** 2) / (2 * G)

# =========================
# Leitura de curva de bomba
# =========================
def read_pump_curve(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df = df.rename(columns={c: c.strip() for c in df.columns})
    # nomes padrão
    mapping = {
        "Q_m3_h": "Q_m3_h",
        "H_m": "H_m",
        "eta_pct": "eta_pct",
        "eta_%": "eta_pct",
        "eta": "eta_pct"
    }
    for k, v in mapping.items():
        if k in df.columns:
            df.rename(columns={k: v}, inplace=True)
    if "Q_m3_h" not in df.columns or "H_m" not in df.columns:
        raise ValueError("CSV deve conter colunas: Q_m3_h e H_m. (eta_pct é opcional)")
    df = df.sort_values("Q_m3_h").reset_index(drop=True)
    return df

def interp_curve(df: pd.DataFrame, Q_grid: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    H = np.interp(Q_grid, df["Q_m3_h"].values, df["H_m"].values, left=np.nan, right=np.nan)
    eta = None
    if "eta_pct" in df.columns:
        eta = np.interp(Q_grid, df["Q_m3_h"].values, df["eta_pct"].values, left=np.nan, right=np.nan)
    return H, eta

# =========================
# Sistema: curva aproximada
# =========================
def system_curve_parabola(H_static: float, K_sys: float, Q_grid_m3_h: np.ndarray) -> np.ndarray:
    """
    H_sys(Q) ≈ H_static + K_sys * (Q_m3_s)^2
    K_sys calculado a partir do ponto de operação informado.
    """
    Q_grid_m3_s = Q_grid_m3_h / 3600.0
    return H_static + K_sys * (Q_grid_m3_s ** 2)

def estimate_Ksys_from_point(H_static: float, Q_m3_h: float, hf_total_m: float) -> float:
    """K_sys = (hf_total) / Q^2  (Q em m³/s)"""
    Q_m3_s = Q_m3_h / 3600.0
    if Q_m3_s <= 0:
        return 0.0
    return max(0.0, hf_total_m) / (Q_m3_s ** 2)

def find_operating_point(Q_grid: np.ndarray, H_pump: np.ndarray, H_sys: np.ndarray) -> Tuple[float, float, int]:
    """
    Encontra o ponto de menor diferença |H_pump - H_sys|.
    Retorna (Q*, H* , idx).
    """
    diff = np.abs(H_pump - H_sys)
    idx = int(np.nanargmin(diff))
    return float(Q_grid[idx]), float(H_pump[idx]), idx

# =========================
# Dataclasses para cenários
# =========================
@dataclass
class ScenarioResult:
    name: str
    Q_m3_h: float
    TDH_m: float
    P_el_kW: float
    kWh_year: float
    cost_year: float
    details: Dict[str, float]

# =========================
# Sidebar — Entradas
# =========================
st.sidebar.title("⚙️ Parâmetros do Sistema")

st.sidebar.subheader("Fluido")
fluid_preset = st.sidebar.selectbox("Fluido", ["Água (20 °C)", "Personalizado"])
if fluid_preset == "Água (20 °C)":
    rho = 998.2       # kg/m³
    mu = 1.002e-3     # Pa.s
else:
    rho = st.sidebar.number_input("Densidade ρ (kg/m³)", 10.0, 3000.0, 1000.0, 1.0)
    mu = st.sidebar.number_input("Viscosidade μ (Pa·s)", 1e-5, 10.0, 1e-3, 1e-5, format="%.6f")

st.sidebar.subheader("Tubulação (trecho equivalente)")
L = st.sidebar.number_input("Comprimento L (m)", 0.0, 1e6, 100.0, 1.0)
D = st.sidebar.number_input("Diâmetro interno D (m)", 0.001, 10.0, 0.15, 0.001, format="%.3f")
eps = st.sidebar.number_input("Rugosidade ε (m)", 0.0, 0.01, 0.00015, 0.00001, format="%.5f")
K_minor = st.sidebar.number_input("Perdas singulares ΣK", 0.0, 1e3, 3.0, 0.1)

calc_method = st.sidebar.selectbox("Modelo de perda de carga", ["Darcy–Weisbach", "Hazen–Williams (água)"])
C_HW = None
if calc_method == "Hazen–Williams (água)":
    C_HW = st.sidebar.number_input("Coeficiente C (HW)", 1.0, 200.0, 120.0, 1.0)

st.sidebar.subheader("Condições de operação")
Q_user = st.sidebar.number_input("Vazão Q (m³/h)", 0.0, 1e6, 100.0, 1.0)
H_static = st.sidebar.number_input("Carga estática (m)", -500.0, 1e4, 10.0, 0.5)

st.sidebar.subheader("Bomba & Motor (se não houver curva)")
eta_pump_user = st.sidebar.slider("Rendimento bomba η_p", 0.10, 0.90, 0.75, 0.01)
eta_motor_user = st.sidebar.slider("Rendimento motor η_m", 0.50, 0.98, 0.95, 0.01)

st.sidebar.subheader("Energia")
hours_day = st.sidebar.number_input("Horas/dia", 0.0, 24.0, 20.0, 0.5)
days_year = st.sidebar.number_input("Dias/ano", 0.0, 366.0, 330.0, 1.0)
tariff = st.sidebar.number_input("Tarifa (R$/kWh)", 0.0, 5.0, 0.85, 0.01)

st.sidebar.subheader("NPSH (opcional)")
use_npsh = st.sidebar.checkbox("Calcular NPSH disponível")
if use_npsh:
    Patm_kPa = st.sidebar.number_input("Pressão atmosférica (kPa)", 60.0, 120.0, 101.325, 0.5)
    T_C = st.sidebar.number_input("Temperatura do fluido (°C)", 0.0, 120.0, 25.0, 0.5)
    H_suc_static = st.sidebar.number_input("Cabeça estática sucção (m) [+afogado / -sucção]", -20.0, 20.0, 1.5, 0.1)
    L_suc = st.sidebar.number_input("L sucção (m)", 0.0, 1e4, 10.0, 0.5)
    D_suc = st.sidebar.number_input("D sucção (m)", 0.001, 10.0, D, 0.001, format="%.3f")
    K_suc = st.sidebar.number_input("ΣK sucção", 0.0, 500.0, 2.0, 0.1)

st.sidebar.subheader("Curva da bomba (CSV opcional)")
pump_file = st.sidebar.file_uploader("CSV com colunas: Q_m3_h, H_m, (eta_pct opcional)", type=["csv"])
pump_df = None
if pump_file is not None:
    try:
        pump_df = read_pump_curve(pump_file)
    except Exception as e:
        st.sidebar.error(f"Erro no CSV: {e}")

# =========================
# Cálculos base
# =========================
hours_year = hours_day * days_year
Q_m3_s = Q_user / 3600.0

if calc_method == "Darcy–Weisbach":
    h_major, h_minor, f_used, Re = darcy_weisbach_hf(L, Q_m3_s, D, rho, mu, eps, K_minor)
else:
    # Hazen-Williams: só major; tratamos K_minor separadamente assumindo v de D
    h_major = hazen_williams_hf(L, Q_m3_s, D, C_HW)
    v_tmp = velocity(Q_m3_s, D)
    h_minor = K_minor * (v_tmp ** 2) / (2 * G)
    # Para exibir algo coerente na UI:
    v_for_Re = v_tmp
    Re = reynolds(rho, mu, v_for_Re, D)
    # f_used não se aplica em HW; deixamos em branco
    f_used = np.nan

TDH = total_dynamic_head(H_static, h_major, h_minor)

# Rendimento: se houver curva com eta, podemos estimar no ponto (apenas quando operando na interseção)
eta_p_for_calc = eta_pump_user
eta_m_for_calc = eta_motor_user

# =========================
# Curvas: Sistema x Bomba
# =========================
curve_tab, results_tab, whatif_tab, npsh_tab = st.tabs(["📈 Curvas", "📊 Resultados", "💡 Melhorias & Cenários", "🧪 NPSH"])

with curve_tab:
    st.markdown("### Curva do Sistema vs. Curva da Bomba")
    colA, colB = st.columns([1,1])
    with colA:
        st.write("**Resumo das perdas no ponto informado**")
        st.write(f"- Método: **{calc_method}**")
        st.write(f"- f (Darcy) ~ **{('—' if np.isnan(f_used) else f'{f_used:.4f}')}**, Re = **{Re:,.0f}**")
        st.write(f"- Perda major: **{h_major:.2f} m**")
        st.write(f"- Perdas singulares: **{h_minor:.2f} m**")
        st.write(f"- Carga estática: **{H_static:.2f} m**")
        st.write(f"- **TDH (ponto informado)**: **{TDH:.2f} m**")

    with colB:
        st.info("Dica: forneça a **curva da bomba** para encontrar automaticamente o ponto de operação (interseção) e estimar o rendimento real da bomba.")

    # Construímos uma parábola aproximada do sistema usando K_sys calculado no ponto informado
    Ksys = estimate_Ksys_from_point(H_static, Q_user, h_major + h_minor)
    Q_grid = np.linspace(0, max(Q_user*1.6, 1.0), 120)  # m³/h
    H_sys_grid = system_curve_parabola(H_static, Ksys, Q_grid)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=Q_grid, y=H_sys_grid, name="Curva do Sistema", mode="lines"))

    op_point_from_curve = None
    eta_from_curve = None

    if pump_df is not None:
        Hpump_grid, eta_grid = interp_curve(pump_df, Q_grid)
        fig.add_trace(go.Scatter(x=pump_df["Q_m3_h"], y=pump_df["H_m"], mode="markers+lines", name="Curva da Bomba"))
        if eta_grid is not None:
            # mostrar eficiência como cor? Simples: outra escala no hover
            pass
        # Interseção aproximada
        if np.isfinite(Hpump_grid).any():
            Q_op, H_op, idx = find_operating_point(Q_grid, Hpump_grid, H_sys_grid)
            op_point_from_curve = (Q_op, H_op)
            eta_from_curve = None if eta_grid is None else float(eta_grid[idx])
            fig.add_trace(go.Scatter(
                x=[Q_op], y=[H_op],
                mode="markers",
                marker=dict(size=10, symbol="x"),
                name="Ponto de operação (estimado)"
            ))

    st.plotly_chart(fig, use_container_width=True)

# =========================
# Resultados base
# =========================
with results_tab:
    st.markdown("### 📊 Resultados no ponto informado")

    # Se temos interseção da curva, usamos o Q_op/H_op para potência e tentamos um eta vindo da curva
    if op_point_from_curve is not None:
        Q_used_h = float(op_point_from_curve[0])
        TDH_used = float(op_point_from_curve[1])
        if eta_from_curve is not None and np.isfinite(eta_from_curve):
            eta_p_for_calc = max(0.05, min(0.95, eta_from_curve/100.0))
            st.caption(f"Rendimento da bomba estimado da curva: ~{eta_from_curve:.1f}%")
        else:
            st.caption(f"Sem coluna de eficiência no CSV — usando η_p informado: {eta_pump_user*100:.1f}%")
    else:
        Q_used_h = Q_user
        TDH_used = TDH

    P_hid_kW, P_eixo_kW, P_el_kW = pump_power_kW(Q_used_h/3600.0, TDH_used, eta_p_for_calc, eta_m_for_calc, rho)
    kWh_year, cost_year = annual_energy_cost(P_el_kW, hours_year, tariff)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Vazão (m³/h)", f"{Q_used_h:,.1f}")
    m2.metric("TDH (m)", f"{TDH_used:,.2f}")
    m3.metric("Potência elétrica (kW)", f"{P_el_kW:,.2f}")
    m4.metric("Custo anual (R$)", f"{cost_year:,.0f}")

    with st.expander("Detalhamento"):
        st.write(f"- Potência hidráulica: **{P_hid_kW:.2f} kW**")
        st.write(f"- Potência no eixo: **{P_eixo_kW:.2f} kW**")
        st.write(f"- Consumo anual: **{kWh_year:,.0f} kWh/ano**")
        st.write(f"- η_bomba usado: **{eta_p_for_calc*100:.1f}%**, η_motor: **{eta_m_for_calc*100:.1f}%**")
        st.write(f"- Horas/ano: **{hours_year:,.0f}** — Tarifa: **R$ {tariff:.2f}/kWh**")

# =========================
# What-if: VFD, Impulsor, Diâmetro
# =========================
with whatif_tab:
    st.markdown("### 💡 Sugerir melhorias e estimar economia")

    base_result = ScenarioResult(
        name="Base",
        Q_m3_h=Q_used_h,
        TDH_m=TDH_used,
        P_el_kW=P_el_kW,
        kWh_year=kWh_year,
        cost_year=cost_year,
        details={}
    )

    scenarios = [base_result]

    st.subheader("1) Redução de rotação com VFD (controle de velocidade)")
    target_Q_vfd = st.slider("Alvo de vazão com VFD (% da base)", 50, 110, 90, 1)
    Q_target_h = Q_used_h * (target_Q_vfd/100.0)
    Q_target_s = Q_target_h / 3600.0

    # Sistema com mesma tubulação -> recalcula perdas
    if calc_method == "Darcy–Weisbach":
        h_major_t, h_minor_t, _, _ = darcy_weisbach_hf(L, Q_target_s, D, rho, mu, eps, K_minor)
    else:
        h_major_t = hazen_williams_hf(L, Q_target_s, D, C_HW)
        v_tmp = velocity(Q_target_s, D)
        h_minor_t = K_minor * (v_tmp ** 2) / (2 * G)

    TDH_target = total_dynamic_head(H_static, h_major_t, h_minor_t)

    # Ajuste de rotação necessário usando leis de afinidade:
    # H_bomba(velocidade nominal) em Q_target_h ≈ H_nom(Q_target_h) (se curva disponível)
    # n2/n1 ≈ sqrt(H_sys(Q_target)/H_nom(Q_target))
    n_ratio = None
    if pump_df is not None:
        Hpump_Qt, _ = interp_curve(pump_df, np.array([Q_target_h]))
        if np.isfinite(Hpump_Qt[0]) and Hpump_Qt[0] > 0 and TDH_target > 0:
            n_ratio = np.sqrt(TDH_target / Hpump_Qt[0])
    if n_ratio is None or not np.isfinite(n_ratio):
        # fallback: assume potência ∝ Q^3 aproximadamente (boa aproximação)
        n_ratio = (Q_target_h / max(Q_used_h, 1e-9))  # ~ Q ∝ n

    # Potência com VFD ~ P_base * n_ratio^3 (aprox.)
    P_el_vfd = P_el_kW * (n_ratio ** 3)
    kWh_vfd, cost_vfd = annual_energy_cost(P_el_vfd, hours_year, tariff)

    scenarios.append(ScenarioResult(
        name=f"VFD (Q≈{Q_target_h:.0f} m³/h)",
        Q_m3_h=Q_target_h,
        TDH_m=TDH_target,
        P_el_kW=P_el_vfd,
        kWh_year=kWh_vfd,
        cost_year=cost_vfd,
        details={"n_ratio": n_ratio}
    ))

    st.caption(f"Estimativa: razão de rotação n₂/n₁ ≈ **{n_ratio:.3f}** (leis de afinidade).")

    st.subheader("2) Corte de impulsor (diâmetro)")
    H_alvo_trim = st.number_input("Deseja reduzir a carga desenvolvida em (%)", 0.0, 50.0, 10.0, 1.0)
    phi = np.sqrt(max(0.0, 1.0 - H_alvo_trim/100.0))  # D2/D1 ~ sqrt(H2/H1)
    P_el_trim = P_el_kW * (phi ** 3)
    kWh_trim, cost_trim = annual_energy_cost(P_el_trim, hours_year, tariff)
    scenarios.append(ScenarioResult(
        name=f"Corte impulsor (~{100*(1-phi):.1f}%)",
        Q_m3_h=Q_used_h * phi,  # aproximação: Q ∝ D
        TDH_m=TDH_used * (phi ** 2),
        P_el_kW=P_el_trim,
        kWh_year=kWh_trim,
        cost_year=cost_trim,
        details={"D_ratio": phi}
    ))

    st.subheader("3) Otimização de diâmetro de tubulação")
    col1, col2, col3 = st.columns(3)
    with col1:
        D_min = st.number_input("D mínimo (m)", 0.01, 2.0, max(0.01, D*0.6), 0.005, format="%.3f")
    with col2:
        D_max = st.number_input("D máximo (m)", 0.02, 3.0, max(D*1.8, D+0.01), 0.005, format="%.3f")
    with col3:
        n_steps = st.number_input("Passos", 3, 30, 7, 1)

    capex_per_m = st.number_input("CAPEX/m para tubulação (R$/m)", 0.0, 1e6, 900.0, 10.0)
    show_payback = st.checkbox("Calcular payback vs. Base", value=True)

    D_candidates = np.linspace(D_min, D_max, n_steps)
    pipe_table = []
    for Dd in D_candidates:
        Qs = Q_used_h / 3600.0
        if calc_method == "Darcy–Weisbach":
            hj, hm, _, ReD = darcy_weisbach_hf(L, Qs, Dd, rho, mu, eps, K_minor)
        else:
            hj = hazen_williams_hf(L, Qs, Dd, C_HW)
            v_tmp = velocity(Qs, Dd)
            hm = K_minor * (v_tmp ** 2) / (2 * G)
            ReD = reynolds(rho, mu, v_tmp, Dd)
        TDH_d = total_dynamic_head(H_static, hj, hm)
        P_hid_kW_d, P_eixo_kW_d, P_el_kW_d = pump_power_kW(Qs, TDH_d, eta_p_for_calc, eta_m_for_calc, rho)
        kWh_d, cost_d = annual_energy_cost(P_el_kW_d, hours_year, tariff)
        capex_total = capex_per_m * L
        saving_R = cost_year - cost_d
        payback_y = (capex_total / saving_R) if show_payback and saving_R > 0 else np.nan
        pipe_table.append([Dd, ReD, hj+hm, TDH_d, P_el_kW_d, cost_d, saving_R, payback_y])

    df_pipe = pd.DataFrame(pipe_table, columns=["D (m)", "Re", "Perda (m)", "TDH (m)", "P_el (kW)", "Custo anual (R$)", "Economia vs Base (R$)", "Payback (anos)"])
    df_pipe_sorted = df_pipe.sort_values("Custo anual (R$)").reset_index(drop=True)
    st.dataframe(df_pipe_sorted.style.format({
        "D (m)": "{:.3f}", "Re": "{:,.0f}", "Perda (m)": "{:.2f}", "TDH (m)": "{:.2f}",
        "P_el (kW)": "{:.2f}", "Custo anual (R$)": "{:,.0f}", "Economia vs Base (R$)": "{:,.0f}",
        "Payback (anos)": "{:.1f}"
    }), use_container_width=True)

    # Tabela comparativa de cenários
    st.markdown(HLINE)
    st.markdown("#### Comparativo de Cenários")
    comp = pd.DataFrame([
        [s.name, s.Q_m3_h, s.TDH_m, s.P_el_kW, s.kWh_year, s.cost_year]
        for s in scenarios
    ], columns=["Cenário", "Q (m³/h)", "TDH (m)", "P_el (kW)", "kWh/ano", "Custo anual (R$)"])
    st.dataframe(comp.style.format({
        "Q (m³/h)": "{:,.1f}", "TDH (m)": "{:.2f}", "P_el (kW)": "{:.2f}",
        "kWh/ano": "{:,.0f}", "Custo anual (R$)": "{:,.0f}"
    }), use_container_width=True)

    # Insights automáticos
    cheapest = min(scenarios, key=lambda s: s.cost_year)
    if cheapest.name != "Base":
        st.success(f"💰 Maior economia estimada: **{cheapest.name}** → custo anual ≈ **R$ {cheapest.cost_year:,.0f}** (economia de ~R$ {(cost_year - cheapest.cost_year):,.0f}/ano).")
    else:
        st.info("Ainda não há um cenário com economia maior que a base — ajuste os parâmetros.")

# =========================
# NPSH
# =========================
with npsh_tab:
    st.markdown("### 🧪 Verificação de NPSH")
    if use_npsh:
        # perdas na sucção com Q real usado
        Qs = Q_used_h / 3600.0
        v_suc = velocity(Qs, D_suc)
        if calc_method == "Darcy–Weisbach":
            hj_suc, hm_suc, _, _ = darcy_weisbach_hf(L_suc, Qs, D_suc, rho, mu, eps, K_suc)
        else:
            hj_suc = hazen_williams_hf(L_suc, Qs, D_suc, C_HW)
            hm_suc = K_suc * (v_suc ** 2) / (2 * G)
        NPSHa = npsh_available(Patm_kPa, T_C, rho, H_suc_static, hj_suc + hm_suc, v_suc)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("NPSHa (m)", f"{NPSHa:.2f}")
        with col2:
            NPSHr_user = st.number_input("NPSHr da bomba (m)", 0.0, 50.0, 3.0, 0.1)
        margin = NPSHa - NPSHr_user
        if margin < 0.5:
            st.error(f"⚠️ Margem NPSH baixa ({margin:.2f} m). Risco de cavitação.")
            st.caption("Considere aumentar diâmetro/encurtar sucção, reduzir perdas (curvas/válvulas), baixar temperatura ou elevar nível no reservatório.")
        else:
            st.success(f"✅ Margem NPSH adequada ({margin:.2f} m).")
    else:
        st.info("Habilite **Calcular NPSH disponível** na barra lateral para avaliar cavitação.")

# =========================
# Rodapé
# =========================
st.markdown(HLINE)
st.caption(
    "Este aplicativo fornece estimativas baseadas em modelos clássicos (Darcy–Weisbach, Hazen–Williams, leis de afinidade). "
    "Para decisões de CAPEX/OPEX, valide com dados de curva do fabricante e condições reais de processo."
)
