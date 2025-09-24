import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Vocara • Earnings Test, ROI & Alternatives", layout="wide")
DATA = Path(__file__).parent / "data"

@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    p = DATA / name
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()

@st.cache_data
def load_all():
    programs   = load_csv("programs.csv")
    comps      = load_csv("comparators.csv")
    cross      = load_csv("cip_soc_crosswalk.csv")
    matches    = load_csv("vocara_match_library_tx.csv")
    return programs, comps, cross, matches

def two_digit(cip6: str) -> str:
    try:
        return cip6.split(".")[0]
    except Exception:
        return ""

def ug_benchmark(state: str, pct_in_state: float, comps: pd.DataFrame) -> float:
    """Undergrad: compare to HS-only (25–34). State median unless <50% in-state ⇒ US."""
    if comps.empty: return np.nan
    if pct_in_state >= 0.5:
        row = comps[(comps.row_type=="STATE") & (comps.geo==state)].head(1)
        if row.empty:
            row = comps[(comps.row_type=="NATIONAL") & (comps.geo=="US")]
    else:
        row = comps[(comps.row_type=="NATIONAL") & (comps.geo=="US")]
    return float(row.hs_25_34_median.iloc[0]) if not row.empty else np.nan

def grad_prof_benchmark(state: str, pct_in_state: float, field2: str, comps: pd.DataFrame) -> float:
    """Grad/Prof: BA-only (25–34). Lowest of: state BA; BA-in-field (state); BA-in-field (US). If <50% in-state ⇒ lower of US BA or US BA-in-field."""
    if comps.empty: return np.nan
    if pct_in_state >= 0.5:
        state_ba = comps[(comps.row_type=="STATE") & (comps.geo==state)].ba_25_34_median.head(1)
        st_field = comps[(comps.row_type=="STATE_FIELD") & (comps.geo==state) & (comps.field2==field2)].ba_in_field_state.head(1)
        us_field = comps[(comps.row_type=="US_FIELD") & (comps.field2==field2)].ba_in_field_us.head(1)
        cands = [v for v in [state_ba.iloc[0] if len(state_ba) else None,
                              st_field.iloc[0] if len(st_field) else None,
                              us_field.iloc[0] if len(us_field) else None] if pd.notna(v)]
        return float(min(cands)) if cands else np.nan
    else:
        nat_ba  = comps[(comps.row_type=="NATIONAL") & (comps.geo=="US")].ba_25_34_median.head(1)
        us_field= comps[(comps.row_type=="US_FIELD") & (comps.field2==field2)].ba_in_field_us.head(1)
        cands = [v for v in [nat_ba.iloc[0] if len(nat_ba) else None,
                              us_field.iloc[0] if len(us_field) else None] if pd.notna(v)]
        return float(min(cands)) if cands else np.nan

def earnings_test(level: str, state: str, pct_in_state: float, cip6: str, program_median: float, comps: pd.DataFrame):
    field2 = two_digit(cip6)
    bench = ug_benchmark(state, pct_in_state, comps) if level=="UG" else grad_prof_benchmark(state, pct_in_state, field2, comps)
    passes = (program_median >= bench) if pd.notna(bench) else False
    return passes, bench

def roi(program_median: float, bench: float, net_price_year: float, living_cost_year: float, time_years: float):
    total_cost = time_years * (net_price_year + living_cost_year)
    premium = program_median - (bench if pd.notna(bench) else 0)
    annual_premium = max(premium, 0)
    payback = (total_cost / annual_premium) if annual_premium > 0 else np.inf
    # 10-yr NPV of wage premium (real r=3%, growth g=1%)
    r, g = 0.03, 0.01
    npv = -total_cost
    for t in range(1, 11):
        cash = annual_premium * ((1+g)**(t-1))
        npv += cash / ((1+r)**t)
    return dict(total_cost=total_cost, annual_premium=annual_premium, payback_years=payback, npv_10yr=npv)

def performance_score(passed: bool, annual_premium: float, payback_years: float,
                      labor_score: float = 60, purpose_fit: float = 70) -> int:
    """Performance (5th P) = 0–100; adjustable weights."""
    earnings_test_score = 100 if passed else 40
    if annual_premium <= 0:       roi_score = 20
    elif payback_years <= 7:      roi_score = 90
    elif payback_years <= 10:     roi_score = 75
    else:                         roi_score = 50
    perf = 0.30*earnings_test_score + 0.30*roi_score + 0.25*labor_score + 0.15*purpose_fit
    return int(round(perf))

def filter_matches(df: pd.DataFrame, metro: str|None, min_completion_wage: float|None, start_window: str|None):
    if df.empty: return df
    out = df.copy()
    if metro and "metro_or_city" in out.columns:
        out = out[out["metro_or_city"].astype(str).str.contains(metro, case=False, na=False)]
    if min_completion_wage is not None and "wage_completion_annual" in out.columns:
        out["wage_completion_annual"] = pd.to_numeric(out["wage_completion_annual"], errors="coerce")
        out = out[out["wage_completion_annual"].fillna(0) >= float(min_completion_wage)]
    if start_window and "start_windows" in out.columns:
        out = out[out["start_windows"].astype(str).str.contains(start_window, case=False, na=False)]
    return out

programs, comps, cross, matches = load_all()

st.title("Vocara • Earnings Test, ROI & Alternatives (TX-ready)")

tab1, tab2, tab3 = st.tabs(["Run Test", "Alternatives", "Data"])

with tab1:
    st.subheader("A) Test a program (manual input)")
    c1, c2, c3 = st.columns(3)
    cip6  = c1.text_input("CIP-6", "11.0701")   # e.g., 11.0701
    level = c2.selectbox("Level", ["UG","GRAD","PROF"], index=0)
    state = c3.text_input("Institution State", "TX")

    d1, d2, d3 = st.columns(3)
    pct_in_state   = d1.number_input("% Students In-State (0–1)", 0.0, 1.0, 0.7, 0.05)
    program_median = d2.number_input("Program median (4 yrs post)", 0.0, 1_000_000.0, 82000.0, 1000.0)
    time_years     = d3.number_input("Time to degree (years)", 0.5, 8.0, 4.0, 0.5)

    e1, e2 = st.columns(2)
    net_price = e1.number_input("Net price / year (est.)", 0.0, 100_000.0, 9500.0, 500.0)
    living    = e2.number_input("Living cost / year (est.)", 0.0, 100_000.0, 18000.0, 500.0)

    if st.button("Run Earnings Test & ROI", type="primary"):
        passed, bench = earnings_test(level, state, pct_in_state, cip6, program_median, comps)
        metrics = roi(program_median, bench, net_price, living, time_years)
        left, right = st.columns(2)
        with left:
            st.metric("Benchmark (OBBBA)", "—" if pd.isna(bench) else f"${bench:,.0f}")
            st.metric("Program median (4-yr post)", f"${program_median:,.0f}")
            st.write("**Result:**", "✅ PASS" if passed else "❌ FAIL / AT-RISK")
        with right:
            st.metric("Total cost (est.)", f"${metrics['total_cost']:,.0f}")
            st.metric("Annual premium vs. benchmark", f"${metrics['annual_premium']:,.0f}")
            st.metric("Payback period", "N/A" if np.isinf(metrics['payback_years']) else f"{metrics['payback_years']:.1f} years")
            st.metric("10-yr NPV (real)", f"${metrics['npv_10yr']:,.0f}")
        perf = performance_score(passed, metrics["annual_premium"], metrics["payback_years"])
        st.info(f"**Performance (5th P)** composite: **{perf}/100**")

with tab2:
    st.subheader("B) Apprenticeship & Internship Alternatives (Texas)")
    if matches.empty:
        st.warning("Add rows to data/vocara_match_library_tx.csv")
    else:
        c1, c2, c3 = st.columns(3)
        metro = c1.selectbox("Metro", ["", "Houston", "Dallas", "DFW", "Austin", "San Antonio"], index=0)
        min_wage = c2.selectbox("Min completion wage", ["", "50000", "60000", "70000"], index=0)
        window = c3.selectbox("Start window", ["", "rolling", "cohort"], index=0)
        filt = filter_matches(matches, metro or None, float(min_wage) if min_wage else None, window or None)
        st.dataframe(filt, use_container_width=True)

with tab3:
    st.subheader("Data files (upload to override during this session)")
    up1 = st.file_uploader("programs.csv", type="csv", key="p")
    up2 = st.file_uploader("comparators.csv", type="csv", key="c")
    up3 = st.file_uploader("cip_soc_crosswalk.csv", type="csv", key="x")
    up4 = st.file_uploader("vocara_match_library_tx.csv", type="csv", key="m")
    if up1: programs = pd.read_csv(up1); st.success(f"Loaded programs.csv ({len(programs)} rows)")
    if up2: comps    = pd.read_csv(up2);   st.success(f"Loaded comparators.csv ({len(comps)} rows)")
    if up3: cross    = pd.read_csv(up3);   st.success(f"Loaded crosswalk ({len(cross)} rows)")
    if up4: matches  = pd.read_csv(up4);   st.success(f"Loaded matches ({len(matches)} rows)")
    st.caption("Tip: commit these CSVs to /data for persistent deploys.")
