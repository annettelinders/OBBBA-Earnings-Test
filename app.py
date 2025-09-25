# app.py — Vocara • Earnings Test, ROI & Alternatives (Supabase-powered)

import os
import numpy as np
import pandas as pd
import streamlit as st
from supabase import create_client

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Vocara • Earnings Test, ROI & Alternatives", layout="wide")


# -------------------------
# Supabase client + loaders
# -------------------------
@st.cache_resource
def sb_client():
    """Create a Supabase client using Streamlit secrets or environment variables."""
    url = st.secrets.get("SUPABASE_URL", os.environ.get("SUPABASE_URL"))
    key = st.secrets.get("SUPABASE_ANON_KEY", os.environ.get("SUPABASE_ANON_KEY"))
    if not url or not key:
        raise RuntimeError(
            "Missing SUPABASE_URL / SUPABASE_ANON_KEY. "
            "Add them in Render environment or .streamlit/secrets.toml"
        )
    return create_client(url, key)

@st.cache_data(ttl=600)
def fetch_table(name: str, select: str = "*") -> pd.DataFrame:
    """Read a whole table from Supabase; return empty DF if no rows."""
    try:
        res = sb_client().table(name).select(select).execute()
        return pd.DataFrame(res.data or [])
    except Exception as e:
        st.error(f"Could not load table '{name}': {e}")
        return pd.DataFrame()

def cip_to_series(cip6: str) -> str:
    """Turn '11.0701' into the Scorecard 4-digit series '11.0700'."""
    try:
        return cip6[:5] + "00"
    except Exception:
        return cip6


# -------------------------
# Load data from Supabase
# (must match the schemas we created)
# -------------------------
programs_tsu   = fetch_table("programs_tsu")
comparators    = fetch_table("comparators")
crosswalk      = fetch_table("cip_soc_crosswalk")
tx_publics     = fetch_table("programs_tx_publics")        # optional (can be empty)
regional_pubs  = fetch_table("programs_regional_publics")  # optional (can be empty)
ooh_wages      = fetch_table("bls_ooh_wages")              # optional (can be empty)
matches        = fetch_table("vocara_match_library_tx")    # optional (can be empty)

# Small status line so we can confirm data is loaded
st.caption(
    f"Loaded ➜ TSU={len(programs_tsu)} | comparators={len(comparators)} | "
    f"crosswalk={len(crosswalk)} | TX publics={len(tx_publics)} | regional={len(regional_pubs)}"
)


# -------------------------
# OBBBA comparators
# -------------------------
def ug_benchmark(state: str, pct_in_state: float) -> float:
    """
    Undergraduate benchmark = HS-only (age 25–34) median earnings.
    Use STATE unless <50% in-state, then NATIONAL.
    """
    if comparators.empty:
        return np.nan
    if pct_in_state >= 0.5:
        row = comparators.query("row_type=='STATE' and geo==@state")
        if row.empty:
            row = comparators.query("row_type=='NATIONAL' and geo=='US'")
    else:
        row = comparators.query("row_type=='NATIONAL' and geo=='US'")
    return float(row.iloc[0]["hs_25_34_median"]) if not row.empty else np.nan


def grad_prof_benchmark(state: str, pct_in_state: float, field2: str) -> float:
    """
    Graduate/Professional benchmark = BA-only (25–34) using the LOWEST of:
    - state BA median
    - BA-in-field (STATE)
    - BA-in-field (US)
    If <50% in-state, use the LOWER of:
    - national BA median
    - national BA-in-field
    """
    if comparators.empty:
        return np.nan

    if pct_in_state >= 0.5:
        cands = []
        srow = comparators.query("row_type=='STATE' and geo==@state")
        if not srow.empty and pd.notna(srow.iloc[0]["ba_25_34_median"]):
            cands.append(float(srow.iloc[0]["ba_25_34_median"]))
        sf = comparators.query("row_type=='STATE_FIELD' and geo==@state and field2==@field2")
        if not sf.empty and pd.notna(sf.iloc[0]["ba_in_field_state"]):
            cands.append(float(sf.iloc[0]["ba_in_field_state"]))
        uf = comparators.query("row_type=='US_FIELD' and geo=='US' and field2==@field2")
        if not uf.empty and pd.notna(uf.iloc[0]["ba_in_field_us"]):
            cands.append(float(uf.iloc[0]["ba_in_field_us"]))
        return min(cands) if cands else np.nan
    else:
        cands = []
        nrow = comparators.query("row_type=='NATIONAL' and geo=='US'")
        if not nrow.empty and pd.notna(nrow.iloc[0]["ba_25_34_median"]):
            cands.append(float(nrow.iloc[0]["ba_25_34_median"]))
        uf = comparators.query("row_type=='US_FIELD' and geo=='US' and field2==@field2")
        if not uf.empty and pd.notna(uf.iloc[0]["ba_in_field_us"]):
            cands.append(float(uf.iloc[0]["ba_in_field_us"]))
        return min(cands) if cands else np.nan


def earnings_test(level: str, state: str, pct_in_state: float, cip6: str, program_median: float):
    """Return (pass_bool, benchmark_value)."""
    series = cip_to_series(cip6)
    field2 = series[:2]
    bench = ug_benchmark(state, pct_in_state) if level == "UG" else grad_prof_benchmark(state, pct_in_state, field2)
    passed = (program_median >= bench) if pd.notna(bench) else False
    return passed, bench


# -------------------------
# ROI calculations
# -------------------------
def roi(program_median: float, bench: float, net_price_year: float, living_cost_year: float, time_years: float):
    total_cost = time_years * (net_price_year + living_cost_year)
    premium = (program_median - bench) if pd.notna(bench) else 0.0
    premium = max(premium, 0.0)
    payback = (total_cost / premium) if premium > 0 else np.inf
    # Simple 10-yr NPV of wage premium (real r=3%, growth g=1%)
    r, g = 0.03, 0.01
    npv = -total_cost
    for t in range(1, 11):
        cash = premium * ((1 + g) ** (t - 1))
        npv += cash / ((1 + r) ** t)
    return dict(total_cost=total_cost, annual_premium=premium, payback_years=payback, npv_10yr=npv)


def performance_score(passed: bool, annual_premium: float, payback_years: float,
                      labor_score: float = 60, purpose_fit: float = 70) -> int:
    """Composite 'Performance' (5th P) from 0–100."""
    test_score = 100 if passed else 40
    if annual_premium <= 0:
        roi_score = 20
    elif payback_years <= 7:
        roi_score = 90
    elif payback_years <= 10:
        roi_score = 75
    else:
        roi_score = 50
    perf = 0.30 * test_score + 0.30 * roi_score + 0.25 * labor_score + 0.15 * purpose_fit
    return int(round(perf))


# -------------------------
# Fallback helpers
# -------------------------
def tsu_row(cip6: str, level: str):
    """Exact match first, then CIP series fallback."""
    if programs_tsu.empty:
        return None
    series = cip_to_series(cip6)
    df = programs_tsu.query("cip6==@cip6 and level==@level")
    if df.empty:
        df = programs_tsu.query("cip6==@series and level==@level")
    return None if df.empty else df.iloc[0]

def tsu_alternatives(cip6: str, level: str, k: int = 3) -> pd.DataFrame:
    if programs_tsu.empty:
        return pd.DataFrame()
    series2 = cip_to_series(cip6)[:5]  # 'NN.NN'
    alts = programs_tsu[(programs_tsu["level"] == level) & (programs_tsu["cip6"].astype(str).str.startswith(series2))]
    if alts.empty:
        alts = programs_tsu[programs_tsu["level"] == level]
    alts = alts.sort_values("median_earn_4yr", ascending=False)
    return alts.head(k)

def tx_public_alternatives(cip6: str, level: str, k: int = 3) -> pd.DataFrame:
    if tx_publics.empty:
        return pd.DataFrame()
    series2 = cip_to_series(cip6)[:5]
    df = tx_publics[(tx_publics["level"] == level) & (tx_publics["cip6"].astype(str).str.startswith(series2))]
    return df.sort_values("median_earn_4yr", ascending=False).head(k)

def regional_alternatives(cip6: str, level: str, k: int = 3) -> pd.DataFrame:
    if regional_pubs.empty:
        return pd.DataFrame()
    series2 = cip_to_series(cip6)[:5]
    df = regional_pubs[(regional_pubs["level"] == level) & (regional_pubs["cip6"].astype(str).str.startswith(series2))]
    return df.sort_values("median_earn_4yr", ascending=False).head(k)

def ooh_fallback(cip6: str, k: int = 3):
    """
    Map CIP -> SOC via crosswalk and return up to k occupations with any cached wages.
    If no wages cached, still return SOCs and point to OOH site.
    """
    if crosswalk.empty:
        return []
    series = cip_to_series(cip6)
    socs = crosswalk[crosswalk["cip6"] == series]["soc_code"].dropna().unique().tolist()
    if not socs:
        field2 = series[:2]
        socs = crosswalk[crosswalk["cip6"].astype(str).str.startswith(field2)]["soc_code"].dropna().unique().tolist()
    out = []
    for s in socs:
        w = ooh_wages[ooh_wages["soc_code"] == s] if not ooh_wages.empty else pd.DataFrame()
        if not w.empty and pd.notna(w.iloc[0].get("median_annual_wage")):
            out.append({"soc_code": s, "median_wage": float(w.iloc[0]["median_annual_wage"]),
                        "source": w.iloc[0].get("source_url", "https://www.bls.gov/ooh/occupation-finder.htm")})
        else:
            out.append({"soc_code": s, "median_wage": None,
                        "source": "https://www.bls.gov/ooh/occupation-finder.htm"})
    return out[:k]


# -------------------------
# UI
# -------------------------
st.title("Vocara • Earnings Test, ROI & Alternatives")

tab1, tab2, tab3 = st.tabs(["Run Test", "Alternatives", "Data"])

with tab1:
    st.subheader("A) Test a program (TSU-first, with fallbacks)")

    c1, c2, c3 = st.columns(3)
    cip6  = c1.text_input("CIP-6 (e.g., 11.0701)", "11.0701")
    level = c2.selectbox("Level", ["UG", "GRAD", "PROF"], index=0)
    state = c3.text_input("Institution State", "TX")

    d1, d2, d3 = st.columns(3)
    pct_in_state = d1.number_input("% Students In-State (0–1)", 0.0, 1.0, 0.70, 0.05)
    net_price    = d2.number_input("Net price / year (est.)", 0.0, 100_000.0, 9_500.0, 500.0)
    living       = d3.number_input("Living cost / year (est.)", 0.0, 100_000.0, 18_000.0, 500.0)

    if st.button("Run Earnings Test & ROI", type="primary"):
        row = tsu_row(cip6, level)
        if row is not None and pd.notna(row.get("median_earn_4yr", np.nan)):
            prog_med = float(row["median_earn_4yr"])
            ty = float(row.get("time_years", 4.0))
            passed, bench = earnings_test(level, state, float(pct_in_state), cip6, prog_med)
            metrics = roi(prog_med, bench, float(net_price), float(living), ty)
            left, right = st.columns(2)
            with left:
                st.metric("Benchmark (OBBBA)", "—" if pd.isna(bench) else f"${bench:,.0f}")
                st.metric("Program median (4-yr post)", f"${prog_med:,.0f}")
                st.write("**Result:**", "✅ PASS" if passed else "❌ FAIL / AT-RISK")
            with right:
                st.metric("Total cost (est.)", f"${metrics['total_cost']:,.0f}")
                st.metric("Annual premium vs benchmark", f"${metrics['annual_premium']:,.0f}")
                st.metric("Payback period",
                          "N/A" if np.isinf(metrics["payback_years"]) else f"{metrics['payback_years']:.1f} years")
                st.metric("10-yr NPV (real)", f"${metrics['npv_10yr']:,.0f}")
            perf = performance_score(passed, metrics["annual_premium"], metrics["payback_years"])
            st.info(f"**Performance (5th P)** composite: **{perf}/100**")

        else:
            st.warning("No TSU salary data available for this program. Searching alternatives…")

            # (1) Closest TSU alternatives
            alts = tsu_alternatives(cip6, level, 3)
            if not alts.empty:
                st.success("Closest **TSU** alternatives (same CIP family or top-earning TSU):")
                st.dataframe(alts[["program_name", "cip6", "level", "median_earn_4yr"]])
            else:
                # (2) Texas publics (if table populated)
                tx_alts = tx_public_alternatives(cip6, level, 3)
                if not tx_alts.empty:
                    st.success("Closest **Texas public** alternatives:")
                    keep = [c for c in ["inst_name", "cip6", "level", "median_earn_4yr"] if c in tx_alts.columns]
                    st.dataframe(tx_alts[keep])
                else:
                    # (3) Regional publics (if table populated)
                    reg_alts = regional_alternatives(cip6, level, 3)
                    if not reg_alts.empty:
                        st.success("Closest **regional public** alternatives:")
                        keep = [c for c in ["inst_name", "state", "cip6", "level", "median_earn_4yr"] if c in reg_alts.columns]
                        st.dataframe(reg_alts[keep])
                    else:
                        # (4) OOH fallback (CIP→SOC)
                        st.error("No program earnings data found. Using BLS Occupational Outlook for ROI guidance:")
                        ooh = ooh_fallback(cip6, 3)
                        for o in ooh:
                            wage_txt = "N/A" if (o["median_wage"] is None or pd.isna(o["median_wage"])) else f"${o['median_wage']:,.0f}"
                            st.write(f"- SOC **{o['soc_code']}** — median wage {wage_txt}  |  Source: {o['source']}")
                        st.caption("ROI is illustrative when based on occupational wages. "
                                   "Use this to guide efficient, cost-effective paths (e.g., apprenticeship/cert + internship).")

with tab2:
    st.subheader("B) Apprenticeship & Internship Alternatives (Texas)")
    if matches.empty:
        st.info("No apprenticeship/internship table loaded yet (vocara_match_library_tx).")
    else:
        c1, c2, c3 = st.columns(3)
        metro = c1.selectbox("Metro", ["", "Houston", "Dallas", "DFW", "Austin", "San Antonio"], index=0)
        min_wage = c2.selectbox("Min completion wage", ["", "50000", "60000", "70000"], index=0)
        window = c3.selectbox("Start window", ["", "rolling", "cohort"], index=0)

        def _flt(x): return float(x) if x else None
        df = matches.copy()
        if "metro_or_city" in df.columns and metro:
            df = df[df["metro_or_city"].astype(str).str.contains(metro, case=False, na=False)]
        if "wage_completion_annual" in df.columns and min_wage:
            df["wage_completion_annual"] = pd.to_numeric(df["wage_completion_annual"], errors="coerce")
            df = df[df["wage_completion_annual"].fillna(0) >= _flt(min_wage)]
        if "start_windows" in df.columns and window:
            df = df[df["start_windows"].astype(str).str.contains(window, case=False, na=False)]
        st.datafr
