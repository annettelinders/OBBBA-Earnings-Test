# data_loader.py
import os
import pandas as pd
import streamlit as st
from supabase import create_client

@st.cache_resource
def supabase_client():
    # Works in Render (env vars) and Bolt (.streamlit/secrets.toml)
    url = st.secrets.get("SUPABASE_URL", os.environ.get("SUPABASE_URL"))
    key = st.secrets.get("SUPABASE_ANON_KEY", os.environ.get("SUPABASE_ANON_KEY"))
    if not url or not key:
        raise RuntimeError("Missing SUPABASE_URL / SUPABASE_ANON_KEY. Add them in Render env vars or .streamlit/secrets.toml")
    return create_client(url, key)

@st.cache_data(ttl=600)
def fetch_table(name: str, select: str="*") -> pd.DataFrame:
    sb = supabase_client()
    res = sb.table(name).select(select).execute()
    return pd.DataFrame(res.data or [])

def cip_to_series(cip6: str) -> str:
    # 11.0701 -> 11.0700 (Scorecard FoS is 4-digit series)
    try:
        return cip6[:5] + "00"
    except Exception:
        return cip6
