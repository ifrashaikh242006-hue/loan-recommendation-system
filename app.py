"""
Smart Loan Recommendation System Using Data Science
Research Project — Ifra Shaikh (Roll No: TYDS2024067) | Mentor: Shweta Maitri

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

import re
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Loan Recommendation System",
    page_icon="💰",
    layout="wide",
)

# ----------------------------------------------------------------------
# DATA LOADING + CLEANING
# ----------------------------------------------------------------------
RAW_PATH = "loan_data_raw.csv"


def parse_amount(value):
    """Convert strings like '₹1 Lakh', 'Up to ₹35 Lakh', 25000, 'N/A' into a rupee float."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value)
    if "N/A" in s or s.strip() == "":
        return None
    nums = re.findall(r"\d+\.?\d*", s.replace(",", ""))
    if not nums:
        return None
    vals = [float(n) for n in nums]
    multiplier = 1
    if "lakh" in s.lower():
        multiplier = 100_000
    elif "crore" in s.lower():
        multiplier = 10_000_000
    vals = [v * multiplier for v in vals]
    return max(vals)


def parse_amount_min(value):
    """Same as parse_amount but takes the lower number in a range (for Minimum Loan fields)."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value)
    if "N/A" in s or s.strip() == "":
        return None
    nums = re.findall(r"\d+\.?\d*", s.replace(",", ""))
    if not nums:
        return None
    vals = [float(n) for n in nums]
    multiplier = 100_000 if "lakh" in s.lower() else (10_000_000 if "crore" in s.lower() else 1)
    vals = [v * multiplier for v in vals]
    return min(vals)


def parse_interest(value):
    """Return (min_rate, max_rate) from strings like '10%-15% p.a.' or 'From 10.99% p.a.'."""
    nums = re.findall(r"\d+\.?\d*", str(value))
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    elif len(nums) == 1:
        return float(nums[0]), None
    return None, None


def parse_tenure_months(value):
    """Convert '6 years', '12 months', 'N/A' into a number of months."""
    if value is None:
        return None
    s = str(value).lower()
    if "n/a" in s:
        return None
    nums = re.findall(r"\d+\.?\d*", s)
    if not nums:
        return None
    val = float(nums[0])
    if "year" in s:
        return val * 12
    return val


@st.cache_data
def load_data():
    df = pd.read_csv(RAW_PATH)

    df["Min Interest %"], df["Max Interest %"] = zip(*df["Interest Rate"].map(parse_interest))
    df["Avg Interest %"] = df[["Min Interest %", "Max Interest %"]].mean(axis=1)

    df["Min Loan (₹)"] = df["Minimum Loan"].map(parse_amount_min)
    df["Max Loan (₹)"] = df["Maximum Loan"].map(parse_amount)

    df["Min Tenure (months)"] = df["Minimum Tenure"].map(parse_tenure_months)
    df["Max Tenure (months)"] = df["Maximum Tenure"].map(parse_tenure_months)

    return df


df = load_data()

# ----------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------
st.title("💰 Smart Loan Recommendation System")
st.caption("A Data Science–based decision-support tool for comparing personal loan options across banks")
st.caption("By Ifra Shaikh | Roll No: TYDS2024067 | Mentor: Shweta Maitri")

with st.expander("ℹ️ About this research project", expanded=False):
    st.markdown(
        """
**Title:** Smart Loan Recommendation System Using Data Science
**Name:** Ifra Shaikh | **Roll No:** TYDS2024067 | **Mentor:** Shweta Maitri

**Problem:** Loan information is scattered across different bank websites, making manual
comparison time-consuming and confusing for customers.

**Aim:** To develop a Data Science–based loan recommendation system that helps users find
suitable loan options according to their requirements.

**Scope:** This tool focuses on *loan recommendation*, not loan approval. It acts as a
decision-support system based on publicly available bank data and does not guarantee
eligibility or approval.
        """
    )

st.divider()

# ----------------------------------------------------------------------
# SIDEBAR — USER INPUTS
# ----------------------------------------------------------------------
st.sidebar.header("🔎 Tell us what you need")

loan_amount = st.sidebar.number_input(
    "Loan amount required (₹)",
    min_value=10_000,
    max_value=10_000_000,
    value=500_000,
    step=10_000,
)

tenure_years = st.sidebar.slider(
    "Preferred repayment tenure (years)",
    min_value=1,
    max_value=10,
    value=5,
)

sort_by = st.sidebar.selectbox(
    "Sort recommendations by",
    ["Lowest Interest Rate", "Highest Loan Limit", "Shortest Max Tenure"],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Note: Minimum Loan / Tenure marked 'N/A' by the bank is treated as no lower limit."
)

# ----------------------------------------------------------------------
# RECOMMENDATION LOGIC
# ----------------------------------------------------------------------
tenure_months = tenure_years * 12


def matches(row):
    # Loan amount check
    min_ok = pd.isna(row["Min Loan (₹)"]) or loan_amount >= row["Min Loan (₹)"]
    max_ok = pd.isna(row["Max Loan (₹)"]) or loan_amount <= row["Max Loan (₹)"]
    # Tenure check
    tmin_ok = pd.isna(row["Min Tenure (months)"]) or tenure_months >= row["Min Tenure (months)"]
    tmax_ok = pd.isna(row["Max Tenure (months)"]) or tenure_months <= row["Max Tenure (months)"]
    return min_ok and max_ok and tmin_ok and tmax_ok


matched = df[df.apply(matches, axis=1)].copy()

sort_map = {
    "Lowest Interest Rate": ("Avg Interest %", True),
    "Highest Loan Limit": ("Max Loan (₹)", False),
    "Shortest Max Tenure": ("Max Tenure (months)", True),
}
sort_col, ascending = sort_map[sort_by]
matched = matched.sort_values(sort_col, ascending=ascending, na_position="last")

# ----------------------------------------------------------------------
# RESULTS
# ----------------------------------------------------------------------
st.subheader(f"📋 Recommended Banks ({len(matched)} match{'es' if len(matched) != 1 else ''} found)")

if matched.empty:
    st.warning(
        "No bank in the current dataset matches this exact amount/tenure combination. "
        "Showing the full dataset instead so you can compare manually."
    )
    display_df = df
else:
    display_df = matched

show_cols = [
    "Bank", "Loan Type", "Interest Rate", "Minimum Loan", "Maximum Loan",
    "Minimum Tenure", "Maximum Tenure", "Processing Fee", "Source", "Data Source Type",
]
st.dataframe(display_df[show_cols], use_container_width=True, hide_index=True)

if not matched.empty:
    best = matched.iloc[0]
    st.success(
        f"🏆 **Best match:** {best['Bank']} — {best['Interest Rate']} interest, "
        f"loan up to {best['Maximum Loan']}, tenure up to {best['Maximum Tenure']}."
    )

st.divider()

# ----------------------------------------------------------------------
# VISUALIZATIONS
# ----------------------------------------------------------------------
st.subheader("📊 Comparing Banks")

col1, col2 = st.columns(2)

with col1:
    fig1 = go.Figure()
    for _, row in df.iterrows():
        fig1.add_trace(go.Bar(
            x=[row["Bank"]],
            y=[(row["Max Interest %"] or row["Min Interest %"]) - row["Min Interest %"]],
            base=row["Min Interest %"],
            name=row["Bank"],
            showlegend=False,
            hovertext=f"{row['Interest Rate']}",
        ))
    fig1.update_layout(
        title="Interest Rate Range by Bank (%)",
        yaxis_title="Interest Rate (%)",
        xaxis_title="Bank",
    )
    fig1.update_traces(marker_color="#E07A5F")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.bar(
        df.sort_values("Max Loan (₹)", ascending=False),
        x="Bank",
        y="Max Loan (₹)",
        title="Maximum Loan Amount by Bank (₹)",
        text_auto=".2s",
    )
    st.plotly_chart(fig2, use_container_width=True)

fig3 = px.scatter(
    df,
    x="Avg Interest %",
    y="Max Loan (₹)",
    size="Max Tenure (months)",
    color="Bank",
    hover_data=["Interest Rate", "Maximum Loan", "Maximum Tenure"],
    title="Interest Rate vs. Loan Limit (bubble size = max tenure)",
)
st.plotly_chart(fig3, use_container_width=True)

st.divider()
st.caption(
    "⚠️ Data collected from publicly available bank sources for academic purposes only. "
    "This tool does not guarantee loan approval or reflect real-time rates."
)
