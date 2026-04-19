# ─────────────────────────────────────────────
# CUSTOMER CHURN DASHBOARD
# Phase 4: Interactive Streamlit App
# ─────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ───────────────────────────────
st.set_page_config(
    page_title="Churn Analytics Dashboard",
    page_icon="📉",
    layout="wide"
)

# ── Load data ─────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("output/churn_predictions.csv")
    df["Tenure_Group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0–12 months", "13–24 months", "25–48 months", "49–72 months"]
    )
    df["Charge_Band"] = pd.cut(
        df["MonthlyCharges"],
        bins=[0, 35, 65, 120],
        labels=["Low (<$35)", "Mid ($35–65)", "High (>$65)"]
    )
    return df

df = load_data()

# ── Sidebar filters ───────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/combo-chart.png", width=60)
st.sidebar.title("Filters")

contract_filter = st.sidebar.multiselect(
    "Contract type",
    options=df["Contract"].unique().tolist(),
    default=df["Contract"].unique().tolist()
)

internet_filter = st.sidebar.multiselect(
    "Internet service",
    options=df["InternetService"].unique().tolist(),
    default=df["InternetService"].unique().tolist()
)

risk_filter = st.sidebar.multiselect(
    "Risk level",
    options=["High", "Medium", "Low"],
    default=["High", "Medium", "Low"]
)

charge_filter = st.sidebar.slider(
    "Monthly charges ($)",
    min_value=int(df["MonthlyCharges"].min()),
    max_value=int(df["MonthlyCharges"].max()),
    value=(int(df["MonthlyCharges"].min()), int(df["MonthlyCharges"].max()))
)

# ── Apply filters ─────────────────────────────
filtered = df[
    df["Contract"].isin(contract_filter) &
    df["InternetService"].isin(internet_filter) &
    df["Risk_Level"].isin(risk_filter) &
    df["MonthlyCharges"].between(charge_filter[0], charge_filter[1])
]

# ── Header ────────────────────────────────────
st.title("📉 Customer Churn Analytics Dashboard")
st.caption("Telco customer churn analysis — powered by Logistic Regression (AUC: 0.841)")
st.divider()

# ── KPI Cards (Row 1) ─────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)

total         = len(filtered)
churned       = filtered["Churn_Binary"].sum()
churn_rate    = churned / total * 100 if total > 0 else 0
high_risk     = (filtered["Risk_Level"] == "High").sum()
monthly_risk  = filtered[filtered["Risk_Level"] == "High"]["MonthlyCharges"].sum()
annual_risk   = monthly_risk * 12

k1.metric("Total customers",     f"{total:,}")
k2.metric("Churned customers",   f"{churned:,}")
k3.metric("Churn rate",          f"{churn_rate:.1f}%")
k4.metric("High-risk customers", f"{high_risk:,}")
k5.metric("Annual revenue at risk", f"${annual_risk:,.0f}")

st.divider()

# ── Row 2: Churn by contract + by tenure ──────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Churn rate by contract type")
    ct = filtered.groupby("Contract")["Churn_Binary"].mean().reset_index()
    ct.columns = ["Contract", "Churn Rate"]
    ct["Churn Rate"] = ct["Churn Rate"] * 100
    ct = ct.sort_values("Churn Rate", ascending=False)
    fig = px.bar(
        ct, x="Contract", y="Churn Rate",
        color="Churn Rate",
        color_continuous_scale=["#2ecc71", "#e67e22", "#e74c3c"],
        text=ct["Churn Rate"].apply(lambda x: f"{x:.1f}%")
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        coloraxis_showscale=False,
        yaxis_title="Churn rate (%)",
        xaxis_title="",
        margin=dict(t=10, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Churn rate by tenure")
    tg = filtered.groupby("Tenure_Group", observed=True)["Churn_Binary"].mean().reset_index()
    tg.columns = ["Tenure Group", "Churn Rate"]
    tg["Churn Rate"] = tg["Churn Rate"] * 100
    fig2 = px.bar(
        tg, x="Tenure Group", y="Churn Rate",
        color="Churn Rate",
        color_continuous_scale=["#2ecc71", "#e67e22", "#e74c3c"],
        text=tg["Churn Rate"].apply(lambda x: f"{x:.1f}%")
    )
    fig2.update_traces(textposition="outside")
    fig2.update_layout(
        coloraxis_showscale=False,
        yaxis_title="Churn rate (%)",
        xaxis_title="",
        margin=dict(t=10, b=10)
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Row 3: Risk distribution + Monthly charges ─
col3, col4 = st.columns(2)

with col3:
    st.subheader("Customer risk distribution")
    risk_counts = filtered["Risk_Level"].value_counts().reset_index()
    risk_counts.columns = ["Risk Level", "Count"]
    color_map = {"High": "#e74c3c", "Medium": "#e67e22", "Low": "#2ecc71"}
    fig3 = px.pie(
        risk_counts, names="Risk Level", values="Count",
        color="Risk Level", color_discrete_map=color_map,
        hole=0.45
    )
    fig3.update_traces(textinfo="percent+label", textfont_size=13)
    fig3.update_layout(
        showlegend=False,
        margin=dict(t=10, b=10)
    )
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.subheader("Monthly charges vs churn probability")
    sample = filtered.sample(min(800, len(filtered)), random_state=42)
    fig4 = px.scatter(
        sample,
        x="MonthlyCharges",
        y="Churn_Probability",
        color="Churn",
        color_discrete_map={"Yes": "#e74c3c", "No": "#3498db"},
        opacity=0.6,
        labels={
            "MonthlyCharges": "Monthly charges ($)",
            "Churn_Probability": "Churn probability",
            "Churn": "Churned"
        }
    )
    fig4.add_hline(y=0.5, line_dash="dash", line_color="gray",
                   annotation_text="Decision boundary (0.5)")
    fig4.update_layout(margin=dict(t=10, b=10))
    st.plotly_chart(fig4, use_container_width=True)

# ── Row 4: Revenue at risk by segment ─────────
st.subheader("Monthly revenue at risk by contract type")
rev = (
    filtered[filtered["Risk_Level"] == "High"]
    .groupby("Contract")["MonthlyCharges"]
    .agg(["sum", "count"])
    .reset_index()
)
rev.columns = ["Contract", "Revenue at Risk ($)", "High-Risk Customers"]
rev = rev.sort_values("Revenue at Risk ($)", ascending=False)

fig5 = px.bar(
    rev, x="Contract", y="Revenue at Risk ($)",
    color="Contract",
    color_discrete_sequence=["#e74c3c", "#e67e22", "#3498db"],
    text=rev["Revenue at Risk ($)"].apply(lambda x: f"${x:,.0f}"),
    hover_data=["High-Risk Customers"]
)
fig5.update_traces(textposition="outside")
fig5.update_layout(
    showlegend=False,
    xaxis_title="",
    margin=dict(t=10, b=10)
)
st.plotly_chart(fig5, use_container_width=True)

# ── Row 5: High-risk customer table ───────────
st.subheader("🔴 High-risk customer list")
st.caption("Customers with churn probability > 60% — prioritise these for retention outreach")

high_risk_df = filtered[filtered["Risk_Level"] == "High"][[
    "customerID", "Contract", "tenure", "MonthlyCharges",
    "InternetService", "Churn_Probability", "Churn"
]].copy()

high_risk_df.columns = [
    "Customer ID", "Contract", "Tenure (months)",
    "Monthly Charges ($)", "Internet Service",
    "Churn Probability", "Actually Churned"
]

high_risk_df = high_risk_df.sort_values("Churn Probability", ascending=False)
high_risk_df["Churn Probability"] = high_risk_df["Churn Probability"].apply(
    lambda x: f"{x:.1%}"
)
high_risk_df["Monthly Charges ($)"] = high_risk_df["Monthly Charges ($)"].apply(
    lambda x: f"${x:.2f}"
)

st.dataframe(
    high_risk_df.head(50),
    use_container_width=True,
    hide_index=True
)

# ── Footer ────────────────────────────────────
st.divider()
st.caption(
    "Model: Logistic Regression  |  AUC: 0.841  |  "
    "Dataset: IBM Telco Customer Churn (7,043 customers)  |  "
    "Built with Python, Scikit-learn & Streamlit"
)