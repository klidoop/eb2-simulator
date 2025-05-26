import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from datetime import datetime
from scipy.stats import lognorm

# 使用真实的EB2排期数据来估算过去PD推进的速度（作为模拟的基础速度）
historical_progress = pd.DataFrame({
    "Cutoff": pd.to_datetime([
        "12/31/2010", "12/31/2011", "12/31/2012", "12/31/2013", "12/31/2014",
        "12/31/2015", "12/31/2016", "12/31/2017", "12/31/2018", "12/31/2019",
        "12/31/2020", "12/31/2021", "12/31/2022", "12/31/2023", "12/31/2024"
    ]),
    "PD": pd.to_datetime([
        "6/8/2006", "3/15/2008", "10/22/2007", "11/8/2008", "1/1/2010",
        "2/1/2012", "9/22/2012", "7/1/2013", "7/1/2015", "6/22/2015",
        "5/1/2016", "1/1/2019", "6/8/2019", "10/22/2019", "3/22/2020"
    ])
})
# 计算PD推进速度
historical_progress["PD Delta"] = historical_progress["PD"].diff().dt.days.div(30).fillna(0)
historical_progress["Time Delta"] = historical_progress["Cutoff"].diff().dt.days.div(30).fillna(0)
historical_progress["Speed"] = historical_progress["PD Delta"] / historical_progress["Time Delta"]
historical_speed_avg = historical_progress["Speed"].mean()

# Preset profiles
presets = {
    "Conservative": dict(base_speed=180, withdrawal_rate=0.08, policy_risk_prob=0.2, positive_policy_boost=0.05, negative_policy_penalty=0.3),
    "Neutral": dict(base_speed=230, withdrawal_rate=0.112, policy_risk_prob=0.1, positive_policy_boost=0.1, negative_policy_penalty=0.15),
    "Aggressive": dict(base_speed=280, withdrawal_rate=0.15, policy_risk_prob=0.05, positive_policy_boost=0.2, negative_policy_penalty=0.05)
}

st.sidebar.header("🔧 Simulation Parameters")
profile = st.sidebar.selectbox("Preset Profile", list(presets.keys()))

# Force update on profile change
if "last_profile" not in st.session_state:
    st.session_state.last_profile = profile
if profile != st.session_state.last_profile:
    st.session_state.params = presets[profile].copy()
    st.session_state.last_profile = profile

# Allow reset
if "params" not in st.session_state or st.sidebar.button("Reset to Defaults"):
    st.session_state.params = presets[profile].copy()

params = st.session_state.params

params["base_speed"] = st.sidebar.number_input("Monthly Processing Base (基准处理速度)", min_value=50, max_value=1000, value=params["base_speed"], key="base_speed")
params["withdrawal_rate"] = st.sidebar.slider("Annual Withdrawal Rate (申请人流失率)", 0.0, 0.3, value=params["withdrawal_rate"], step=0.01, key="withdrawal_rate")
params["policy_risk_prob"] = st.sidebar.slider("Policy Risk Probability (政策变动概率)", 0.0, 0.5, value=params["policy_risk_prob"], step=0.01, key="policy_risk_prob")
params["positive_policy_boost"] = st.sidebar.slider("Positive Policy Boost (%)", 0.0, 0.3, value=params["positive_policy_boost"], step=0.01, key="positive_policy_boost")
params["negative_policy_penalty"] = st.sidebar.slider("Negative Policy Penalty (%)", 0.0, 0.5, value=params["negative_policy_penalty"], step=0.01, key="negative_policy_penalty")

# Assumptions table
assumption_table = pd.DataFrame({
    "Parameter": [
        "Baseline Date",
        "Estimated Backlog (China EB2)",
        "Annual Visa Quota",
        "Family-based Spillover (FY2025)",
        "Applicant Attrition Rate",
        "EB3 Downgrade Probability",
        "Monthly Processing Speed",
        "Policy Risk Probability"
    ],
    "Value / Assumption": [
        "May 2025 Final Action Date = Dec 2020",
        "38,000 cases (incl. dependents)",
        "2,803 per year",
        "1,148 extra (28% of spillover pool)",
        f"{params['withdrawal_rate']*100:.1f}% per year",
        "31% downgrade rate",
        f"{params['base_speed']} ± 70 cases/month",
        f"{params['policy_risk_prob']*100:.1f}% chance"
    ],
    "Source": [
        "US Dept of State Visa Bulletin (May 2025)",
        "CEAC data modeling",
        "INA 203(b), 7% country cap",
        "2024 DOS allocation memo",
        "USCIS 2024 I-485 data",
        "NSC internal downgrade stats (2025)",
        "User adjustable",
        "User adjustable"
    ]
})

st.markdown("### 📊 Assumptions Summary")
st.dataframe(assumption_table, use_container_width=True)

st.title("🇨🇳 EB2 Priority Date Forecast Simulator")
