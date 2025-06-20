# EB2 Priority Date Simulator (Updated with Real Backlog Data)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy.stats import lognorm

st.set_page_config(page_title="EB2 China Backlog Simulator", layout="wide")

# Historical progress data
historical_progress = pd.DataFrame({
    "Cutoff": pd.to_datetime([
        "12/31/2010", "12/31/2011", "12/31/2012", "12/31/2013", "12/31/2014",
        "12/31/2015", "12/31/2016", "12/31/2017", "12/31/2018", "12/31/2019",
        "12/31/2020", "12/31/2021", "12/31/2022", "12/31/2023", "12/31/2024",
        "6/9/2025"
    ]),
    "PD": pd.to_datetime([
        "6/8/2006", "3/15/2008", "10/22/2007", "11/8/2008", "1/1/2010",
        "2/1/2012", "9/22/2012", "7/1/2013", "7/1/2015", "6/22/2015",
        "5/1/2016", "1/1/2019", "6/8/2019", "10/22/2019", "3/22/2020",
        "12/15/2020"
    ])
})

historical_progress["PD Delta"] = historical_progress["PD"].diff().dt.days.div(30).fillna(0)
historical_progress["Time Delta"] = historical_progress["Cutoff"].diff().dt.days.div(30).fillna(0)
historical_progress["Speed"] = historical_progress["PD Delta"] / historical_progress["Time Delta"]
historical_speed_avg = historical_progress["Speed"].mean()

presets = {
    "Conservative": dict(base_speed=180, withdrawal_rate=0.08, policy_risk_prob=0.2, positive_policy_boost=0.05, negative_policy_penalty=0.3, spillover=1148),
    "Neutral": dict(base_speed=230, withdrawal_rate=0.112, policy_risk_prob=0.1, positive_policy_boost=0.1, negative_policy_penalty=0.15, spillover=1148),
    "Aggressive": dict(base_speed=280, withdrawal_rate=0.15, policy_risk_prob=0.05, positive_policy_boost=0.2, negative_policy_penalty=0.05, spillover=1148)
}

st.sidebar.header("🔧 Simulation Parameters")
profile = st.sidebar.selectbox("Preset Profile", list(presets.keys()))
if "last_profile" not in st.session_state:
    st.session_state.last_profile = profile
if profile != st.session_state.last_profile:
    st.session_state.params = presets[profile].copy()
    st.session_state.last_profile = profile

if "params" not in st.session_state or st.sidebar.button("Reset to Defaults"):
    st.session_state.params = presets[profile].copy()

params = st.session_state.params
params.setdefault("spillover", 1148)
params["base_speed"] = st.sidebar.number_input("Monthly Processing Base", 50, 1000, value=params["base_speed"], key="base_speed")
params["withdrawal_rate"] = st.sidebar.slider("Annual Withdrawal Rate", 0.0, 0.3, value=params["withdrawal_rate"], step=0.01, key="withdrawal_rate")
params["policy_risk_prob"] = st.sidebar.slider("Policy Risk Probability", 0.0, 0.5, value=params["policy_risk_prob"], step=0.01, key="policy_risk_prob")
params["positive_policy_boost"] = st.sidebar.slider("Positive Policy Boost (%)", 0.0, 0.3, value=params["positive_policy_boost"], step=0.01, key="positive_policy_boost")
params["negative_policy_penalty"] = st.sidebar.slider("Negative Policy Penalty (%)", 0.0, 0.5, value=params["negative_policy_penalty"], step=0.01, key="negative_policy_penalty")
params["spillover"] = st.sidebar.number_input("Family-Based Spillover to EB2-China", 0, 10000, value=params["spillover"], key="spillover")
params["backlog_total"] = st.sidebar.number_input("Backlog Cases (USCIS Data)", 10000, 100000, value=34701, step=500, key="backlog_total")

params["monthly_quota"] = (2803 + params["spillover"]) / 12
backlog_mode = st.sidebar.selectbox("Backlog Scenario", ["Optimistic", "Neutral", "Pessimistic"], index=1)

st.title("🇨🇳 EB2 Priority Date Forecast Simulator")

assumption_table = pd.DataFrame({
    "Parameter": [
        "Baseline Date",
        "Estimated Backlog (China EB2)",
        "Annual Visa Quota",
        "Family-based Spillover",
        "Applicant Attrition Rate",
        "EB3 Downgrade Probability",
        "Monthly Processing Speed",
        "Policy Risk Probability"
    ],
    "Value / Assumption": [
        "May 2025 Final Action Date = Dec 2020",
        f"{params['backlog_total']} cases (USCIS Dec 2024)",
        "2,803 per year",
        f"{params['spillover']} extra (user-defined)",
        f"{params['withdrawal_rate']*100:.1f}% per year",
        "31% downgrade rate",
        f"{params['base_speed']} ± 70 cases/month",
        f"{params['policy_risk_prob']*100:.1f}% chance"
    ],
    "Source": [
        "US Dept of State Visa Bulletin",
        "USCIS performance data",
        "INA 203(b)",
        "User input",
        "USCIS I-485 data",
        "NSC downgrade stats",
        "User adjustable",
        "User adjustable"
    ]
})
st.markdown("### 📊 Assumptions Summary")
st.dataframe(assumption_table, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    target_pd = st.date_input("Your Priority Date", value=datetime(2022, 11, 1))
with col2:
    trials = st.slider("Number of Simulations", 100, 2000, value=300, step=100)
    run_simulation_right = st.button("🚀 Run Simulation 🚀", type="primary")

class EB2Predictor:
    def __init__(self, target_pd, backlog_mode, params):
        self.target_pd = pd.to_datetime(target_pd)
        self.backlog_mode = backlog_mode
        self.params = params
        self.speed_variation = lognorm(s=0.25)
        self.policy_impact = {
            'positive': [0.02, params['positive_policy_boost']],
            'negative': [-params['negative_policy_penalty'], -0.05]
        }
        self.histogram = self._generate_backlog()

    def _generate_backlog(self):
        months = pd.date_range('2020-12', '2023-01', freq='MS')
        shape = {
            "Optimistic": np.linspace(1.0, 0.5, len(months)),
            "Neutral": np.linspace(1.2, 0.4, len(months)),
            "Pessimistic": np.linspace(1.5, 0.3, len(months))
        }[self.backlog_mode]
        shape = shape / shape.sum() * self.params["backlog_total"]
        return pd.Series(shape.astype(int), index=months)

    def _policy_adjustment(self):
        if np.random.rand() < self.params['policy_risk_prob']:
            direction = np.random.choice(['positive', 'negative'], p=[0.3, 0.7])
            return 1 + np.random.uniform(*self.policy_impact[direction])
        return 1

    def simulate_once(self):
        backlog = self.histogram.copy()
        current = 0
        target_idx = np.searchsorted(backlog.index, self.target_pd)
        months = 0

        while current < target_idx and months < 120:
            policy_factor = self._policy_adjustment()
            actual_speed = int(min(
                self.params['monthly_quota'],
                self.params['base_speed'] * historical_speed_avg * self.speed_variation.rvs() * policy_factor
            ))
            withdrawals = int(backlog.sum() * (self.params['withdrawal_rate'] / 12))
            actual_speed += withdrawals

            for i in range(current, len(backlog)):
                if actual_speed <= 0:
                    break
                if backlog.iloc[i] <= actual_speed:
                    actual_speed -= backlog.iloc[i]
                    backlog.iloc[i] = 0
                    current = i + 1
                else:
                    backlog.iloc[i] -= actual_speed
                    actual_speed = 0
            months += 1

        return months

    def simulate(self, n):
        return pd.Series([self.simulate_once() for _ in range(n)])

if "simulation_history" not in st.session_state:
    st.session_state.simulation_history = []

if run_simulation_right:
    with st.spinner("Running simulation..."):
        model = EB2Predictor(target_pd=target_pd, backlog_mode=backlog_mode, params=params)
        results = model.simulate(trials)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(results, bins=30, kde=True, ax=ax, color='skyblue')
    ax.axvline(results.median(), color='red', linestyle='--', label=f'Median: {results.median()} months')
    ax.set_title("Projected Wait Time Distribution")
    ax.set_xlabel("Months to Current")
    ax.set_ylabel("Simulation Count")
    ax.legend()
    st.pyplot(fig)

    projected_date = pd.to_datetime("2025-05") + pd.DateOffset(months=int(results.median()))
    st.session_state.simulation_history.append({
        "Profile": profile,
        "Backlog": backlog_mode,
        "Spillover": params["spillover"],
        "MonthlySpeed": params["base_speed"],
        "WithdrawalRate": params["withdrawal_rate"],
        "MedianMonths": int(results.median()),
        "MinMonths": int(results.min()),
        "MaxMonths": int(results.max()),
        "ProjectedDate": projected_date.strftime('%Y-%m')
    })

    st.markdown(f"""
    ### 🧠 Simulation Summary
    - Median wait time: **{int(results.median())} months**
    - Expected PD becomes current: **{projected_date.strftime('%Y-%m')}**
    - Range: {int(results.min())} to {int(results.max())} months
    - Assumption Mode: **{backlog_mode}**
    """)

    if st.session_state.simulation_history:
        st.markdown("### 📂 Comparison of Saved Simulations")
        hist_df = pd.DataFrame(st.session_state.simulation_history)
        st.dataframe(hist_df)
