import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy.stats import lognorm
from scipy.interpolate import UnivariateSpline

# ==== 参数配置 ====
historical_approvals_by_pd = pd.DataFrame({
    'PD': pd.to_datetime([
        "2020-12-15", "2021-01-01", "2021-06-01", "2021-12-01", "2022-06-01", "2022-12-01", "2023-06-01"
    ]),
    'Approved_I140_Main': [12000, 13000, 15500, 21000, 26000, 29762, 33000]
})

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

params["monthly_quota"] = (2803 + params["spillover"]) / 12
params["monthly_retention"] = (1 - params["withdrawal_rate"]) ** (1/12)

st.title("🇨🇳 EB2 Priority Date Forecast Simulator")

col1, col2 = st.columns(2)
with col1:
    use_pd = st.checkbox("Estimate from Priority Date (根据PD估算排位)", value=True)
    if use_pd:
        pd_input = st.date_input("Your Priority Date (你的优先日)", value=datetime(2022, 11, 10))
        multiplier = st.number_input("Family Multiplier (配偶子女占位因子)", min_value=1.0, max_value=3.0, value=1.7, step=0.1)

        def estimate_position_by_pd(pd_date):
            df = historical_approvals_by_pd.sort_values("PD")
            timestamps = df["PD"].astype('int64') / 1e9
            approvals = df["Approved_I140_Main"]
            spline = UnivariateSpline(timestamps, approvals, s=0.5 * len(df))
            estimated_main = spline(pd_date.timestamp())
            return int(estimated_main * multiplier)
            
        # ✅ 新增这段类型转换处理
    if isinstance(pd_date, datetime.date) and not isinstance(pd_date, datetime):
        pd_date = datetime.combine(pd_date, datetime.min.time())

    estimated_main = spline(pd_date.timestamp())
    return int(estimated_main * multiplier)

        user_position = estimate_position_by_pd(pd_input)
        st.markdown(f"#### 🎯 Estimated Queue Position: **{user_position:,}**")
    else:
        user_position = st.number_input("Your Queue Position (你的排位人数)", min_value=1000, max_value=100000, value=50595)

with col2:
    trials = st.slider("Number of Simulations", 100, 2000, 300, step=100)
    run_simulation_right = st.button("🚀 Run Simulation 🚀", type="primary")
    if run_simulation_right:
        run_simulation = True

class EB2Predictor:
    def __init__(self, user_position, params):
        self.user_position = user_position
        self.params = params
        self.monthly_quota = (2803 + params.get("spillover", 0)) / 12
        self.monthly_retention = (1 - params.get("withdrawal_rate", 0.135)) ** (1 / 12)

    def simulate_once(self):
        position = self.user_position
        cumulative = 0
        months = 0

        while cumulative < position and months < 240:
            if np.random.rand() < self.params['policy_risk_prob']:
                direction = np.random.choice(['positive', 'negative'], p=[0.3, 0.7])
                if direction == 'positive':
                    multiplier = 1 + np.random.uniform(0.02, self.params['positive_policy_boost'])
                else:
                    multiplier = 1 - np.random.uniform(self.params['negative_policy_penalty'], 0.05)
            else:
                multiplier = 1

            speed_variation = lognorm(s=0.25).rvs()
            actual_speed = min(self.monthly_quota, self.params["base_speed"] * speed_variation * multiplier)

            cumulative += actual_speed
            position = int(position * self.monthly_retention)
            months += 1

        return months

    def simulate(self, n):
        return pd.Series([self.simulate_once() for _ in range(n)])

if "simulation_history" not in st.session_state:
    st.session_state.simulation_history = []

if run_simulation_right:
    with st.spinner("Running simulation..."):
        model = EB2Predictor(user_position=user_position, params=params)
        results = model.simulate(trials)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(results, bins=30, kde=True, ax=ax, color='skyblue')
    ax.axvline(results.median(), color='red', linestyle='--', label=f'Median: {results.median()} months')
    ax.set_title("Projected Wait Time Distribution")
    ax.set_xlabel("Months to Current")
    ax.set_ylabel("Simulation Count")
    ax.legend()
    st.pyplot(fig)

    projected_date = pd.Timestamp("2025-07-01") + pd.DateOffset(months=int(results.median()))

    st.session_state.simulation_history.append({
        "Profile": profile,
        "Position": user_position,
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
    - Assumption Mode: **{profile}**
    """)

    if st.session_state.simulation_history:
        st.markdown("### 📂 Comparison of Saved Simulations")
        hist_df = pd.DataFrame(st.session_state.simulation_history)
        st.dataframe(hist_df)
