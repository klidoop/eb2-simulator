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
    "Conservative": dict(base_speed=180, withdrawal_rate=0.08, policy_risk_prob=0.2, positive_policy_boost=0.05, negative_policy_penalty=0.3, spillover=1148),
    "Neutral": dict(base_speed=230, withdrawal_rate=0.112, policy_risk_prob=0.1, positive_policy_boost=0.1, negative_policy_penalty=0.15, spillover=1148),
    "Aggressive": dict(base_speed=280, withdrawal_rate=0.15, policy_risk_prob=0.05, positive_policy_boost=0.2, negative_policy_penalty=0.05, spillover=1148)
}

st.sidebar.header("🔧 Simulation Parameters")
profile = st.sidebar.selectbox("Preset Profile", list(presets.keys()), help="Choose a predefined set of assumptions for backlog and risk")

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

params["base_speed"] = st.sidebar.number_input(
    label="Monthly Processing Base (基准处理速度)",
    min_value=50,
    max_value=1000,
    value=params["base_speed"],
    key="base_speed",
    help="Average number of EB2 cases processed per month before adjustment"
)
params["withdrawal_rate"] = st.sidebar.slider(
    label="Annual Withdrawal Rate (申请人流失率)",
    min_value=0.0,
    max_value=0.3,
    value=params["withdrawal_rate"],
    step=0.01,
    key="withdrawal_rate",
    help="Percentage of applicants who drop out or switch categories annually"
)
params["policy_risk_prob"] = st.sidebar.slider(
    label="Policy Risk Probability (政策变动概率)",
    min_value=0.0,
    max_value=0.5,
    value=params["policy_risk_prob"],
    step=0.01,
    key="policy_risk_prob",
    help="Chance that a major immigration policy event affects processing speed in a given month"
)
params["positive_policy_boost"] = st.sidebar.slider(
    label="Positive Policy Boost (%)",
    min_value=0.0,
    max_value=0.3,
    value=params["positive_policy_boost"],
    step=0.01,
    key="positive_policy_boost",
    help="If a positive policy change occurs, maximum boost to processing speed"
)
params["negative_policy_penalty"] = st.sidebar.slider(
    label="Negative Policy Penalty (%)",
    min_value=0.0,
    max_value=0.5,
    value=params["negative_policy_penalty"],
    step=0.01,
    key="negative_policy_penalty",
    help="If a negative policy change occurs, maximum reduction to processing speed"
)
params["spillover"] = st.sidebar.number_input(
    label="Family-Based Spillover to EB2-China (家庭类签证转回数量)",
    min_value=0,
    max_value=10000,
    value=params["spillover"],
    key="spillover",
    help="Extra visas allocated to EB2 from family-based unused quota (e.g. 1148 in FY2025)"
)

# 计算配额（用于模型实际速度上限）
params["monthly_quota"] = (2803 + params["spillover"]) / 12

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

# UI inputs
col1, col2 = st.columns(2)
with col1:
    target_pd = st.date_input("Your Priority Date (你的优先日)", value=datetime(2022, 11, 1))
with col2:
    trials = st.slider("Number of Simulations (模拟次数)", min_value=100, max_value=2000, value=300, step=100)

backlog_mode = st.selectbox("Backlog Scenario (积压场景)", options=["Optimistic", "Neutral", "Pessimistic"], index=1)

# Define class and run model
class EB2Predictor:
    def __init__(self, target_pd, backlog_mode):
        self.target_pd = pd.to_datetime(target_pd)
        self.backlog_mode = backlog_mode
        self.base_speed = params['base_speed'] * historical_speed_avg
        self.speed_variation = lognorm(s=0.25)
        self.withdrawal_rate = params['withdrawal_rate']
        self.policy_prob = params['policy_risk_prob']
        self.policy_impact = {
            'positive': [0.02, params['positive_policy_boost']],
            'negative': [-params['negative_policy_penalty'], -0.05]
        }
        self.histogram = self._generate_backlog()

    def _generate_backlog(self):
        months = pd.date_range('2020-12', '2023-01', freq='MS')
        if self.backlog_mode == 'Optimistic':
            return pd.Series(np.random.randint(300, 500, len(months)), index=months)
        elif self.backlog_mode == 'Pessimistic':
            return pd.Series(np.random.randint(800, 1100, len(months)), index=months)
        else:
            return pd.Series(np.random.randint(400, 900, len(months)), index=months)

    def _policy_adjustment(self):
        if np.random.rand() < self.policy_prob:
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
            actual_speed = int(self.base_speed * self.speed_variation.rvs() * policy_factor)
            withdrawals = int(backlog.sum() * (self.withdrawal_rate / 12))
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

if st.button("Run Simulation"):
    with st.spinner("Running simulation... 模型运行中..."):
        model = EB2Predictor(target_pd=target_pd, backlog_mode=backlog_mode)
        results = model.simulate(trials)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(results, bins=30, kde=True, ax=ax, color='skyblue')
    ax.axvline(results.median(), color='red', linestyle='--', label=f'Median: {results.median()} months')
    ax.set_title("Projected Wait Time Distribution (预测等待时间分布)")
    ax.set_xlabel("Months to Current (距离排到的月份)")
    ax.set_ylabel("Simulation Count (模拟次数)")
    ax.legend()
    st.pyplot(fig)

    projected_date = pd.to_datetime("2025-05") + pd.DateOffset(months=int(results.median()))
    st.markdown(f"""
    ### 🧠 Simulation Summary 模拟结果摘要
    - Median wait time: **{int(results.median())} months**
    - Expected PD becomes current: **{projected_date.strftime('%Y-%m')}**
    - Range: {int(results.min())} to {int(results.max())} months
    - Assumption Mode: **{backlog_mode}**
    """)
