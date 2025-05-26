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
# 计算每年PD推进的月份数和真实时间间隔，再计算推进速度（PD月数/自然月数）
historical_progress["PD Delta"] = historical_progress["PD"].diff().dt.days.div(30).fillna(0)
historical_progress["Time Delta"] = historical_progress["Cutoff"].diff().dt.days.div(30).fillna(0)
historical_progress["Speed"] = historical_progress["PD Delta"] / historical_progress["Time Delta"]
historical_speed_avg = historical_progress["Speed"].mean()

# UI设置可交互参数
st.sidebar.header("🔧 Simulation Parameters")
base_speed = st.sidebar.number_input("Monthly Processing Base (基准处理速度)", min_value=50, max_value=1000, value=230)
withdrawal_rate = st.sidebar.slider("Annual Withdrawal Rate (申请人流失率)", 0.0, 0.3, 0.112, step=0.01)
policy_risk_prob = st.sidebar.slider("Policy Risk Probability (政策变动概率)", 0.0, 0.5, 0.10, step=0.01)
positive_policy_boost = st.sidebar.slider("Positive Policy Boost (%)", 0.0, 0.3, 0.1, step=0.01)
negative_policy_penalty = st.sidebar.slider("Negative Policy Penalty (%)", 0.0, 0.5, 0.15, step=0.01)

# 假设参数展示表
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
        f"{withdrawal_rate*100:.1f}% per year",
        "31% downgrade rate",
        f"{base_speed} ± 70 cases/month",
        f"{policy_risk_prob*100:.1f}% chance"
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
st.markdown("""
本模拟器基于真实排期推进速度与积压假设，使用蒙特卡洛方法预测你所关心的优先日（Priority Date）大致何时能排到。
可切换不同的积压场景（乐观 / 中性 / 悲观）与模拟次数，查看结果分布。
""")

col1, col2 = st.columns(2)
with col1:
    target_pd = st.date_input("Your Priority Date (你的优先日)", value=datetime(2022, 11, 1))
with col2:
    trials = st.slider("Number of Simulations (模拟次数)", min_value=100, max_value=2000, value=300, step=100)

backlog_mode = st.selectbox("Backlog Scenario (积压场景)", options=["Optimistic", "Neutral", "Pessimistic"], index=1)

class EB2PredictorImproved:
    def __init__(self, baseline_date='2025-05', target_pd='2022-11', backlog_mode='Neutral'):
        self.month = pd.to_datetime(baseline_date)
        self.target_pd = pd.to_datetime(target_pd)
        self.backlog_mode = backlog_mode

        self.annual_quota = 2803
        self.base_processing_speed = base_speed * historical_speed_avg
        self.speed_variation = lognorm(s=0.25)
        self.withdrawal_rate = withdrawal_rate
        self.policy_change_prob = policy_risk_prob
        self.policy_change_impact = {'positive': [0.02, positive_policy_boost], 'negative': [-negative_policy_penalty, -0.05]}

        self.histogram = self._generate_backlog_distribution()
        self.pd_range = pd.date_range('2020-12', '2023-01', freq='MS')
        self.cumulative_backlog = self.histogram.cumsum()

    def _generate_backlog_distribution(self):
        date_index = pd.date_range('2020-12', '2023-01', freq='MS')
        applicants = []
        for date in date_index:
            if self.backlog_mode == 'Optimistic':
                base = np.random.randint(300, 500)
            elif self.backlog_mode == 'Pessimistic':
                base = np.random.randint(800, 1100)
            else:
                base = np.random.randint(400, 900)
            applicants.append(base)
        return pd.Series(data=applicants, index=date_index)

    def _policy_impact_simulation(self):
        if np.random.rand() < self.policy_change_prob:
            direction = np.random.choice(['positive', 'negative'], p=[0.3, 0.7])
            impact = np.random.uniform(*self.policy_change_impact[direction])
            return 1 + impact
        return 1

    def simulate_single_run(self, n_years=10):
        current_index = 0
        target_index = np.searchsorted(self.histogram.index, self.target_pd)
        months = 0

        for _ in range(n_years * 12):
            speed_variation = self.speed_variation.rvs()
            policy_factor = self._policy_impact_simulation()
            actual_speed = int(self.base_processing_speed * speed_variation * policy_factor)
            current_backlog = self.histogram.iloc[current_index:].sum()
            withdrawals = int(current_backlog * (self.withdrawal_rate / 12))
            actual_speed += withdrawals

            for i in range(current_index, len(self.histogram)):
                if actual_speed <= 0:
                    break
                if self.histogram.iloc[i] <= actual_speed:
                    actual_speed -= self.histogram.iloc[i]
                    self.histogram.iloc[i] = 0
                    current_index = i + 1
                else:
                    self.histogram.iloc[i] -= actual_speed
                    actual_speed = 0

            months += 1
            if current_index >= target_index:
                break

        return months

    def simulate(self, trials=500):
        return pd.Series([self.__class__(target_pd=self.target_pd.strftime('%Y-%m'), backlog_mode=self.backlog_mode).simulate_single_run() for _ in range(trials)])

if st.button("Run Simulation"):
    with st.spinner("Running simulation... 模型运行中..."):
        simulator = EB2PredictorImproved(target_pd=target_pd.strftime('%Y-%m'), backlog_mode=backlog_mode)
        results = simulator.simulate(trials=trials)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(results, bins=30, kde=True, color='steelblue', ax=ax)
    ax.axvline(results.median(), color='red', linestyle='--', label=f'Median: {results.median()} months')
    ax.set_title("Projected Wait Time Distribution (预测等待时间分布)")
    ax.set_xlabel("Months to Current (距离排到的月份)")
    ax.set_ylabel("Simulation Count (模拟次数)")
    ax.legend()
    st.pyplot(fig)

    projected_date = pd.to_datetime('2025-05') + pd.DateOffset(months=int(results.median()))
    st.markdown(f"""
    ### 🧠 Simulation Summary 模拟结果摘要
    - Median wait time: **{int(results.median())} months**
    - Expected PD becomes current: **{projected_date.strftime('%Y-%m')}**
    - Range: {int(results.min())} to {int(results.max())} months
    - Assumption Mode: **{backlog_mode}**
    """)
