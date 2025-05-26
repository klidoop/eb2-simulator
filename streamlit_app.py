import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from datetime import datetime
from scipy.stats import lognorm

class EB2PredictorImproved:
    def __init__(self, baseline_date='2025-05', target_pd='2022-11', backlog_mode='Neutral'):
        self.month = pd.to_datetime(baseline_date)
        self.target_pd = pd.to_datetime(target_pd)
        self.backlog_mode = backlog_mode

        self.annual_quota = 2803
        self.base_processing_speed = 230
        self.speed_variation = lognorm(s=0.25)
        self.withdrawal_rate = 0.112
        self.policy_change_prob = 0.10
        self.policy_change_impact = {'positive': [0.02, 0.1], 'negative': [-0.05, -0.15]}

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
            else:  # Neutral
                if date.year == 2021:
                    base = np.random.randint(700, 1000)
                elif date.year == 2022:
                    base = np.random.randint(400, 600)
                else:
                    base = np.random.randint(300, 400)
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

# Streamlit UI
st.set_page_config(page_title="EB2/EB3 Priority Date Forecast", layout="centered")
st.title("🇨🇳 EB2 vs EB3 Priority Date Forecast")

st.markdown("""
### Model Overview
This simulator uses Monte Carlo methods to estimate how long it will take for your Priority Date (PD) to become current under different assumptions:
- You can simulate EB2 or EB3 backlog independently.
- Backlog scenarios (Optimistic / Neutral / Pessimistic) represent filing intensity and visa demand.
- All estimates are probabilistic, not official.
""")

# User inputs
col1, col2 = st.columns(2)
with col1:
    target_pd = st.date_input("Your Priority Date", value=datetime(2022, 11, 1), key="eb2_date")
with col2:
    trials = st.slider("Number of Simulations", min_value=100, max_value=2000, value=300, step=100)

backlog_mode_eb2 = st.selectbox("EB2 Backlog Assumption", options=["Optimistic", "Neutral", "Pessimistic"], index=1, key="eb2_mode")
backlog_mode_eb3 = st.selectbox("EB3 Backlog Assumption", options=["Optimistic", "Neutral", "Pessimistic"], index=1, key="eb3_mode")

if st.button("Compare EB2 vs EB3"):
    with st.spinner("Running simulations..."):
        eb2_simulator = EB2PredictorImproved(target_pd=target_pd.strftime('%Y-%m'), backlog_mode=backlog_mode_eb2)
        eb3_simulator = EB2PredictorImproved(target_pd=target_pd.strftime('%Y-%m'), backlog_mode=backlog_mode_eb3)
        eb2_results = eb2_simulator.simulate(trials=trials)
        eb3_results = eb3_simulator.simulate(trials=trials)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(eb2_results, bins=30, kde=True, label='EB2', color='skyblue', ax=ax)
    sns.histplot(eb3_results, bins=30, kde=True, label='EB3', color='salmon', ax=ax)
    ax.axvline(eb2_results.median(), color='blue', linestyle='--', label=f'EB2 Median: {eb2_results.median()} mo')
    ax.axvline(eb3_results.median(), color='red', linestyle='--', label=f'EB3 Median: {eb3_results.median()} mo')
    ax.set_title("Comparison of EB2 vs EB3 Wait Times")
    ax.set_xlabel("Months to Current")
    ax.set_ylabel("Simulation Count")
    ax.legend()
    st.pyplot(fig)

    projected_eb2 = pd.to_datetime('2025-05') + pd.DateOffset(months=int(eb2_results.median()))
    projected_eb3 = pd.to_datetime('2025-05') + pd.DateOffset(months=int(eb3_results.median()))

    diff = int(eb2_results.median() - eb3_results.median())
    recommendation = "**🟢 Downgrade is likely beneficial**" if diff > 4 else "**⚪️ Downgrade advantage is marginal or uncertain**"

    st.markdown(f"""
    ### 🧠 Simulation Summary
    **EB2**
    - Median wait: **{int(eb2_results.median())} months** → ~{projected_eb2.strftime('%Y-%m')}
    - Range: {int(eb2_results.min())} to {int(eb2_results.max())} months
    - Scenario: **{backlog_mode_eb2}**

    **EB3**
    - Median wait: **{int(eb3_results.median())} months** → ~{projected_eb3.strftime('%Y-%m')}
    - Range: {int(eb3_results.min())} to {int(eb3_results.max())} months
    - Scenario: **{backlog_mode_eb3}**

    **📌 Recommendation:** {recommendation}
    (EB3 is faster by {abs(diff)} months)
    """)
