import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import os
import matplotlib.font_manager as fm
from datetime import datetime
from scipy.stats import lognorm

# 设置 matplotlib 中文字体（使用稳定 CDN）
font_path = '/tmp/NotoSansSC-Regular.otf'
if not os.path.exists(font_path):
    import urllib.request
    urllib.request.urlretrieve(
        'https://raw.githubusercontent.com/googlefonts/noto-cjk/main/Sans/OTF/SimplifiedChinese/NotoSansSC-Regular.otf',
        font_path
    )
fm.fontManager.addfont(font_path)
matplotlib.rc('font', family='Noto Sans SC')

class EB2PredictorImproved:
    def __init__(self, baseline_date='2025-05', target_pd='2022-11', backlog_mode='中性'):
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
            if self.backlog_mode == '乐观':
                base = np.random.randint(300, 500)
            elif self.backlog_mode == '悲观':
                base = np.random.randint(800, 1100)
            else:  # 中性
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
st.set_page_config(page_title="EB2 China PD Simulator", layout="centered")
st.title("\U0001F1E8\U0001F1F3 EB2 中国排期预测模拟器")

st.markdown("""
### 模型说明
- 本模拟器基于蒙特卡洛方法，考虑申请积压量、每月处理能力、政策扰动等多种变量。
- 模拟以“你指定的优先日”为目标，计算当前排期推进到你这一日期所需的月数。
- 支持选择不同积压假设（乐观 / 中性 / 悲观），反映不同 backlog 场景下的预测结果。
- 注意：模拟结果仅用于娱乐参考：），非官方预测。
""")

target_pd = st.date_input("你的 Priority Date", value=datetime(2022, 11, 1))
trials = st.slider("模拟次数", min_value=100, max_value=2000, value=300, step=100)
backlog_mode = st.selectbox("选择积压模型", options=["乐观", "中性", "悲观"], index=1)

if st.button("开始模拟"):
    with st.spinner("模型正在运行中，请稍候……"):
        simulator = EB2PredictorImproved(target_pd=target_pd.strftime('%Y-%m'), backlog_mode=backlog_mode)
        results = simulator.simulate(trials=trials)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(results, bins=30, kde=True, color='steelblue', ax=ax)
    ax.axvline(results.median(), color='red', linestyle='--', label=f'中位数: {results.median()} 月')
    ax.set_title("排期推进至目标 PD 所需月份分布")
    ax.set_xlabel("月份数")
    ax.set_ylabel("模拟次数")
    ax.legend()
    st.pyplot(fig)

    projected_date = pd.to_datetime('2025-05') + pd.DateOffset(months=int(results.median()))
    st.markdown(f"""
    ### \U0001F9BE 模拟摘要
    - 中位等待时间：**{int(results.median())} 个月**
    - 预计排到时间：**{projected_date.strftime('%Y-%m')}**
    - 最快可能时间：**{int(results.min())} 个月**
    - 最慢可能时间：**{int(results.max())} 个月**
    - 当前模型假设：**{backlog_mode} 积压量模型**
    """)
