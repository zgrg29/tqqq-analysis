import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# --- 1. Streamlit 侧边栏设置 (替代原有的 #@param) ---
st.set_page_config(page_title="TQQQ 波动看板", layout="wide")
st.title("🚀 TQQQ 每周波动全维度看板")

with st.sidebar:
    st.header("参数设置")
    ticker_symbol = st.text_input("股票代码", value="TQQQ")
    confidence_level = st.slider("置信水平 (%)", min_value=80, max_value=99, value=95)
    sigma_multiplier = st.slider("标准差倍数 (σ)", min_value=1.0, max_value=4.0, value=2.0, step=0.1)
    lookback_period = st.selectbox("回顾周期", options=["1y", "2y", "5y", "10y", "max"], index=3)
    run_button = st.button("开始分析")

# --- 2. 核心分析逻辑 ---
def run_analysis():
    # 数据获取
    tq = yf.Ticker(ticker_symbol)
    vxn_data = yf.Ticker("^VXN").history(period="1d")
    current_vxn = vxn_data['Close'].iloc[-1] if not vxn_data.empty else 0

    hist = tq.history(period=lookback_period)
    if len(hist) < 20:
        st.error("数据不足，无法分析")
        return

    # 核心计算
    high_low = hist['High'] - hist['Low']
    true_range = np.maximum(high_low, np.abs(hist['High'] - hist['Close'].shift()))
    current_atr = true_range.rolling(14).mean().iloc[-1]

    weekly_resample = hist.resample('W-MON').agg({'Open': 'first', 'Close': 'last'}).dropna()
    weekly_returns = (weekly_resample['Close'] - weekly_resample['Open']) / weekly_resample['Open']

    current_price = hist['Close'].iloc[-1]
    std_dev = weekly_returns.std()
    mean_ret = weekly_returns.mean()

    # 3. 风险看板 (Streamlit 渲染 HTML)
    def get_risk_config(vxn):
        if vxn < 20: return "#2e7d32", "Safe (低波动)", 1.5, "低波动环境，可适当追求权利金。"
        if vxn < 25: return "#fbc02d", "Caution (标准)", 2.0, "标准波动，建议维持 2σ 保护。"
        if vxn < 30: return "#fb8c00", "Warning (高波动)", 2.5, "波动加剧，建议拉开距离至 2.5σ。"
        return "#d32f2f", "CRISIS (极端恐慌)", 3.0, "极端恐慌！建议至少 3σ 或观望。"

    bg_color, status_text, suggested_s, advice = get_risk_config(current_vxn)
    
    st.markdown(f"""
    <div style="background-color: {bg_color}; color: white; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
        <h2 style="margin: 0; color: white;">VXN 风险等级: {status_text} (当前指数: {current_vxn:.2f})</h2>
        <p style="margin: 10px 0 0 0; font-size: 16px;">
            <b>策略建议：</b> 建议使用 <b>{suggested_s}σ</b> | <b>备注：</b> {advice}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 4. 绘图 (Streamlit 渲染 Matplotlib)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(weekly_returns, kde=True, bins=40, color="#8884d8", stat="density", alpha=0.3, ax=ax)
    
    lower_q, upper_q = weekly_returns.quantile([(100 - confidence_level) / 100, confidence_level / 100])
    l_sigma, u_sigma = mean_ret - sigma_multiplier * std_dev, mean_ret + sigma_multiplier * std_dev

    plot_lines = [
        (l_sigma, 'red', f'-{sigma_multiplier}σ: {l_sigma:.2%}'),
        (lower_q, 'green', f'Q_{100-confidence_level}%: {lower_q:.2%}'),
        (upper_q, 'blue', f'Q_{confidence_level}%: {upper_q:.2%}'),
        (u_sigma, 'red', f'+{sigma_multiplier}σ: {u_sigma:.2%}')
    ]

    y_limit = ax.get_ylim()[1]
    for i, (val, col, lbl) in enumerate(plot_lines):
        ax.axvline(val, color=col, linestyle='--', lw=2, alpha=0.7)
        ax.text(val, y_limit * (0.85 if i%2==0 else 0.7), lbl, color=col, fontweight='bold',
                 ha='center', bbox=dict(facecolor='white', alpha=0.9, edgecolor=col, boxstyle='round'))

    ax.set_title(f"{ticker_symbol} Weekly Returns Distribution", fontsize=14)
    st.pyplot(fig)

    # 5. 表格计算
    try:
        iv = tq.option_chain(tq.options[0]).puts.iloc[(tq.option_chain(tq.options[0]).puts['strike'] - current_price).abs().argsort()[:1]]['impliedVolatility'].iloc[0]
    except:
        iv = std_dev * np.sqrt(52)

    def calc_prob(target_p, direction='down'):
        t = 5 / 365
        d2 = (np.log(current_price / target_p) + (- 0.5 * iv**2) * t) / (iv * np.sqrt(t))
        return norm.cdf(-d2) if direction == 'down' else (1 - norm.cdf(-d2))

    atr_buf = current_atr * np.sqrt(5) * 1.5
    
    # 构建 DataFrame 并显示
    st.subheader(f"💎 {ticker_symbol} 统计面板 | 当前价: ${current_price:.2f}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 支撑位分析 (Buy/Put)")
        df_buy = pd.DataFrame([
            ["历史概率支撑", current_price * (1 + lower_q), f"{100-confidence_level}% 分位数"],
            ["ATR 动态支撑", current_price - atr_buf, "日线 ATR 映射"],
            [f"{sigma_multiplier}σ 支撑", current_price * (1 + l_sigma), f"{sigma_multiplier}倍标准差"]
        ], columns=["策略方案", "下周建议位", "逻辑参考"])
        df_buy["跌破概率(IV)"] = df_buy["下周建议位"].apply(lambda x: f"{calc_prob(x, 'down'):.2%}")
        st.table(df_buy.style.format({"下周建议位": "${:.2f}"}))

    with col2:
        st.write("### 阻力位分析 (Sell/Call)")
        df_sell = pd.DataFrame([
            ["历史概率阻力", current_price * (1 + upper_q), f"{confidence_level}% 分位数"],
            ["ATR 动态阻力", current_price + atr_buf, "日线 ATR 映射"],
            [f"{sigma_multiplier}σ 阻力", current_price * (1 + u_sigma), f"{sigma_multiplier}倍标准差"]
        ], columns=["策略方案", "下周建议位", "逻辑参考"])
        df_sell["突破概率(IV)"] = df_sell["下周建议位"].apply(lambda x: f"{calc_prob(x, 'up'):.2%}")
        st.table(df_sell.style.format({"下周建议位": "${:.2f}"}))

# 运行应用
if run_button or ticker_symbol:
    run_analysis()
