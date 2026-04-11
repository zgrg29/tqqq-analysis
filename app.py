import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# --- 1. 多语言字典配置 (新增) ---
LANG_DICT = {
    "简体中文": {
        "title": "🚀 TQQQ 每周波动全维度看板",
        "settings": "参数设置",
        "ticker": "股票代码",
        "conf": "置信水平 (%)",
        "sigma": "标准差倍数 (σ)",
        "period": "回顾周期",
        "run": "开始分析",
        "risk_lvl": "VXN 风险等级",
        "strategy": "策略建议",
        "note": "备注",
        "hist_dist": "周收益率分布图",
        "stats_panel": "统计面板",
        "current_price": "当前价",
        "support": "支撑位分析 (Buy/Put)",
        "resistance": "阻力位分析 (Sell/Call)",
        "scheme": "策略方案",
        "target": "下周建议位",
        "logic": "逻辑参考",
        "prob_down": "跌破概率(IV)",
        "prob_up": "突破概率(IV)",
        "data_short": "数据不足，无法分析"
    },
    "English": {
        "title": "🚀 TQQQ Weekly Volatility Dashboard",
        "settings": "Settings",
        "ticker": "Ticker Symbol",
        "conf": "Confidence Level (%)",
        "sigma": "Sigma Multiplier (σ)",
        "period": "Lookback Period",
        "run": "Run Analysis",
        "risk_lvl": "VXN Risk Level",
        "strategy": "Strategy",
        "note": "Note",
        "hist_dist": "Weekly Returns Distribution",
        "stats_panel": "Statistics Panel",
        "current_price": "Current Price",
        "support": "Support Analysis (Buy/Put)",
        "resistance": "Resistance Analysis (Sell/Call)",
        "scheme": "Scheme",
        "target": "Target Price",
        "logic": "Logic",
        "prob_down": "Prob. of Drop (IV)",
        "prob_up": "Prob. of Breakout (IV)",
        "data_short": "Insufficient data for analysis"
    },
    "日本語": {
        "title": "🚀 TQQQ 週間ボラティリティ看板",
        "settings": "設定",
        "ticker": "ティッカー",
        "conf": "信頼水準 (%)",
        "sigma": "標準偏差倍率 (σ)",
        "period": "遡及期間",
        "run": "分析実行",
        "risk_lvl": "VXN リスクレベル",
        "strategy": "戦略提案",
        "note": "備考",
        "hist_dist": "週間収益率分布",
        "stats_panel": "統計パネル",
        "current_price": "現在価格",
        "support": "サポート線 (Buy/Put)",
        "resistance": "レジスタンス線 (Sell/Call)",
        "scheme": "戦略",
        "target": "目標価格",
        "logic": "根拠",
        "prob_down": "下落確率(IV)",
        "prob_up": "上昇確率(IV)",
        "data_short": "データ不足で分析できません"
    }
}

# --- 2. 界面初始化 ---
st.set_page_config(page_title="TQQQ Dashboard", layout="wide")

with st.sidebar:
    # 语言选择器 (新增)
    selected_lang = st.selectbox("Language / 语言 / 言語", options=list(LANG_DICT.keys()))
    L = LANG_DICT[selected_lang] # 当前语言包
    
    st.divider() # 分割线
    st.header(L["settings"])
    ticker_symbol = st.text_input(L["ticker"], value="TQQQ")
    confidence_level = st.slider(L["conf"], min_value=80, max_value=99, value=95)
    sigma_multiplier = st.slider(L["sigma"], min_value=1.0, max_value=4.0, value=2.0, step=0.1)
    lookback_period = st.selectbox(L["period"], options=["1y", "2y", "5y", "10y", "max"], index=3)
    run_button = st.button(L["run"])

st.title(L["title"])

# --- 3. 核心分析逻辑 (仅修改文本显示) ---
def run_analysis():
    tq = yf.Ticker(ticker_symbol)
    vxn_data = yf.Ticker("^VXN").history(period="1d")
    current_vxn = vxn_data['Close'].iloc[-1] if not vxn_data.empty else 0

    hist = tq.history(period=lookback_period)
    if len(hist) < 20:
        st.error(L["data_short"])
        return

    high_low = hist['High'] - hist['Low']
    true_range = np.maximum(high_low, np.abs(hist['High'] - hist['Close'].shift()))
    current_atr = true_range.rolling(14).mean().iloc[-1]

    weekly_resample = hist.resample('W-MON').agg({'Open': 'first', 'Close': 'last'}).dropna()
    weekly_returns = (weekly_resample['Close'] - weekly_resample['Open']) / weekly_resample['Open']

    current_price = hist['Close'].iloc[-1]
    std_dev = weekly_returns.std()
    mean_ret = weekly_returns.mean()

    # 风险看板逻辑 (保持原有逻辑，仅翻译文本)
    def get_risk_config(vxn):
        if vxn < 20: return "#2e7d32", "Safe", 1.5, "Low volatility environment."
        if vxn < 25: return "#fbc02d", "Caution", 2.0, "Standard volatility."
        if vxn < 30: return "#fb8c00", "Warning", 2.5, "Increased volatility."
        return "#d32f2f", "CRISIS", 3.0, "Extreme panic!"

    bg_color, status_text, suggested_s, advice = get_risk_config(current_vxn)
    
    st.markdown(f"""
    <div style="background-color: {bg_color}; color: white; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
        <h2 style="margin: 0; color: white;">{L['risk_lvl']}: {status_text} ({current_vxn:.2f})</h2>
        <p style="margin: 10px 0 0 0; font-size: 16px;">
            <b>{L['strategy']}:</b> {suggested_s}σ | <b>{L['note']}:</b> {advice}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 绘图
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(weekly_returns, kde=True, bins=40, color="#8884d8", stat="density", alpha=0.3, ax=ax)
    
    lower_q, upper_q = weekly_returns.quantile([(100 - confidence_level) / 100, confidence_level / 100])
    l_sigma, u_sigma = mean_ret - sigma_multiplier * std_dev, mean_ret + sigma_multiplier * std_dev

    plot_lines = [(l_sigma, 'red', f'-{sigma_multiplier}σ'), (lower_q, 'green', f'Q_low'), (upper_q, 'blue', f'Q_high'), (u_sigma, 'red', f'+{sigma_multiplier}σ')]
    y_limit = ax.get_ylim()[1]
    for val, col, lbl in plot_lines:
        ax.axvline(val, color=col, linestyle='--', lw=2, alpha=0.7)
    
    ax.set_title(f"{ticker_symbol} {L['hist_dist']}", fontsize=14)
    st.pyplot(fig)

    # 表格
    try:
        iv = tq.option_chain(tq.options[0]).puts.iloc[(tq.option_chain(tq.options[0]).puts['strike'] - current_price).abs().argsort()[:1]]['impliedVolatility'].iloc[0]
    except:
        iv = std_dev * np.sqrt(52)

    def calc_prob(target_p, direction='down'):
        t = 5 / 365
        d2 = (np.log(current_price / target_p) + (- 0.5 * iv**2) * t) / (iv * np.sqrt(t))
        return norm.cdf(-d2) if direction == 'down' else (1 - norm.cdf(-d2))

    atr_buf = current_atr * np.sqrt(5) * 1.5
    st.subheader(f"💎 {ticker_symbol} {L['stats_panel']} | {L['current_price']}: ${current_price:.2f}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"### {L['support']}")
        df_buy = pd.DataFrame([
            ["Quantile", current_price * (1 + lower_q), "History"],
            ["ATR", current_price - atr_buf, "Dynamic"],
            [f"{sigma_multiplier}σ", current_price * (1 + l_sigma), "Standard Dev"]
        ], columns=[L["scheme"], L["target"], L["logic"]])
        df_buy[L["prob_down"]] = df_buy[L["target"]].apply(lambda x: f"{calc_prob(x, 'down'):.2%}")
        st.table(df_buy.style.format({L["target"]: "${:.2f}"}))

    with col2:
        st.write(f"### {L['resistance']}")
        df_sell = pd.DataFrame([
            ["Quantile", current_price * (1 + upper_q), "History"],
            ["ATR", current_price + atr_buf, "Dynamic"],
            [f"{sigma_multiplier}σ", current_price * (1 + u_sigma), "Standard Dev"]
        ], columns=[L["scheme"], L["target"], L["logic"]])
        df_sell[L["prob_up"]] = df_sell[L["target"]].apply(lambda x: f"{calc_prob(x, 'up'):.2%}")
        st.table(df_sell.style.format({L["target"]: "${:.2f}"}))

if run_button or ticker_symbol:
    run_analysis()
