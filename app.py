import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# --- 1. 多语言配置中心 ---
LANG_DICT = {
    "简体中文": {
        "nav_label": "功能导航",
        "nav_vol": "TQQQ 波动看板",
        "nav_idx": "指数新低分析",
        "settings": "参数设置",
        "run_btn": "运行分析",
        "report_title": "统计报告",
        "new_low_trigger": "新低触发总次数",
        "absolute_low_confirm": "绝对低点确认（绿线）次数",
        "data_error": "数据不足，无法分析",
        "idx_settings": "指数参数设置",
        "select_idx": "选择指数名称",
        "back_weeks": "设置回溯周数",
        "conf_days": "设置确认天数",
        "start_date": "设置起始日期"
    },
    "English": {
        "nav_label": "Navigation",
        "nav_vol": "TQQQ Dashboard",
        "nav_idx": "Index New Low Analysis",
        "settings": "Settings",
        "run_btn": "Run Analysis",
        "report_title": "Statistics Report",
        "new_low_trigger": "Total New Low Triggers",
        "absolute_low_confirm": "Absolute Low Confirmations",
        "data_error": "Insufficient data for analysis",
        "idx_settings": "Index Settings",
        "select_idx": "Select Index Name",
        "back_weeks": "Lookback Weeks",
        "conf_days": "Confirmation Days",
        "start_date": "Start Date"
    },
    "日本語": {
        "nav_label": "ナビゲーション",
        "nav_vol": "TQQQ ボラティリティ看板",
        "nav_idx": "指数新安値分析",
        "settings": "パラメータ設定",
        "run_btn": "分析実行",
        "report_title": "統計レポート",
        "new_low_trigger": "新安値トリガー合計回数",
        "absolute_low_confirm": "底打ち確認（緑線）回数",
        "data_error": "データ不足、分析できません",
        "idx_settings": "指数パラメータ設定",
        "select_idx": "指数名を選択",
        "back_weeks": "遡及週間の設定",
        "conf_days": "確認日数の設定",
        "start_date": "開始日の設定"
    }
}

# --- 2. 状态管理 (解决页面跳转问题) ---
st.set_page_config(page_title="Market Analytics Hub", layout="wide")

# 初始化 Session State
if "current_nav_index" not in st.session_state:
    st.session_state.current_nav_index = 0

with st.sidebar:
    # 语言选择
    selected_lang = st.selectbox("Language / 语言 / 言語", options=list(LANG_DICT.keys()))
    L = LANG_DICT[selected_lang]
    
    st.divider()
    
    # 获取当前导航选项的列表
    nav_options = [L["nav_vol"], L["nav_idx"]]
    
    # 使用 radio，并通过 index 参数与 session_state 绑定
    app_mode = st.radio(
        L["nav_label"], 
        nav_options, 
        index=st.session_state.current_nav_index,
        key="nav_radio"
    )
    
    # 更新索引，确保下次重新运行时还在这一页
    st.session_state.current_nav_index = nav_options.index(app_mode)

# --- 3. 功能模块：TQQQ 波动看板 (原始逻辑) ---
if app_mode == L["nav_vol"]:
    st.title(f"🚀 {L['nav_vol']}")
    
    with st.sidebar:
        st.header(L["settings"])
        ticker_symbol = st.text_input("Ticker Symbol", value="TQQQ")
        confidence_level = st.slider("Confidence Level (%)", 80, 99, 95)
        sigma_multiplier = st.slider("Sigma Multiplier", 1.0, 4.0, 2.0, 0.1)
        lookback_period = st.selectbox("Lookback Period", ["1y", "2y", "5y", "10y", "max"], index=3)
        run_v = st.button(L["run_btn"], key="run_v")

    if run_v or ticker_symbol:
        tq = yf.Ticker(ticker_symbol)
        vxn_data = yf.Ticker("^VXN").history(period="1d")
        current_vxn = vxn_data['Close'].iloc[-1] if not vxn_data.empty else 0
        hist = tq.history(period=lookback_period)

        if len(hist) < 20:
            st.error(L["data_error"])
        else:
            # 核心计算逻辑保持不变
            high_low = hist['High'] - hist['Low']
            true_range = np.maximum(high_low, np.abs(hist['High'] - hist['Close'].shift()))
            current_atr = true_range.rolling(14).mean().iloc[-1]
            weekly_resample = hist.resample('W-MON').agg({'Open': 'first', 'Close': 'last'}).dropna()
            weekly_returns = (weekly_resample['Close'] - weekly_resample['Open']) / weekly_resample['Open']
            current_price = hist['Close'].iloc[-1]
            std_dev = weekly_returns.std()
            mean_ret = weekly_returns.mean()

            # 绘图
            fig1, ax = plt.subplots(figsize=(12, 6))
            sns.histplot(weekly_returns, kde=True, bins=40, color="#8884d8", stat="density", alpha=0.3, ax=ax)
            lower_q, upper_q = weekly_returns.quantile([(100 - confidence_level) / 100, confidence_level / 100])
            l_sigma, u_sigma = mean_ret - sigma_multiplier * std_dev, mean_ret + sigma_multiplier * std_dev
            
            y_limit = ax.get_ylim()[1]
            plot_lines = [(l_sigma, 'red', f'-{sigma_multiplier}σ'), (lower_q, 'green', 'Q_low'), (upper_q, 'blue', 'Q_high'), (u_sigma, 'red', f'+{sigma_multiplier}σ')]
            for val, col, lbl in plot_lines:
                ax.axvline(val, color=col, linestyle='--', lw=2)
            st.pyplot(fig1)

            # 风险评估颜色逻辑保持不变
            def get_risk_config(vxn):
                if vxn < 20: return "#2e7d32", "Safe (低波动)", 1.5, "低波动环境，可适当追求权利金。"
                if vxn < 25: return "#fbc02d", "Caution (标准)", 2.0, "标准波动，建议维持 2σ 保护。"
                if vxn < 30: return "#fb8c00", "Warning (高波动)", 2.5, "波动加剧，建议拉开距离至 2.5σ。"
                return "#d32f2f", "CRISIS (极端恐慌)", 3.0, "极端恐慌！建议至少 3σ 或观望。"

            bg_color, status_text, suggested_s, advice = get_risk_config(current_vxn)
            st.markdown(f"<div style='background-color:{bg_color};color:white;padding:15px;border-radius:8px;'><h2>VXN Risk: {status_text} ({current_vxn:.2f})</h2><p>{advice}</p></div>", unsafe_allow_html=True)

            # 表格部分
            try:
                iv = tq.option_chain(tq.options[0]).puts.iloc[(tq.option_chain(tq.options[0]).puts['strike'] - current_price).abs().argsort()[:1]]['impliedVolatility'].iloc[0]
            except:
                iv = std_dev * np.sqrt(52)
            
            def calc_prob(target_p, direction='down'):
                t = 5 / 365
                d2 = (np.log(current_price / target_p) + (- 0.5 * iv**2) * t) / (iv * np.sqrt(t))
                return norm.cdf(-d2) if direction == 'down' else (1 - norm.cdf(-d2))

            atr_buf = current_atr * np.sqrt(5) * 1.5
            df_buy = pd.DataFrame([["历史概率支撑", current_price * (1 + lower_q)], ["ATR 动态支撑", current_price - atr_buf], [f"{sigma_multiplier}σ 支撑", current_price * (1 + l_sigma)]], columns=["策略方案", "建议位"])
            df_buy["跌破概率(IV)"] = df_buy["建议位"].apply(lambda x: f"{calc_prob(x, 'down'):.2%}")
            st.write("### Buy/Put Side")
            st.table(df_buy.style.format({"建议位": "${:.2f}"}))

# --- 4. 功能模块：指数新低分析 (原始逻辑) ---
elif app_mode == L["nav_idx"]:
    st.title(f"📉 {L['nav_idx']}")
    
    symbol_map = {
        "纳斯达克100 (NDX)": "^NDX", "标普500 (S&P 500)": "^GSPC",
        "恒生指数 (HSI)": "^HSI", "沪深300 (CSI 300)": "000300.SS",
        "日经225 (Nikkei 225)": "^N225", "德国DAX (DAX)": "^GDAXI",
        "英国富时100 (FTSE 100)": "^FTSE", "韩国综合指数 (KOSPI)": "^KS11"
    }

    with st.sidebar:
        st.header(L["idx_settings"])
        index_display_name = st.selectbox(L["select_idx"], list(symbol_map.keys()))
        index_symbol = symbol_map[index_display_name]
        lookback_weeks = st.slider(L["back_weeks"], 1, 104, 26)
        confirm_days = st.slider(L["conf_days"], 1, 20, 5)
        start_date = st.text_input(L["start_date"], "2019-01-01")
        run_idx = st.button(L["run_btn"], key="run_idx")

    if run_idx:
        window_size = lookback_weeks * 5
        df = yf.download(index_symbol, start=start_date)
        
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                close = df['Close'][index_symbol].copy()
            else:
                close = df['Close'].copy()
            close = close.squeeze().tz_localize(None)

            rolling_min = close.shift(1).rolling(window=window_size).min()
            is_new_low = close < rolling_min
            confirmed_rebound_dates = []
            new_low_dates = close[is_new_low].index

            for low_date in new_low_dates:
                try:
                    current_idx = close.index.get_loc(low_date)
                    target_idx = current_idx + confirm_days
                    if target_idx < len(close):
                        price_at_low = close.iloc[current_idx]
                        price_at_confirm = close.iloc[target_idx]
                        period_prices = close.iloc[current_idx + 1 : target_idx + 1]
                        if (period_prices >= price_at_low).all() and price_at_confirm > price_at_low:
                            no_more_new_lows = not is_new_low.iloc[current_idx + 1 : target_idx + 1].any()
                            if no_more_new_lows:
                                confirmed_rebound_dates.append(close.index[target_idx])
                except: continue

            # 绘图逻辑
            fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
            ax1.plot(close.index, close.values, label='Price', alpha=0.4)
            for date in confirmed_rebound_dates:
                ax1.axvline(x=date, color='#00FF00', linestyle='--', alpha=0.8)
            ax1.set_title(f"{index_display_name} Analysis")
            
            rolling_max = close.rolling(window=window_size, min_periods=1).max()
            drawdown = (close / rolling_max - 1) * 100
            ax2.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
            st.pyplot(fig2)

            st.write(f"### 📊 {L['report_title']}")
            st.write(f"{L['new_low_trigger']}: {is_new_low.sum()}")
            st.write(f"{L['absolute_low_confirm']}: {len(confirmed_rebound_dates)}")
