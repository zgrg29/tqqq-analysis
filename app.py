import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from datetime import datetime

# --- 1. 全维度多语言配置中心 ---
LANG_DICT = {
    "简体中文": {
        "nav_label": "功能导航",
        "nav_vol": "期权波动看板",
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
        "start_date": "设置起始日期",
        "strategy": "策略方案",
        "suggested_price": "下周建议位",
        "logic_ref": "逻辑参考",
        "prob_drop": "跌破概率",
        "prob_break": "突破概率",
        "risk_level_title": "实时监控状态",
        "current_price": "当前价",
        "vol_weekly": "历史周σ",
        "hist_support": "历史概率支撑",
        "atr_support": "ATR 动态支撑",
        "sigma_support": "σ 支撑",
        "hist_resist": "历史概率阻力",
        "atr_resist": "ATR 动态阻力",
        "sigma_resist": "σ 阻力",
        "quantile_desc": "分位数",
        "atr_desc": "日线 ATR 映射",
        "sigma_desc": "倍标准差"
    },
    "English": {
        "nav_label": "Navigation",
        "nav_vol": "Option Dashboard",
        "nav_idx": "Index New Low Analysis",
        "settings": "Settings",
        "run_btn": "Run Analysis",
        "report_title": "Statistics Report",
        "new_low_trigger": "Total New Low Triggers",
        "absolute_low_confirm": "Absolute Low Confirmations",
        "data_error": "Insufficient data",
        "idx_settings": "Index Settings",
        "select_idx": "Select Index Name",
        "back_weeks": "Lookback Weeks",
        "conf_days": "Confirmation Days",
        "start_date": "Start Date",
        "strategy": "Strategy",
        "suggested_price": "Target Price",
        "logic_ref": "Logic",
        "prob_drop": "Prob. Drop",
        "prob_break": "Prob. Break",
        "risk_level_title": "Live Status",
        "current_price": "Price",
        "vol_weekly": "Weekly σ",
        "hist_support": "Hist. Support",
        "atr_support": "ATR Support",
        "sigma_support": "σ Support",
        "hist_resist": "Hist. Resistance",
        "atr_resist": "ATR Resistance",
        "sigma_resist": "σ Resistance",
        "quantile_desc": "Quantile",
        "atr_desc": "ATR Mapping",
        "sigma_desc": "Sigma Multiplier"
    }
}

# --- 2. 状态管理 ---
st.set_page_config(page_title="Market Analysis Hub", layout="wide")
if "current_nav_index" not in st.session_state:
    st.session_state.current_nav_index = 0

with st.sidebar:
    selected_lang = st.selectbox("Language / 语言", options=list(LANG_DICT.keys()))
    L = LANG_DICT[selected_lang]
    st.divider()
    nav_options = [L["nav_vol"], L["nav_idx"]]
    app_mode = st.radio(L["nav_label"], nav_options, index=st.session_state.current_nav_index)
    st.session_state.current_nav_index = nav_options.index(app_mode)

# --- 3. 核心功能 A: 股票/ETF 波动看板 ---
if app_mode == L["nav_vol"]:
    st.title(f"🚀 {L['nav_vol']}")
    with st.sidebar:
        st.header(L["settings"])
        ticker_options = ["SOXL", "TQQQ", "Custom"]
        selected_option = st.selectbox("Select Ticker", options=ticker_options, index=0)
        
        if selected_option == "Custom":
            ticker_symbol = st.text_input("Enter Custom Symbol", value="NVDA").upper()
        else:
            ticker_symbol = selected_option

        confidence_level = st.slider("Confidence Level (%)", 80, 99, 95)
        sigma_multiplier = st.slider("Manual Sigma Multiplier", 1.0, 4.0, 2.0, 0.1)
        
        # --- 计算周期逻辑更新 ---
        weekday_now = datetime.now().weekday()  # 0是周一，4是周五
        default_days = max(1, 4 - weekday_now + 1) if weekday_now <= 4 else 7
        calc_days = st.slider("Calculation Days (T)", 1, 10, default_days, help="到期天数：周一=5, 周二=4... 也可以根据资金占用设为7")

        lookback_period = st.selectbox("Lookback Period", ["1y", "2y", "5y", "max"], index=1)
        run_v = st.button(L["run_btn"], key="run_v")

    if run_v or ticker_symbol:
        tq = yf.Ticker(ticker_symbol)
        hist = tq.history(period="2y")

        if len(hist) < 30:
            st.error(L["data_error"])
        else:
            current_price = hist['Close'].iloc[-1]
            
            # RSI 计算
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]

            # 波动率核心逻辑
            weekly_resample = hist.resample('W-MON').agg({'Open': 'first', 'Close': 'last'}).dropna()
            weekly_returns = (weekly_resample['Close'] - weekly_resample['Open']) / weekly_resample['Open']
            std_dev = weekly_returns.std()
            hv_annual = std_dev * np.sqrt(52)
            
            iv_realtime = 0
            try:
                options = tq.options
                if options:
                    opt_chain = tq.option_chain(options[0])
                    puts = opt_chain.puts
                    iv_realtime = puts.iloc[(puts['strike'] - current_price).abs().argsort()[:1]]['impliedVolatility'].iloc[0]
            except:
                iv_realtime = 0

            # 动态选择 IV/HV 较大值
            final_vol = max(iv_realtime, hv_annual) if iv_realtime > 0.10 else hv_annual
            vol_source = f"混合 (IV:{iv_realtime:.1%} / HV:{hv_annual:.1%})" if iv_realtime > 0.10 else "纯历史"

            # 风险状态
            def get_risk_config(rsi_val):
                if rsi_val > 70: return "#d32f2f", "超买 (高风险)", "⚠️ 建议降低行权价"
                elif rsi_val < 30: return "#1976d2", "超卖 (机会)", "✅ 适合卖出收高IV权利金"
                else: return "#2e7d32", "正常区间", "ℹ️ 按模型建议操作"

            bg_color, status_text, advice = get_risk_config(current_rsi)

            # 可视化
            st.subheader(f"📊 波动分析 (计算周期: {calc_days}天)")
            v_col1, v_col2 = st.columns(2)
            with v_col1:
                fig_rsi, ax_rsi = plt.subplots(figsize=(10, 4))
                ax_rsi.plot(rsi.tail(60).index, rsi.tail(60).values, color='#8884d8', lw=2)
                ax_rsi.axhline(70, color='r', linestyle='--')
                ax_rsi.axhline(30, color='g', linestyle='--')
                ax_rsi.set_title(f"RSI: {current_rsi:.2f}")
                st.pyplot(fig_rsi)
            with v_col2:
                fig_vol, ax_vol = plt.subplots(figsize=(10, 4))
                ax_vol.bar(['Real-time IV', 'Historical HV'], [iv_realtime, hv_annual], color=['#bb86fc', '#03dac6'])
                ax_vol.set_title(f"Vol Comparison (Final: {final_vol:.1%})")
                st.pyplot(fig_vol)

            st.markdown(f"""
                <div style='background-color:{bg_color};color:white;padding:15px;border-radius:8px;margin-bottom:20px;'>
                    <h3>状态: {status_text} | 计算天数: {calc_days}d</h3>
                    <p><b>建议逻辑:</b> {advice}</p>
                </div>
                """, unsafe_allow_html=True)

            # 概率计算 (基于动态 calc_days)
            def calc_prob(target_p, direction='down'):
                t = calc_days / 365 
                if final_vol <= 0: return 0.5
                d2 = (np.log(current_price / target_p) + (- 0.5 * final_vol**2) * t) / (final_vol * np.sqrt(t))
                return norm.cdf(-d2) if direction == 'down' else (1 - norm.cdf(-d2))

            # 价位建议
            l_sigma = weekly_returns.mean() - sigma_multiplier * (final_vol / np.sqrt(52))
            
            df_buy = pd.DataFrame([
                [f"{sigma_multiplier}σ {L['sigma_support']}", current_price * (1 + l_sigma), f"周期 {calc_days}天 | {vol_source}"]
            ], columns=[L["strategy"], L["suggested_price"], L["logic_ref"]])
            
            prob_col = f"{L['prob_drop']}({calc_days}d)"
            df_buy[prob_col] = df_buy[L["suggested_price"]].apply(lambda x: f"{calc_prob(x, 'down'):.2%}")
            st.table(df_buy.style.format({L["suggested_price"]: "${:.2f}"}))

# --- 4. 核心功能 B: 指数新低分析 (不相关代码保持不变) ---
elif app_mode == L["nav_idx"]:
    st.title(f"📉 {L['nav_idx']}")
    symbol_map = {"纳斯达克100 (NDX)": "^NDX", "标普500 (S&P 500)": "^GSPC", "恒生指数 (HSI)": "^HSI", "沪深300 (CSI 300)": "000300.SS", "日经225 (Nikkei 225)": "^N225"}
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
            close = df['Close'].squeeze().tz_localize(None)
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
                        if (close.iloc[current_idx + 1 : target_idx + 1] >= price_at_low).all() and close.iloc[target_idx] > price_at_low:
                            confirmed_rebound_dates.append(close.index[target_idx])
                except: continue

            fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
            ax1.plot(close.index, close.values, label='Price', color='#1f77b4', alpha=0.4)
            for i, date in enumerate(confirmed_rebound_dates):
                ax1.axvline(x=date, color='#00FF00', linestyle='--', alpha=0.8, linewidth=1.5)
            low_points = close[is_new_low]
            if not low_points.empty:
                ax1.scatter(low_points.index, low_points.values, color='red', s=15, label=f'{lookback_weeks}-Week New Low')
            ax1.legend()
            ax2.fill_between(close.index, (close / close.rolling(window_size).max() - 1) * 100, 0, color='red', alpha=0.3)
            ax2.set_ylabel('Drawdown %')
            st.pyplot(fig2)
