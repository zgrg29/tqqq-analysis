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
        "suggested_price": "建议价位",
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
        
        # 动态天数逻辑
        weekday_now = datetime.now().weekday()
        default_days = max(1, 4 - weekday_now + 1) if weekday_now <= 4 else 7
        calc_days = st.slider("Calculation Days (T)", 1, 10, default_days)

        lookback_period = st.selectbox("Lookback Period", ["1y", "2y", "5y", "10y", "max"], index=3) # 默认10年
        run_v = st.button(L["run_btn"], key="run_v")

    if run_v or ticker_symbol:
        tq = yf.Ticker(ticker_symbol)
        hist = tq.history(period=lookback_period)

        if len(hist) < 30:
            st.error(L["data_error"])
        else:
            current_price = hist['Close'].iloc[-1]
            
            # RSI 计算
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rsi = 100 - (100 / (1 + (gain / loss)))
            current_rsi = rsi.iloc[-1]

            # 收益率与波动率
            weekly_resample = hist.resample('W-MON').agg({'Open': 'first', 'Close': 'last'}).dropna()
            weekly_returns = (weekly_resample['Close'] - weekly_resample['Open']) / weekly_resample['Open']
            std_dev = weekly_returns.std()
            mean_ret = weekly_returns.mean()
            hv_annual = std_dev * np.sqrt(52)
            
            # 实时 IV 获取
            iv_realtime = 0
            try:
                options = tq.options
                if options:
                    opt_chain = tq.option_chain(options[0])
                    puts = opt_chain.puts
                    iv_realtime = puts.iloc[(puts['strike'] - current_price).abs().argsort()[:1]]['impliedVolatility'].iloc[0]
            except:
                iv_realtime = 0

            # 最终计算波动率 (取 IV/HV 大值)
            final_vol = max(iv_realtime, hv_annual) if iv_realtime > 0.10 else hv_annual
            vol_source = f"混合 (IV:{iv_realtime:.1%} / HV:{hv_annual:.1%})" if iv_realtime > 0.10 else "纯历史"

            # 风险状态 UI
            def get_risk_config(rsi_val):
                if rsi_val > 70: return "#d32f2f", "超买 (回调风险高)", "⚠️ 建议使用更保守的行权价。"
                elif rsi_val < 30: return "#1976d2", "超卖 (底部机会)", "✅ 适合卖出，权利金可能极其丰厚。"
                else: return "#2e7d32", "正常区间", "ℹ️ 正常操作，关注 Sigma 支撑位。"

            bg_color, status_text, advice = get_risk_config(current_rsi)

            # --- 图表区 ---
            st.subheader("📈 综合市场分析")
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                # 不改变画图部分代码
                fig_dist, ax_dist = plt.subplots(figsize=(10, 5))
                sns.histplot(weekly_returns, kde=True, bins=40, color="#8884d8", stat="density", alpha=0.3, ax=ax_dist)
                lower_q, upper_q = weekly_returns.quantile([(100 - confidence_level) / 100, confidence_level / 100])
                l_sigma_line, u_sigma_line = mean_ret - sigma_multiplier * std_dev, mean_ret + sigma_multiplier * std_dev
                for val, col in [(l_sigma_line, 'red'), (lower_q, 'green'), (upper_q, 'blue'), (u_sigma_line, 'red')]:
                    ax_dist.axvline(val, color=col, linestyle='--', lw=2)
                ax_dist.set_title("Weekly Returns Distribution (KDE)")
                st.pyplot(fig_dist)

            with col_chart2:
                # 不改变画图部分代码
                fig_vol, ax_vol = plt.subplots(figsize=(10, 5))
                ax_vol.bar(['Real-time IV', 'Historical HV'], [iv_realtime, hv_annual], color=['#bb86fc', '#03dac6'])
                ax_vol.set_title(f"Volatility Comparison (Final Used: {final_vol:.1%})")
                st.pyplot(fig_vol)

            st.markdown(f"""
                <div style='background-color:{bg_color};color:white;padding:15px;border-radius:8px;margin-bottom:20px;'>
                    <h3>状态: {status_text} | RSI: {current_rsi:.2f} | 计算天数: {calc_days}d</h3>
                    <p>建议: {advice}</p>
                </div>
                """, unsafe_allow_html=True)

            # --- 概率与建议表格 ---
            def calc_prob(target_p, direction='down'):
                t = calc_days / 365
                if final_vol <= 0: return 0.5
                d2 = (np.log(current_price / target_p) + (- 0.5 * final_vol**2) * t) / (final_vol * np.sqrt(t))
                return norm.cdf(-d2) if direction == 'down' else (1 - norm.cdf(-d2))

            high_low = hist['High'] - hist['Low']
            true_range = np.maximum(high_low, np.abs(hist['High'] - hist['Close'].shift()))
            current_atr = true_range.rolling(14).mean().iloc[-1]
            
            # 【动态修正点 1】：ATR 动态支撑根据计算天数 (calc_days) 进行平方根缩放
            # 原公式固定乘开方5（1周），现在改为动态乘开方 calc_days
            atr_buf = current_atr * np.sqrt(calc_days) * 1.5

            # 【动态修正点 2】：历史概率支撑根据计算天数从“整周5天”转换为“T天”
            # 转换公式：T天收益率 = 5天收益率 / sqrt(5) * sqrt(T)
            dynamic_lower_q = (lower_q / np.sqrt(5)) * np.sqrt(calc_days)

            # 【动态修正点 3】：Sigma 支撑位根据计算天数进行标准年化时间缩放
            # 平均收益率按天线性缩放，标准差按时间开方缩放
            t_annual = calc_days / 365
            l_sigma_val = (mean_ret * (calc_days / 5)) - (sigma_multiplier * final_vol * np.sqrt(t_annual))

            st.write(f"💎 {ticker_symbol} | {L['current_price']}: ${current_price:.2f} | 选定年化波动率: {final_vol:.2%}")
            
            # 支撑表格 (使用经过 calc_days 动态调整后的变量)
            df_buy = pd.DataFrame([
                [L["hist_support"], current_price * (1 + dynamic_lower_q), f"{100-confidence_level}% {L['quantile_desc']} (已按{calc_days}天调整)"],
                [L["atr_support"], current_price - atr_buf, f"{L['atr_desc']} (已按{calc_days}天调整)"],
                [f"{sigma_multiplier}{L['sigma_support']}", current_price + l_sigma_val if l_sigma_val < 0 else current_price * (1 + l_sigma_val), f"基于 {vol_source} (已按{calc_days}天调整)"]
            ], columns=[L["strategy"], L["suggested_price"], L["logic_ref"]])
            
            prob_col = f"{L['prob_drop']}({calc_days}d)"
            df_buy[prob_col] = df_buy[L["suggested_price"]].apply(lambda x: f"{calc_prob(x, 'down'):.2%}")
            st.table(df_buy.style.format({L["suggested_price"]: "${:.2f}"}))

# --- 4. 核心功能 B: 指数分析 (完全保持不变) ---
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
                ax1.scatter(low_points.index, low_points.values, color='red', s=15)
            ax2.fill_between(close.index, (close / close.rolling(window_size).max() - 1) * 100, 0, color='red', alpha=0.3)
            st.pyplot(fig2)