import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# --- 1. 多语言配置 ---
LANG_DICT = {
    "简体中文": {
        "nav_vol": "TQQQ 波动看板",
        "nav_idx": "指数新低分析",
        "lang_label": "语言选择",
        "report_title": "统计报告",
        "new_low_trigger": "新低触发总次数",
        "absolute_low_confirm": "绝对低点确认（绿线）次数"
    },
    "English": {
        "nav_vol": "TQQQ Dashboard",
        "nav_idx": "Index New Low Analysis",
        "lang_label": "Language",
        "report_title": "Statistics Report",
        "new_low_trigger": "Total New Low Triggers",
        "absolute_low_confirm": "Absolute Low Confirmations"
    }
}

# --- 2. 页面初始化 ---
st.set_page_config(page_title="Market Analytics Hub", layout="wide")

with st.sidebar:
    selected_lang = st.selectbox("Language / 语言", options=list(LANG_DICT.keys()))
    L = LANG_DICT[selected_lang]
    st.divider()
    app_mode = st.radio("功能导航", [L["nav_vol"], L["nav_idx"]])

# --- 3. 功能模块：TQQQ 波动看板 (还原原始代码逻辑) ---
if app_mode == L["nav_vol"]:
    st.title("🚀 TQQQ 每周波动全维度看板 (精简美化版)")
    
    with st.sidebar:
        st.header("参数设置")
        ticker_symbol = st.text_input("Ticker Symbol", value="TQQQ")
        confidence_level = st.slider("Confidence Level (%)", min_value=80, max_value=99, value=95)
        sigma_multiplier = st.slider("Sigma Multiplier", min_value=1.0, max_value=4.0, value=2.0, step=0.1)
        lookback_period = st.selectbox("Lookback Period", ["1y", "2y", "5y", "10y", "max"], index=3)
        run_v = st.button("运行分析", key="run_v")

    if run_v or ticker_symbol:
        # --- 原始代码逻辑开始 ---
        tq = yf.Ticker(ticker_symbol)
        vxn_data = yf.Ticker("^VXN").history(period="1d")
        current_vxn = vxn_data['Close'].iloc[-1] if not vxn_data.empty else 0

        hist = tq.history(period=lookback_period)
        if len(hist) < 20:
            st.error("数据不足，无法分析")
        else:
            # 2. 核心计算
            high_low = hist['High'] - hist['Low']
            true_range = np.maximum(high_low, np.abs(hist['High'] - hist['Close'].shift()))
            current_atr = true_range.rolling(14).mean().iloc[-1]

            weekly_resample = hist.resample('W-MON').agg({'Open': 'first', 'Close': 'last'}).dropna()
            weekly_returns = (weekly_resample['Close'] - weekly_resample['Open']) / weekly_resample['Open']

            current_price = hist['Close'].iloc[-1]
            std_dev = weekly_returns.std()
            mean_ret = weekly_returns.mean()

            # 3. 绘图 (适配 Streamlit)
            fig1, ax = plt.subplots(figsize=(12, 6))
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
            st.pyplot(fig1)

            # 4. 风险评估逻辑
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

            # 6. 表格计算与渲染
            try:
                iv = tq.option_chain(tq.options[0]).puts.iloc[(tq.option_chain(tq.options[0]).puts['strike'] - current_price).abs().argsort()[:1]]['impliedVolatility'].iloc[0]
            except:
                iv = std_dev * np.sqrt(52)

            def calc_prob(target_p, direction='down'):
                t = 5 / 365
                d2 = (np.log(current_price / target_p) + (- 0.5 * iv**2) * t) / (iv * np.sqrt(t))
                return norm.cdf(-d2) if direction == 'down' else (1 - norm.cdf(-d2))

            atr_buf = current_atr * np.sqrt(5) * 1.5
            df_buy = pd.DataFrame([
                ["历史概率支撑", current_price * (1 + lower_q), f"{100-confidence_level}% 分位数"],
                ["ATR 动态支撑", current_price - atr_buf, "日线 ATR 映射"],
                [f"{sigma_multiplier}σ 支撑", current_price * (1 + l_sigma), f"{sigma_multiplier}倍标准差"]
            ], columns=["策略方案", "下周建议位", "逻辑参考"])

            df_sell = pd.DataFrame([
                ["历史概率阻力", current_price * (1 + upper_q), f"{confidence_level}% 分位数"],
                ["ATR 动态阻力", current_price + atr_buf, "日线 ATR 映射"],
                [f"{sigma_multiplier}σ 阻力", current_price * (1 + u_sigma), f"{sigma_multiplier}倍标准差"]
            ], columns=["策略方案", "下周建议位", "逻辑参考"])

            df_buy["跌破概率(IV)"] = df_buy["下周建议位"].apply(lambda x: f"{calc_prob(x, 'down'):.2%}")
            df_sell["突破概率(IV)"] = df_sell["下周建议位"].apply(lambda x: f"{calc_prob(x, 'up'):.2%}")

            st.write(f"💎 {ticker_symbol} 统计面板 | 当前价: ${current_price:.2f} | 周σ: {std_dev:.2%} | IV: {iv:.2%}")
            st.table(df_buy.style.format({"下周建议位": "${:.2f}"}))
            st.table(df_sell.style.format({"下周建议位": "${:.2f}"}))
            # --- 原始代码逻辑结束 ---

# --- 4. 功能模块：指数新低分析 (还原原始代码逻辑) ---
elif app_mode == L["nav_idx"]:
    st.title("📉 多指数『绝对低点锁定』分析")
    
    symbol_map = {
        "纳斯达克100 (NDX)": "^NDX", "标普500 (S&P 500)": "^GSPC",
        "恒生指数 (HSI)": "^HSI", "沪深300 (CSI 300)": "000300.SS",
        "日经225 (Nikkei 225)": "^N225", "德国DAX (DAX)": "^GDAXI",
        "英国富时100 (FTSE 100)": "^FTSE", "韩国综合指数 (KOSPI)": "^KS11"
    }

    with st.sidebar:
        st.header("指数参数设置")
        index_display_name = st.selectbox("选择指数名称", list(symbol_map.keys()))
        index_symbol = symbol_map[index_display_name]
        lookback_weeks = st.slider("设置回溯周数", 1, 104, 26)
        confirm_days = st.slider("设置确认天数", 1, 20, 5)
        start_date = st.text_input("设置起始日期", "2019-01-01")
        run_idx = st.button("开始分析指数", key="run_idx")

    if run_idx:
        # --- 原始代码逻辑开始 ---
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
                        is_absolute_low = (period_prices >= price_at_low).all()
                        future_low_signals = is_new_low.iloc[current_idx + 1 : target_idx + 1]
                        no_more_new_lows = not future_low_signals.any()

                        if price_at_confirm > price_at_low and is_absolute_low and no_more_new_lows:
                            confirmed_rebound_dates.append(close.index[target_idx])
                except: continue

            rolling_max = close.rolling(window=window_size, min_periods=1).max()
            drawdown = (close / rolling_max - 1) * 100

            fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
            ax1.plot(close.index, close.values, label='Price', color='#1f77b4', alpha=0.4, zorder=1)
            
            for i, date in enumerate(confirmed_rebound_dates):
                ax1.axvline(x=date, color='#00FF00', linestyle='--', alpha=0.8, linewidth=1.5, label="Bottom Confirmed" if i == 0 else "", zorder=2)
            
            low_points = close[is_new_low]
            if not low_points.empty:
                ax1.scatter(low_points.index, low_points.values, color='red', label=f'{lookback_weeks}-Week New Low', s=15, zorder=3)

            ax1.set_title(f'{index_display_name} - {lookback_weeks}W New Low Analysis', fontsize=16)
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)
            ax2.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
            ax2.set_ylabel('Drawdown %')
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)

            st.write(f"### --- {index_display_name} {L['report_title']} ---")
            st.write(f"{L['new_low_trigger']}: {is_new_low.sum()}")
            st.write(f"{L['absolute_low_confirm']}: {len(confirmed_rebound_dates)}")
        # --- 原始代码逻辑结束 ---
