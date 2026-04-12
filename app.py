import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

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
        "prob_drop": "跌破概率(IV)",
        "prob_break": "突破概率(IV)",
        "risk_level_title": "实时波动强度",
        "current_price": "当前价",
        "vol_weekly": "周σ",
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
        "prob_drop": "Prob. Drop(IV)",
        "prob_break": "Prob. Break(IV)",
        "risk_level_title": "Real-time Vol Intensity",
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
    },
    "日本語": {
        "nav_label": "ナビゲーション",
        "nav_vol": "オプション ボラティリティ看板",
        "nav_idx": "指数新安値分析",
        "settings": "パラメータ設定",
        "run_btn": "分析実行",
        "report_title": "統計レポート",
        "new_low_trigger": "新安値トリガー合計回数",
        "absolute_low_confirm": "底打ち確認（緑線）回数",
        "data_error": "データ不足",
        "idx_settings": "指数パラメータ設定",
        "select_idx": "指数名を選択",
        "back_weeks": "遡及週間の設定",
        "conf_days": "確認日数の設定",
        "start_date": "開始日の設定",
        "strategy": "戦略",
        "suggested_price": "推奨価格",
        "logic_ref": "根拠",
        "prob_drop": "下落確率(IV)",
        "prob_break": "上昇確率(IV)",
        "risk_level_title": "リアルタイムボラ強度",
        "current_price": "現在値",
        "vol_weekly": "週間σ",
        "hist_support": "歴史的サポート",
        "atr_support": "ATR サポート",
        "sigma_support": "σ サポート",
        "hist_resist": "歴史的レジスタンス",
        "atr_resist": "ATR レジスタンス",
        "sigma_resist": "σ レジスタンス",
        "quantile_desc": "パーセンタイル",
        "atr_desc": "ATR マッピング",
        "sigma_desc": "標準偏差倍率"
    }
}

# --- 2. 状态管理 ---
st.set_page_config(page_title="Market Analysis Hub", layout="wide")
if "current_nav_index" not in st.session_state:
    st.session_state.current_nav_index = 0

with st.sidebar:
    selected_lang = st.selectbox("Language / 语言 / 言語", options=list(LANG_DICT.keys()))
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
        ticker_symbol = st.text_input("Ticker Symbol", value="TQQQ").upper()
        confidence_level = st.slider("Confidence Level (%)", 80, 99, 95)
        # 默认 sigma 将由 IV 强度自动推荐，但用户仍可手动调整
        sigma_multiplier = st.slider("Manual Sigma Multiplier", 1.0, 4.0, 2.0, 0.1)
        lookback_period = st.selectbox("Lookback Period", ["1y", "2y", "5y", "10y", "max"], index=3)
        run_v = st.button(L["run_btn"], key="run_v")

    if run_v or ticker_symbol:
        tq = yf.Ticker(ticker_symbol)
        hist = tq.history(period=lookback_period)

        if len(hist) < 20:
            st.error(L["data_error"])
        else:
            current_price = hist['Close'].iloc[-1]
            weekly_resample = hist.resample('W-MON').agg({'Open': 'first', 'Close': 'last'}).dropna()
            weekly_returns = (weekly_resample['Close'] - weekly_resample['Open']) / weekly_resample['Open']
            std_dev = weekly_returns.std()
            mean_ret = weekly_returns.mean()
            
            # --- 核心波动率算法切换 ---
            iv = 0
            vol_source = "Historical"

            # 1. 针对 TQQQ 优先尝试 VXN
            if ticker_symbol == "TQQQ":
                try:
                    vxn_data = yf.Ticker("^VXN").history(period="1d")
                    if not vxn_data.empty:
                        iv = vxn_data['Close'].iloc[-1] / 100
                        vol_source = "VXN Index"
                except:
                    pass

            # 2. 通用：尝试实时隐含波动率 (IV)
            if iv == 0:
                try:
                    options = tq.options
                    if options:
                        opt_chain = tq.option_chain(options[0])
                        calls_puts = opt_chain.puts
                        iv = calls_puts.iloc[(calls_puts['strike'] - current_price).abs().argsort()[:1]]['impliedVolatility'].iloc[0]
                        vol_source = "Real-time IV"
                except:
                    iv = 0

            # 3. 保底：历史波动率 (HV)
            if iv == 0:
                iv = std_dev * np.sqrt(52)
                vol_source = "Historical Vol"

            # --- 针对 IV% 重新设计的 Sigma 倍数建议逻辑 ---
            def get_risk_config(vol_val):
                ref_v = vol_val * 100
                if ref_v < 20: 
                    return "#2e7d32", "Low Vol", 1.5, "极低波动：市场非常平稳 / Calm Market"
                elif ref_v < 40: 
                    return "#fbc02d", "Standard", 2.0, "标准波动：正常个股/ETF水平 / Normal Range"
                elif ref_v < 70: 
                    return "#fb8c00", "High Vol", 2.8, "高波动：杠杆ETF或剧烈震荡 / Aggressive Move"
                else: 
                    return "#d32f2f", "EXTREME", 3.5, "极端波动：高度恐慌或异常行情 / Crisis Mode"

            bg_color, status_text, auto_sigma, advice = get_risk_config(iv)
            
            # 提示用户当前使用的是自动推荐的 Sigma 还是手动 Sigma
            effective_sigma = sigma_multiplier if "Manual Sigma Multiplier" in st.session_state else auto_sigma

            st.markdown(f"""
                <div style='background-color:{bg_color};color:white;padding:15px;border-radius:8px;'>
                    <h2>{L['risk_level_title']}: {status_text} ({iv:.2%})</h2>
                    <p><b>Source:</b> {vol_source} | <b>Recommended Sigma:</b> {auto_sigma} | <b>Advice:</b> {advice}</p>
                </div>
                """, unsafe_allow_html=True)

            # 绘图部分
            fig1, ax = plt.subplots(figsize=(12, 6))
            sns.histplot(weekly_returns, kde=True, bins=40, color="#8884d8", stat="density", alpha=0.3, ax=ax)
            lower_q, upper_q = weekly_returns.quantile([(100 - confidence_level) / 100, confidence_level / 100])
            l_sigma, u_sigma = mean_ret - effective_sigma * std_dev, mean_ret + effective_sigma * std_dev
            for val, col in [(l_sigma, 'red'), (lower_q, 'green'), (upper_q, 'blue'), (u_sigma, 'red')]:
                ax.axvline(val, color=col, linestyle='--', lw=2)
            st.pyplot(fig1)

            def calc_prob(target_p, direction='down'):
                t = 5 / 365
                if iv <= 0: return 0.5
                d2 = (np.log(current_price / target_p) + (- 0.5 * iv**2) * t) / (iv * np.sqrt(t))
                return norm.cdf(-d2) if direction == 'down' else (1 - norm.cdf(-d2))

            high_low = hist['High'] - hist['Low']
            true_range = np.maximum(high_low, np.abs(hist['High'] - hist['Close'].shift()))
            current_atr = true_range.rolling(14).mean().iloc[-1]
            atr_buf = current_atr * np.sqrt(5) * 1.5

            st.write(f"💎 {ticker_symbol} | {L['current_price']}: ${current_price:.2f} | {L['vol_weekly']}: {std_dev:.2%} | Calc-IV: {iv:.2%}")
            
            # 支撑表格
            df_buy = pd.DataFrame([
                [L["hist_support"], current_price * (1 + lower_q), f"{100-confidence_level}% {L['quantile_desc']}"],
                [L["atr_support"], current_price - atr_buf, L["atr_desc"]],
                [f"{effective_sigma}{L['sigma_support']}", current_price * (1 + l_sigma), f"{effective_sigma}{L['sigma_desc']}"]
            ], columns=[L["strategy"], L["suggested_price"], L["logic_ref"]])
            df_buy[L["prob_drop"]] = df_buy[L["suggested_price"]].apply(lambda x: f"{calc_prob(x, 'down'):.2%}")
            st.table(df_buy.style.format({L["suggested_price"]: "${:.2f}"}))

            # 阻力表格
            df_sell = pd.DataFrame([
                [L["hist_resist"], current_price * (1 + upper_q), f"{confidence_level}% {L['quantile_desc']}"],
                [L["atr_resist"], current_price + atr_buf, L["atr_desc"]],
                [f"{effective_sigma}{L['sigma_resist']}", current_price * (1 + u_sigma), f"{effective_sigma}{L['sigma_desc']}"]
            ], columns=[L["strategy"], L["suggested_price"], L["logic_ref"]])
            df_sell[L["prob_break"]] = df_sell[L["suggested_price"]].apply(lambda x: f"{calc_prob(x, 'up'):.2%}")
            st.table(df_sell.style.format({L["suggested_price"]: "${:.2f}"}))

# --- 4. 核心功能 B: 指数分析 (不修改任何相关逻辑) ---
elif app_mode == L["nav_idx"]:
    st.title(f"📉 {L['nav_idx']}")
    symbol_map = {"纳斯达克100 (NDX)": "^NDX", "标普500 (S&P 500)": "^GSPC", "恒生指数 (HSI)": "^HSI", "沪深300 (CSI 300)": "000300.SS", "日经225 (Nikkei 225)": "^N225", "德国DAX (DAX)": "^GDAXI", "英国富时100 (FTSE 100)": "^FTSE", "韩国综合指数 (KOSPI)": "^KS11"}
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
            close = df['Close'][index_symbol].copy() if isinstance(df.columns, pd.MultiIndex) else df['Close'].copy()
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
                        if (close.iloc[current_idx + 1 : target_idx + 1] >= price_at_low).all() and close.iloc[target_idx] > price_at_low:
                            if not is_new_low.iloc[current_idx + 1 : target_idx + 1].any():
                                confirmed_rebound_dates.append(close.index[target_idx])
                except: continue

            fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
            ax1.plot(close.index, close.values, label='Price', color='#1f77b4', alpha=0.4, zorder=1)
            for i, date in enumerate(confirmed_rebound_dates):
                ax1.axvline(x=date, color='#00FF00', linestyle='--', alpha=0.8, linewidth=1.5, label="Bottom Confirmed" if i == 0 else "", zorder=2)
            
            low_points = close[is_new_low]
            if not low_points.empty:
                ax1.scatter(low_points.index, low_points.values, color='red', label=f'{lookback_weeks}-Week New Low', s=15, zorder=3)
            
            ax1.set_title(f'{index_display_name} Analysis')
            ax1.legend(loc='upper left')
            ax2.fill_between(close.index, (close / close.rolling(window_size, min_periods=1).max() - 1) * 100, 0, color='red', alpha=0.3)
            ax2.set_ylabel('Drawdown %')
            st.pyplot(fig2)

            st.write(f"### 📊 {L['report_title']}")
            st.write(f"{L['new_low_trigger']}: {is_new_low.sum()}")
            st.write(f"{L['absolute_low_confirm']}: {len(confirmed_rebound_dates)}")
