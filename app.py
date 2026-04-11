import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# --- 1. 多语言字典配置 ---
LANG_DICT = {
    "简体中文": {
        "nav_vol": "TQQQ 波动看板",
        "nav_idx": "指数抄底分析",
        "settings": "参数设置",
        "conf": "置信水平 (%)",
        "sigma": "标准差倍数 (σ)",
        "period": "回顾周期",
        "run": "开始分析",
        "risk_lvl": "VXN 风险等级",
        "hist_dist": "周收益率分布图",
        "support": "支撑位分析 (Buy/Put)",
        "resistance": "阻力位分析 (Sell/Call)",
        "target": "下周建议位",
        "prob": "概率(IV)",
        "idx_select": "选择指数",
        "weeks": "回溯周数",
        "confirm": "确认天数",
        "start": "起始日期",
        "report": "统计报告",
        "new_low_count": "新低触发总次数",
        "bottom_confirm": "绝对低点确认次数"
    },
    "English": {
        "nav_vol": "TQQQ Volatility",
        "nav_idx": "Index Bottom Analysis",
        "settings": "Settings",
        "conf": "Conf. Level (%)",
        "sigma": "Sigma Multiplier",
        "period": "Period",
        "run": "Run Analysis",
        "risk_lvl": "VXN Risk Level",
        "hist_dist": "Weekly Returns Dist.",
        "support": "Support (Buy/Put)",
        "resistance": "Resistance (Sell/Call)",
        "target": "Target Price",
        "prob": "Prob.(IV)",
        "idx_select": "Select Index",
        "weeks": "Lookback Weeks",
        "confirm": "Confirm Days",
        "start": "Start Date",
        "report": "Stats Report",
        "new_low_count": "Total New Lows",
        "bottom_confirm": "Bottom Confirmed"
    }
}

# --- 2. 侧边栏统一控制 ---
st.set_page_config(page_title="Market Analysis Hub", layout="wide")

with st.sidebar:
    selected_lang = st.selectbox("Language / 语言", options=list(LANG_DICT.keys()))
    L = LANG_DICT[selected_lang]
    
    st.divider()
    # 导航菜单
    app_mode = st.radio("功能切换", [L["nav_vol"], L["nav_idx"]])

# --- 3. 核心功能模块 A: TQQQ 波动看板 ---
if app_mode == L["nav_vol"]:
    st.title("🚀 TQQQ 每周波动全维度看板")
    
    with st.sidebar:
        st.header(L["settings"])
        v_ticker = st.text_input("Ticker", value="TQQQ")
        v_conf = st.slider(L["conf"], 80, 99, 95)
        v_sigma = st.slider(L["sigma"], 1.0, 4.0, 2.0, 0.1)
        v_period = st.selectbox(L["period"], ["1y", "2y", "5y", "10y", "max"], index=3)
        v_run = st.button(L["run"])

    if v_run or v_ticker:
        # 逻辑 1: 数据获取
        tq = yf.Ticker(v_ticker)
        vxn_data = yf.Ticker("^VXN").history(period="1d")
        current_vxn = vxn_data['Close'].iloc[-1] if not vxn_data.empty else 0
        hist = tq.history(period=v_period)
        
        if len(hist) > 20:
            current_price = hist['Close'].iloc[-1]
            weekly_resample = hist.resample('W-MON').agg({'Open': 'first', 'Close': 'last'}).dropna()
            weekly_returns = (weekly_resample['Close'] - weekly_resample['Open']) / weekly_resample['Open']
            std_dev = weekly_returns.std()
            mean_ret = weekly_returns.mean()

            # 风险展示
            bg = "#2e7d32" if current_vxn < 20 else "#fbc02d" if current_vxn < 25 else "#d32f2f"
            st.markdown(f"<div style='background:{bg};padding:15px;border-radius:8px;color:white'><h2>{L['risk_lvl']}: {current_vxn:.2f}</h2></div>", unsafe_allow_html=True)

            # 绘图
            fig, ax = plt.subplots(figsize=(12, 5))
            sns.histplot(weekly_returns, kde=True, ax=ax, color="#8884d8")
            st.pyplot(fig)

            # 表格部分 (简化版逻辑展示)
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"### {L['support']}")
                l_sigma = mean_ret - v_sigma * std_dev
                st.metric(L["target"], f"${current_price*(1+l_sigma):.2f}")
            with col2:
                st.write(f"### {L['resistance']}")
                u_sigma = mean_ret + v_sigma * std_dev
                st.metric(L["target"], f"${current_price*(1+u_sigma):.2f}")

# --- 4. 核心功能模块 B: 指数新低分析 ---
elif app_mode == L["nav_idx"]:
    st.title("📉 多指数『绝对低点锁定』分析")
    
    symbol_map = {
        "纳斯达克100 (NDX)": "^NDX", "标普500 (S&P 500)": "^GSPC",
        "恒生指数 (HSI)": "^HSI", "沪深300 (CSI 300)": "000300.SS",
        "日经225 (Nikkei 225)": "^N225", "德国DAX (DAX)": "^GDAXI",
        "英国富时100 (FTSE 100)": "^FTSE", "韩国综合指数 (KOSPI)": "^KS11"
    }

    with st.sidebar:
        st.header(L["settings"])
        idx_name = st.selectbox(L["idx_select"], list(symbol_map.keys()))
        idx_weeks = st.slider(L["weeks"], 1, 104, 26)
        idx_confirm = st.slider(L["confirm"], 1, 20, 5)
        idx_start = st.text_input(L["start"], "2019-01-01")
        idx_run = st.button(L["run"])

    if idx_run:
        symbol = symbol_map[idx_name]
        window_size = idx_weeks * 5
        
        # 1. 下载数据
        df = yf.download(symbol, start=idx_start)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                close = df['Close'][symbol].copy()
            else:
                close = df['Close'].copy()
            close = close.squeeze().tz_localize(None)

            # 2. 计算逻辑
            rolling_min = close.shift(1).rolling(window=window_size).min()
            is_new_low = close < rolling_min
            
            confirmed_rebound_dates = []
            new_low_dates = close[is_new_low].index

            for low_date in new_low_dates:
                try:
                    current_idx = close.index.get_loc(low_date)
                    target_idx = current_idx + idx_confirm
                    if target_idx < len(close):
                        price_at_low = close.iloc[current_idx]
                        price_at_confirm = close.iloc[target_idx]
                        period_prices = close.iloc[current_idx + 1 : target_idx + 1]
                        
                        is_absolute_low = (period_prices >= price_at_low).all()
                        no_more_new_lows = not is_new_low.iloc[current_idx + 1 : target_idx + 1].any()

                        if price_at_confirm > price_at_low and is_absolute_low and no_more_new_lows:
                            confirmed_rebound_dates.append(close.index[target_idx])
                except: continue

            # 3. 绘图适配 Streamlit
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
            ax1.plot(close.index, close.values, alpha=0.4, label='Price')
            
            for i, date in enumerate(confirmed_rebound_dates):
                ax1.axvline(x=date, color='#00FF00', linestyle='--', alpha=0.8, label="Bottom Confirmed" if i==0 else "")
            
            low_pts = close[is_new_low]
            if not low_pts.empty:
                ax1.scatter(low_pts.index, low_pts.values, color='red', s=15, label='New Low')
            
            ax1.set_title(f"{idx_name} Analysis")
            ax1.legend()
            
            rolling_max = close.rolling(window=window_size, min_periods=1).max()
            drawdown = (close / rolling_max - 1) * 100
            ax2.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
            
            st.pyplot(fig)

            # 4. 统计报告
            st.write(f"### 📊 {L['report']}")
            c1, c2 = st.columns(2)
            c1.metric(L["new_low_count"], is_new_low.sum())
            c2.metric(L["bottom_confirm"], len(confirmed_rebound_dates))
