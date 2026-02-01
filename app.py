import os
from datetime import datetime, timedelta
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# -----------------------
# AUTHENTICATION
# -----------------------
def check_password():
    """
    Use Streamlit secrets (set in Streamlit Cloud secrets)
    key: dashboard_password
    Fallback: environment variable DASHBOARD_PASSWORD
    """
    secret_pwd = None
    try:
        secret_pwd = st.secrets.get("dashboard_password")
    except Exception:
        secret_pwd = None

    if not secret_pwd:
        secret_pwd = os.environ.get("DASHBOARD_PASSWORD")

    if not secret_pwd:
        st.error("Dashboard password not configured. Set `dashboard_password` in Streamlit secrets.")
        st.stop()

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        pwd = st.text_input("Enter Dashboard Password", type="password", key="pwd_input")
        if pwd:
            if pwd == secret_pwd:
                st.session_state.authenticated = True
                # use experimental_rerun to refresh app after auth
                try:
                    st.experimental_rerun()
                except Exception:
                    # fallback to rerun if API differs
                    try:
                        st.rerun()
                    except Exception:
                        pass
            else:
                st.warning("Incorrect password.")
                st.stop()

# Call auth
check_password()

# -----------------------
# CONFIG / STRATEGIES
# -----------------------
st.set_page_config(page_title="Nifty Ultimate Quant Lab", layout="wide")

STRATEGY_GROUPS = {
    "Bullish": ["Buy Call", "Sell Put", "Bull Call Spread", "Bull Put Spread", "Buy Future", "Long Synthetic Future"],
    "Bearish": ["Buy Put", "Sell Call", "Bear Put Spread", "Bear Call Spread", "Short Synthetic Future"],
    "Neutral": ["Short Straddle", "Short Strangle", "Iron Butterfly", "Short Iron Condor", "Jade Lizard", "Batman"],
    "Volatility": ["Long Straddle", "Long Strangle", "Long Iron Condor", "Call Ratio Back Spread"]
}

# -----------------------
# DATA ENGINE
# -----------------------
@st.cache_data(ttl=86400)  # refresh daily
def get_nifty_data() -> pd.DataFrame:
    df = yf.download("^NSEI", period="10y", interval="1d", progress=False)
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['MonthName'] = df['Date'].dt.month_name()
    # pandas >=1.1: isocalendar() returns a DataFrame
    try:
        df['Week'] = df['Date'].dt.isocalendar().week
    except Exception:
        df['Week'] = df['Date'].dt.week
    df['Day'] = df['Date'].dt.day_name()
    df['Daily_Return'] = df['Close'].pct_change()
    df['Vol'] = df['Daily_Return'].rolling(20).std() * np.sqrt(252)
    df['Intraday_Vol'] = (df['High'] - df['Low']) / df['Low'] * 100
    return df.dropna().reset_index(drop=True)

data = get_nifty_data()

# -----------------------
# BACKTEST ENGINE (FIXED FOR PANDAS DIMENSIONS)
# -----------------------
@st.cache_data
def run_backtest_cached(df: pd.DataFrame, strat_name: str) -> pd.DataFrame:
    expiries = df[df['Day'] == 'Tuesday'].reset_index(drop=True)
    n = len(expiries)
    if n < 2:
        return pd.DataFrame(columns=["Date", "Year", "Week", "PnL", "Exit_Price", "Cumulative"])

    entry = expiries['Close'].values[:-1].astype(float)
    exitp = expiries['Close'].values[1:].astype(float)
    years = expiries['Year'].values[1:].astype(int)
    weeks = expiries['Week'].values[1:].astype(int)
    vol = expiries['Vol'].values[:-1].astype(float)
    move = exitp - entry
    atm_prem = 0.4 * vol * entry * np.sqrt(7/365)

    pnl = np.zeros_like(move, dtype=float)

    if strat_name == "Buy Call":
        pnl = np.maximum(0, move) - atm_prem
    elif strat_name == "Sell Put":
        pnl = atm_prem - np.maximum(0, entry - exitp)
    elif strat_name == "Bull Call Spread":
        pnl = np.minimum(entry*0.015, np.maximum(0, move)) - (atm_prem*0.6)
    elif strat_name == "Bull Put Spread":
        pnl = (atm_prem*0.4) - np.minimum(entry*0.015, np.maximum(0, entry - exitp))
    elif strat_name in ("Buy Future", "Long Synthetic Future"):
        pnl = move
    elif strat_name == "Buy Put":
        pnl = np.maximum(0, -move) - atm_prem
    elif strat_name == "Sell Call":
        pnl = atm_prem - np.maximum(0, move)
    elif strat_name == "Bear Put Spread":
        pnl = np.minimum(entry*0.015, np.maximum(0, -move)) - (atm_prem*0.6)
    elif strat_name == "Bear Call Spread":
        pnl = (atm_prem*0.4) - np.minimum(entry*0.015, np.maximum(0, move))
    elif strat_name == "Short Synthetic Future":
        pnl = -move
    elif strat_name == "Short Straddle":
        pnl = (atm_prem * 2) - np.abs(move)
    elif strat_name == "Short Strangle":
        pnl = (atm_prem * 1.5) - np.abs(move)
    elif strat_name == "Short Iron Condor":
        pnl = np.where(np.abs(move) < entry*0.02, (atm_prem * 0.8) - np.abs(move), -(entry*0.01))
    elif strat_name == "Iron Butterfly":
        pnl = np.where(np.abs(move) < entry*0.01, (atm_prem * 1.2) - np.abs(move), -(entry*0.005))
    elif strat_name == "Jade Lizard":
        pnl = np.where(move > -50, atm_prem, (atm_prem + (move + 50)))
    elif strat_name == "Batman":
        pnl = np.where(np.abs(move) < entry*0.01, atm_prem * 1.2, -atm_prem)
    elif strat_name == "Long Straddle":
        pnl = np.abs(move) - (atm_prem * 2)
    elif strat_name == "Long Strangle":
        pnl = np.abs(move) - (atm_prem * 1.5)
    elif strat_name == "Call Ratio Back Spread":
        pnl = np.where(move > 0, np.maximum(0, move*2) - atm_prem, -atm_prem*0.5)

    # Use np.ravel() to ensure 1D arrays to prevent ValueError
    res = pd.DataFrame({
        "Date": expiries['Date'].iloc[1:].values,
        "Year": np.ravel(years),
        "Week": np.ravel(weeks),
        "PnL": np.ravel(pnl),
        "Exit_Price": np.ravel(exitp)
    })
    res['Cumulative'] = res['PnL'].cumsum()
    return res

@st.cache_data
def compute_all_summaries(df: pd.DataFrame, strategy_groups: dict) -> pd.DataFrame:
    summaries = []
    for view, strategies in strategy_groups.items():
        for s in strategies:
            r = run_backtest_cached(df, s)
            summaries.append({"Strategy": s, "View": view, "Total Pts": float(r['PnL'].sum())})
    return pd.DataFrame(summaries)

def get_trade_results_for_choice(choice: str) -> pd.DataFrame:
    with st.spinner(f"Running backtest for {choice}..."): 
        return run_backtest_cached(data, choice)

# -----------------------
# UI TABS
# -----------------------
st.title("🏛️ Nifty 50: Ultimate Strategy & Predictive Lab")
tabs = st.tabs(["🎯 Strategy Lab", "🏆 Leaderboard", "📅 Seasonality", "🔬 Micro-Insights", "🔮 Forecast", "⚡ Tactical Simulator","📅 Deep-Dive", "📊 Dividends"])

# TAB 1: Strategy Lab
with tabs[0]:
    group = st.selectbox("Market View", list(STRATEGY_GROUPS.keys()), index=0)
    choice = st.selectbox("Select Strategy", STRATEGY_GROUPS[group])
    trade_results = get_trade_results_for_choice(choice)
    c1, c2 = st.columns(2)
    c1.metric("Total PnL (Pts)", f"{trade_results['PnL'].sum():,.0f}" if not trade_results.empty else "0")
    c2.metric("Win Rate", f"{(trade_results['PnL'] > 0).mean()*100:.1f}%" if not trade_results.empty else "0.0%")
    st.plotly_chart(px.line(trade_results, x="Date", y="Cumulative", title=f"Equity Curve: {choice}"), use_container_width=True)

# TAB 2: Leaderboard
with tabs[1]:
    st.header("🏆 Multi-Strategy Leaderboard")
    all_summary_df = compute_all_summaries(data, STRATEGY_GROUPS)
    if not all_summary_df.empty:
        st.dataframe(all_summary_df.sort_values("Total Pts", ascending=False), use_container_width=True)
    else:
        st.info("No strategy summaries available.")

# TAB 3: Seasonality
with tabs[2]:
    import seaborn as sns
    import matplotlib.pyplot as plt
    view_type = st.radio("Level", ["Month", "Day"], horizontal=True)
    if view_type == "Month":
        pivot = data.pivot_table(index='Year', columns='MonthName', values='Daily_Return', aggfunc='sum') * 100
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(pivot, annot=True, cmap="RdYlGn", fmt=".1f", ax=ax)
        st.pyplot(fig)
    else:
        pivot = data.pivot_table(index='MonthName', columns='Day', values='Daily_Return', aggfunc='mean') * 100
        # ensure consistent month ordering
        month_order = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        pivot = pivot.reindex(index=month_order)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(pivot, annot=True, cmap="RdYlGn", ax=ax)
        st.pyplot(fig)

# TAB 4: Micro-Insights
with tabs[3]:
    v_choice = st.selectbox("Select Strategy for Insights", sum(list(STRATEGY_GROUPS.values()), []), index=0)
    tr = get_trade_results_for_choice(v_choice)
    if not tr.empty:
        st.plotly_chart(px.bar(tr.tail(20), x="Date", y="PnL", color="PnL"), use_container_width=True)
    else:
        st.info("No PnL data for selected strategy.")

# TAB 5: Forecast
with tabs[4]:
    st.header("🔮 Monte Carlo Forecast")
    recent = data.tail(500)
    mu, sigma, last_p = recent['Daily_Return'].mean(), recent['Daily_Return'].std(), data['Close'].iloc[-1]
    sims = np.zeros((5, 1000))
    for i in range(1000):
        p = last_p
        for d in range(5):
            p *= (1 + np.random.normal(mu, sigma))
            sims[d, i] = p
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=np.percentile(sims, 95, axis=1), name="Upper 95%"))
    fig.add_trace(go.Scatter(y=np.median(sims, axis=1), name="Median"))
    fig.add_trace(go.Scatter(y=np.percentile(sims, 5, axis=1), name="Lower 5%"))
    st.plotly_chart(fig, use_container_width=True)

# TAB 6: Tactical Simulator
with tabs[5]:
    st.header("⚡ Tactical Strategy Lab: Performance & Win Rate")
    all_years = sorted(data['Year'].unique(), reverse=True)
    sel_years = st.multiselect("Select Backtest Years", options=all_years, default=all_years[:2])
    t1, t2, t3 = st.columns(3)
    with t1:
        sel_day = st.selectbox("Execution Day", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'], index=1)
        st_action = st.selectbox("Strategy", ["Sell Call", "Sell Put", "Buy Call", "Buy Put"])
        vix_lim = st.slider("VIX Filter (Skip if VIX > X)", 10.0, 30.0, 19.0)
    with t2:
        hedge_pts = st.number_input("Hedge Distance (0 = Naked)", value=150, step=50)
        lot_size = 65
        num_lots = st.number_input("Number of Lots", value=1, min_value=1)
    with t3:
        sl_pct = st.slider("Hard Stop Loss (%)", 0.1, 3.0, 1.2)
        cap_hedged = 42000
        cap_naked = 135000
        margin_per_lot = cap_naked if (hedge_pts == 0 or "Buy" in st_action) else cap_hedged
        initial_capital = margin_per_lot * num_lots

    df_lab = data[data['Year'].isin(sel_years)].copy()
    df_lab['VIX_Proxy'] = df_lab['Vol'] * 100
    df_lab = df_lab.sort_values('Date')

    def run_calculation(row, target_day):
        if row['Day'] != target_day or row['VIX_Proxy'] > vix_lim:
            return (None, "Skip")
        move = row['Close'] - row['Open']
        move_pct = (abs(move) / row['Open']) * 100
        s_prem = row['Open'] * 0.005
        b_prem = row['Open'] * 0.001 if hedge_pts > 0 else 0
        if move_pct > sl_pct:
            return (-(row['Open'] * (sl_pct/100)), "SL Hit")
        if st_action == "Sell Call":
            p = (s_prem - b_prem) - (max(0, move) if hedge_pts == 0 else min(max(0, move), hedge_pts))
        elif st_action == "Sell Put":
            p = (s_prem - b_prem) - (max(0, -move) if hedge_pts == 0 else min(max(0, -move), hedge_pts))
        elif st_action == "Buy Call":
            p = max(0, move) - s_prem
        elif st_action == "Buy Put":
            p = max(0, -move) - s_prem
        else:
            p = 0
        return (p, "Success")

    results = df_lab.apply(lambda x: run_calculation(x, sel_day), axis=1)
    df_lab['Pts'], df_lab['Status'] = zip(*results)
    exec_df = df_lab[df_lab['Status'] != "Skip"].copy()

    if not exec_df.empty:
        exec_df['PnL_Cash'] = exec_df['Pts'] * lot_size * num_lots
        exec_df['Running_Cap'] = initial_capital + exec_df['PnL_Cash'].cumsum()

        total_pnl = exec_df['PnL_Cash'].sum()
        win_rate = (len(exec_df[exec_df['PnL_Cash'] > 0]) / len(exec_df)) * 100
        sharpe = (exec_df['PnL_Cash'].mean() / exec_df['PnL_Cash'].std()) * np.sqrt(52) if exec_df['PnL_Cash'].std() != 0 else 0
        days_span = (exec_df['Date'].max() - exec_df['Date'].min()).days if not exec_df['Date'].empty else 1
        # annualize CAGR properly
        years_span = max(1, days_span / 365)
        cagr = (((exec_df['Running_Cap'].iloc[-1]) / initial_capital) ** (1/years_span) - 1) * 100

        exec_df['Peak'] = exec_df['Running_Cap'].cummax()
        exec_df['Drawdown'] = (exec_df['Running_Cap'] - exec_df['Peak']) / exec_df['Peak'] * 100

        st.divider() 
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Initial Capital", f"₹{initial_capital:,.0f}")
        m2.metric("Total PnL", f"₹{total_pnl:,.0f}")
        m3.metric("CAGR", f"{cagr:.2f}%")
        m4.metric("Sharpe Ratio", f"{sharpe:.2f}")

        w1, w2, w3, w4 = st.columns(4)
        w1.metric("Win Rate", f"{win_rate:.1f}%")
        w2.metric("Max Drawdown", f"{exec_df['Drawdown'].min():.2f}%")
        w3.metric("Lot Size", lot_size)
        w4.metric("Total Lots", num_lots)

        st.subheader(f"📈 Equity Curve: {st_action} on {sel_day}s")
        fig_equity = px.area(exec_df, x='Date', y='Running_Cap', title="Portfolio Growth Over Time", color_discrete_sequence=['#00CC96'])
        st.plotly_chart(fig_equity, use_container_width=True)

        st.subheader("🗓️ Performance Insights")
        h1, h2 = st.columns(2)
        with h1:
            pivot_pnl = exec_df.groupby(['Year', 'Month'])['PnL_Cash'].sum().unstack().fillna(0)
            st.plotly_chart(px.imshow(pivot_pnl, text_auto=True, title="Monthly Profit Heatmap", color_continuous_scale='RdYlGn'), use_container_width=True)
        with h2:
            day_comp = []
            for d in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
                d_pnl = df_lab[df_lab['Day'] == d].apply(lambda x: run_calculation(x, d)[0], axis=1).fillna(0).sum() * lot_size * num_lots
                day_comp.append({"Day": d, "Total PnL": d_pnl})
            st.plotly_chart(px.bar(pd.DataFrame(day_comp), x='Day', y='Total PnL', title="Strategy Comparison by Day", color='Total PnL', color_continuous_scale='RdYlGn'), use_container_width=True)

        st.subheader(f"📋 Trade Log ({sel_day} Execution Only)")
        st.dataframe(exec_df[['Date', 'Status', 'VIX_Proxy', 'PnL_Cash', 'Running_Cap', 'Drawdown']].sort_values('Date', ascending=False), use_container_width=True)
    else:
        st.warning("No trades match these filters. Adjust VIX or Day selection.")

# TAB 7: Weekly Game Plan (Deep-Dive / Professional)
with tabs[6]:
    st.header("📅 Weekly Strategy Game Plan")
    with st.expander("⚙️ Settings, Risk & Brokerage", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            all_yrs_t7 = sorted(data['Year'].unique(), reverse=True)
            sel_yrs_t7 = st.multiselect("Backtest Years", options=all_yrs_t7, default=all_yrs_t7[:2], key="t7_y_fin")
            capital_per_lot = st.number_input("Capital per Lot (₹)", value=150000, step=10000)
        with c2:
            h_pts_t7 = st.number_input("Hedge Pts (0=Naked)", value=150, step=50, key="t7_h_fin")
            lot_size_base = 65
            sl_val_t7 = st.slider("Hard Stop Loss (%)", 0.1, 5.0, 1.2, key="t7_sl_fin")
        with c3:
            st.write("**Brokerage Settings**")
            brokerage_per_order = st.number_input("Brokerage per Order (₹)", value=20)
            tax_stt_pct = st.number_input("Taxes/STT/Slippage (%)", value=0.05, step=0.01) / 100

    st.subheader("🛠️ Step 1: Define Your Weekly Playbook")
    day_cols = st.columns(5)
    days_list = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    playbook = {}
    for i, day in enumerate(days_list):
        with day_cols[i]:
            playbook[day] = st.radio(f"Action: {day}", ["No Trade", "Sell Call", "Sell Put", "Buy Call", "Buy Put"], index=0, key=f"p_fin_{day}")

    def run_professional_backtest(row):
        current_day = row['Day']
        if current_day not in playbook or playbook[current_day] == "No Trade":
            return (0, "Skip", 0, 0)
        strat = playbook[current_day]
        move, h_move, l_move = row['Close']-row['Open'], row['High']-row['Open'], row['Low']-row['Open']
        s_prem = row['Open'] * 0.005
        b_prem = row['Open'] * 0.001 if h_pts_t7 > 0 else 0
        net_c = s_prem - b_prem
        if strat == "Sell Call":
            pnl = net_c - (max(0, move) if h_pts_t7 == 0 else min(max(0, move), h_pts_t7))
            m_l = net_c - (max(0, h_move) if h_pts_t7 == 0 else min(max(0, h_move), h_pts_t7))
            m_h = net_c - (max(0, l_move) if h_pts_t7 == 0 else min(max(0, l_move), h_pts_t7))
        elif strat == "Sell Put":
            pnl = net_c - (max(0, -move) if h_pts_t7 == 0 else min(max(0, -move), h_pts_t7))
            m_l = net_c - (max(0, -l_move) if h_pts_t7 == 0 else min(max(0, -l_move), h_pts_t7))
            m_h = net_c - (max(0, -h_move) if h_pts_t7 == 0 else min(max(0, -h_move), h_pts_t7))
        elif strat == "Buy Call":
            pnl = net_c - s_prem
            m_h, m_l = max(0, move)-s_prem, max(0, h_move)-s_prem, max(0, l_move)-s_prem
        elif strat == "Buy Put":
            pnl = max(0, -move)-s_prem
            m_h, m_l = max(0, -l_move)-s_prem, max(0, -h_move)-s_prem
        else:
            pnl = 0; m_l = 0; m_h = 0

        if (abs(move)/row['Open']*100) > sl_val_t7:
            pnl, status = -(row['Open'] * (sl_val_t7/100)), "SL Hit"
        else:
            status = "Success"
        return (pnl, status, m_l, m_h)

    df_t7 = data[data['Year'].isin(sel_yrs_t7)].copy().sort_values('Date')
    active_days = [d for d, s in playbook.items() if s != "No Trade"]

    if active_days:
        res = df_t7.apply(run_professional_backtest, axis=1)
        df_t7['Pts'], df_t7['Status'], df_t7['MTM_L_Pts'], df_t7['MTM_H_Pts'] = zip(*res)
        exec_df = df_t7[df_t7['Status'].isin(["Success", "SL Hit"])].copy()

        if not exec_df.empty:
            current_cap = capital_per_lot
            results_list = []
            for i, row in exec_df.iterrows():
                lots = max(1, int(current_cap // capital_per_lot))
                gross_pnl = row['Pts'] * lot_size_base * lots
                charges = (2 * brokerage_per_order) + (row['Open'] * lot_size_base * lots * tax_stt_pct)
                net_pnl = gross_pnl - charges
                current_cap += net_pnl
                results_list.append({
                    'Net_PnL': net_pnl,
                    'Gross_PnL': gross_pnl,
                    'Charges': charges,
                    'Running_Cap': current_cap,
                    'Lots': lots,
                    'MTM_L': row['MTM_L_Pts'] * lot_size_base * lots,
                    'MTM_H': row['MTM_H_Pts'] * lot_size_base * lots
                })
            res_df = pd.DataFrame(results_list)
            exec_df = pd.concat([exec_df.reset_index(drop=True), res_df], axis=1)

        # Performance calculations & UI display with guards
        if exec_df.empty:
            st.info("No executed trades for selected playbook & years.")
        else:
            total_net = exec_df['Net_PnL'].sum()
            exec_df['Peak'] = exec_df['Running_Cap'].cummax()
            exec_df['Drawdown_Pct'] = (exec_df['Running_Cap'] - exec_df['Peak']) / exec_df['Peak'] * 100

            is_win = (exec_df['Net_PnL'] > 0).astype(int)
            streaks = is_win.groupby((is_win != is_win.shift()).cumsum()).cumcount() + 1
            max_w_s = streaks[is_win == 1].max() if any(is_win == 1) else 0
            max_l_s = streaks[is_win == 0].max() if any(is_win == 0) else 0

            st.divider()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Net Profit", f"₹{total_net:,.0f}")
            m2.metric("CAGR (Net)", f"{(((exec_df['Running_Cap'].iloc[-1]/capital_per_lot)**(1/max(1,len(sel_yrs_t7))))-1)*100:.2f}%")
            m3.metric("Win Rate", f"{(len(exec_df[exec_df['Net_PnL'] > 0]) / len(exec_df) * 100):.1f}%")
            m4.metric("Sharpe Ratio", f"{(exec_df['Net_PnL'].mean()/exec_df['Net_PnL'].std())*np.sqrt(252):.2f}" if exec_df['Net_PnL'].std() != 0 else "0.00")

            st.subheader("📈 Institutional Yearly Performance")
            yoy = exec_df.groupby('Year').agg(
                Net_Profit=('Net_PnL', 'sum'),
                Total_Trades=('Net_PnL', 'count'),
                Profit_Trades=('Net_PnL', lambda x: (x > 0).sum()),
                Win_Rate=('Net_PnL', lambda x: (x > 0).mean() * 100),
            )
            yoy['Max_DD_%'] = exec_df.groupby('Year').apply(lambda x: ((x['Running_Cap'] - x['Running_Cap'].cummax()) / x['Running_Cap'].cummax()).min() * 100).values
            st.table(yoy.style.format({
                "Net_Profit": "₹{:,.0f}", "Total_Trades": "{:,.0f}",
                "Profit_Trades": "{:,.0f}", "Win_Rate": "{:.1f}%", "Max_DD_%": "{:.2f}%"
            }))

            st.write("**Monthly Breakdown (Balanced at ₹0)**")
            mom = exec_df.groupby(['Year', 'MonthName'])['Net_PnL'].sum().unstack().fillna(0)
            month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
            mom = mom.reindex(columns=[m for m in month_order if m in mom.columns])
            if not mom.empty:
                limit = max(abs(mom.values.min()), abs(mom.values.max()))
                st.plotly_chart(px.imshow(mom, text_auto=",.0f", color_continuous_scale='RdYlGn', range_color=[-limit, limit], aspect="auto").update_layout(coloraxis_showscale=False), use_container_width=True)
            st.area_chart(exec_df.set_index('Date')['Running_Cap'])
            st.subheader("📋 Detailed Trade Log")
            st.dataframe(exec_df[['Date', 'Day', 'Open', 'Close', 'Lots', 'Net_PnL', 'Gross_PnL', 'Charges', 'MTM_L', 'MTM_H', 'Running_Cap', 'Status']].sort_values('Date', ascending=False), use_container_width=True)
    else:
        st.info("💡 **MCP Note:** Complete your Strategy Playbook to see AI Performance Insights here.")

# TAB 8: Dividends / Placeholder
with tabs[7]:
    st.write("Dividend analysis placeholder.")