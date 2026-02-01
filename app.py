def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        password = st.text_input("Enter Dashboard Password", type="password")
        if password == "password": # Change this!
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.stop()

check_password()



import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def get_mcp_insights(data, context="general"):
    win_rate = (data['Net_PnL'] > 0).mean() * 100
    avg_win = data[data['Net_PnL'] > 0]['Net_PnL'].mean()
    avg_loss = data[data['Net_PnL'] <= 0]['Net_PnL'].mean()
    expectancy = (win_rate/100 * avg_win) + ((1-win_rate/100) * avg_loss)

    if context == "performance":
        return {
            "insight": f"Current Profit Expectancy is ₹{expectancy:.2f} per trade.",
            "suggestion": "Increase position size by 20% on high-win-rate days to accelerate compounding.",
            "plan": "Limit Max Daily Loss to 1.5% of total capital to protect the current equity curve."
        }
    elif context == "seasonality":
        best_day = data.groupby('Day')['Net_PnL'].mean().idxmax()
        return {
            "insight": f"{best_day} is your 'Alpha Day' with the highest average consistency.",
            "suggestion": f"Avoid heavy 'Mean Reversion' trades on {best_day}; stick to 'Trend Following'.",
            "plan": f"Deploy 60% of capital on {best_day} and reduce to 30% on your weakest day."
        }

# 1. PAGE SETUP
st.set_page_config(page_title="Nifty Ultimate Quant Lab", layout="wide")

# 2. COMPLETE STRATEGY DEFINITIONS
STRATEGY_GROUPS = {
    "Bullish": ["Buy Call", "Sell Put", "Bull Call Spread", "Bull Put Spread", "Buy Future", "Long Synthetic Future"],
    "Bearish": ["Buy Put", "Sell Call", "Bear Put Spread", "Bear Call Spread", "Short Synthetic Future"],
    "Neutral": ["Short Straddle", "Short Strangle", "Iron Butterfly", "Short Iron Condor", "Jade Lizard", "Batman"],
    "Volatility": ["Long Straddle", "Long Strangle", "Long Iron Condor", "Call Ratio Back Spread"]
}

# 3. DATA ENGINE
@st.cache_data
def get_nifty_data():
    df = yf.download("^NSEI", period="10y", interval="1d", multi_level_index=False)
    df.reset_index(inplace=True)
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['MonthName'] = df['Date'].dt.month_name()
    df['Week'] = df['Date'].dt.isocalendar().week
    df['Day'] = df['Date'].dt.day_name()
    df['Daily_Return'] = df['Close'].pct_change()
    df['Vol'] = df['Daily_Return'].rolling(20).std() * np.sqrt(252)
    df['Intraday_Vol'] = (df['High'] - df['Low']) / df['Low'] * 100
    return df.dropna()

data = get_nifty_data()

# 4. FULL BACKTEST ENGINE
def run_backtest(df, strat_name):
    logs = []
    # Using Tuesday as proxy for weekly cycles
    expiries = df[df['Day'] == 'Tuesday'].copy()
    
    for i in range(len(expiries) - 1):
        entry, exit = expiries.iloc[i]['Close'], expiries.iloc[i+1]['Close']
        vol, move = expiries.iloc[i]['Vol'], exit - entry
        atm_prem = 0.4 * vol * entry * np.sqrt(7/365)
        
        pnl = 0
        if strat_name == "Buy Call": pnl = max(0, move) - atm_prem
        elif strat_name == "Sell Put": pnl = atm_prem - max(0, entry - exit)
        elif strat_name == "Bull Call Spread": pnl = min(entry*0.015, max(0, move)) - (atm_prem*0.6)
        elif strat_name == "Bull Put Spread": pnl = (atm_prem*0.4) - min(entry*0.015, max(0, entry-exit))
        elif strat_name == "Buy Future" or strat_name == "Long Synthetic Future": pnl = move
        elif strat_name == "Buy Put": pnl = max(0, -move) - atm_prem
        elif strat_name == "Sell Call": pnl = atm_prem - max(0, move)
        elif strat_name == "Bear Put Spread": pnl = min(entry*0.015, max(0, -move)) - (atm_prem*0.6)
        elif strat_name == "Bear Call Spread": pnl = (atm_prem*0.4) - min(entry*0.015, max(0, move))
        elif strat_name == "Short Synthetic Future": pnl = -move
        elif strat_name == "Short Straddle": pnl = (atm_prem * 2) - abs(move)
        elif strat_name == "Short Strangle": pnl = (atm_prem * 1.5) - abs(move)
        elif strat_name == "Short Iron Condor": pnl = (atm_prem * 0.8) - abs(move) if abs(move) < entry*0.02 else -(entry*0.01)
        elif strat_name == "Iron Butterfly": pnl = (atm_prem * 1.2) - abs(move) if abs(move) < entry*0.01 else -(entry*0.005)
        elif strat_name == "Jade Lizard": pnl = atm_prem if move > -50 else (atm_prem + (move + 50))
        elif strat_name == "Batman": pnl = atm_prem * 1.2 if abs(move) < entry*0.01 else -atm_prem
        elif strat_name == "Long Straddle": pnl = abs(move) - (atm_prem * 2)
        elif strat_name == "Long Strangle": pnl = abs(move) - (atm_prem * 1.5)
        elif strat_name == "Call Ratio Back Spread": pnl = max(0, move*2) - atm_prem if move > 0 else -atm_prem*0.5

        logs.append({
            "Date": expiries.iloc[i+1]['Date'],
            "Year": int(expiries.iloc[i+1]['Year']),
            "Week": int(expiries.iloc[i+1]['Week']),
            "PnL": pnl,
            "Exit_Price": exit
        })
    res = pd.DataFrame(logs)
    res['Cumulative'] = res['PnL'].cumsum()
    return res

# 5. DASHBOARD TABS
st.title("🏛️ Nifty 50: Ultimate Strategy & Predictive Lab")
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🎯 Strategy Lab", "🏆 Leaderboard", "📅 Seasonality",
    "🔬 Micro-Insights", "🔮 Forecast", "⚡ Tactical Simulator","📅 Deep-Dive Calendar Backtester",
    "📊 Dividend Impact & Ex-Date Analyzer"
])

with tab1:
    group = st.selectbox("Market View", list(STRATEGY_GROUPS.keys()))
    choice = st.selectbox("Select Strategy", STRATEGY_GROUPS[group])
    trade_results = run_backtest(data, choice)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total PnL (Pts)", f"{trade_results['PnL'].sum():,.0f}")
    c2.metric("Win Rate", f"{(trade_results['PnL'] > 0).mean()*100:.1f}%")
    st.plotly_chart(px.line(trade_results, x="Date", y="Cumulative", title=f"Equity Curve: {choice}"), width='stretch')
      

with tab2:
    st.header("🏆 Multi-Strategy Leaderboard")
    all_summary = []
    for g, strats in STRATEGY_GROUPS.items():
        for s in strats:
            r = run_backtest(data, s)
            all_summary.append({"Strategy": s, "View": g, "Total Pts": r['PnL'].sum()})
    st.dataframe(pd.DataFrame(all_summary).sort_values("Total Pts", ascending=False).style.background_gradient(cmap='RdYlGn'), width='stretch')

with tab3:
    st.header("📅 Seasonality Intelligence")
    min_year, max_year = int(data['Year'].min()), int(data['Year'].max())
    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        selected_years = st.slider("Select Analysis Period", min_year, max_year, (2018, max_year))
    with col_f2:
        view_type = st.radio("Select Detail Level", ["Month over Month", "Week over Week", "Day over Day", "Week over Day"], horizontal=True)

    f_data = data[(data['Year'] >= selected_years[0]) & (data['Year'] <= selected_years[1])]
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

    if view_type == "Month over Month":
        mom_pivot = f_data.pivot_table(index='Year', columns='Month', values='Daily_Return', aggfunc='sum') * 100
        mom_pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(mom_pivot, annot=True, cmap="RdYlGn", center=0, fmt=".1f", ax=ax)
        st.pyplot(fig)

    elif view_type == "Week over Week":
        wow_pivot = f_data.pivot_table(index='Year', columns='Week', values='Daily_Return', aggfunc='sum') * 100
        fig, ax = plt.subplots(figsize=(16, 8))
        sns.heatmap(wow_pivot, annot=False, cmap="RdYlGn", center=0, ax=ax)
        st.pyplot(fig)

    elif view_type == "Day over Day":
        dod_pivot = f_data.pivot_table(index='MonthName', columns='Day', values='Daily_Return', aggfunc='mean') * 100
        dod_pivot = dod_pivot.reindex(index=month_order, columns=day_order).fillna(0)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(dod_pivot, annot=True, cmap="RdYlGn", center=0, fmt=".2f", ax=ax)
        st.pyplot(fig)

    elif view_type == "Week over Day":
        wod_pivot = f_data.pivot_table(index='Week', columns='Day', values='Daily_Return', aggfunc='mean') * 100
        wod_pivot = wod_pivot.reindex(columns=day_order).fillna(0)
        fig_wod, ax_wod = plt.subplots(figsize=(12, 14))
        sns.heatmap(wod_pivot, annot=True, cmap="RdYlGn", center=0, fmt=".2f", ax=ax_wod, linewidths=0.5)
        st.pyplot(fig_wod)
        best_week = wod_pivot.mean(axis=1).idxmax()
        st.info(f"💡 Historically, **Week {best_week}** has been the most bullish.")
    # --- MCP AI STRATEGY INSIGHTS ---
    st.divider()
    st.subheader("🔮 MCP AI: Seasonality Strategy Recommendations")
    
    # Calculate daily performance metrics for AI analysis
    day_stats = f_data.groupby('Day')['Daily_Return'].agg(['mean', 'std', 'count']).reindex(day_order)
    day_stats['Win_Rate'] = f_data.groupby('Day').apply(lambda x: (x['Daily_Return'] > 0).mean()) * 100
    
    # Logic for finding the "Best" strategies
    # We look for high mean returns (Bullish), low mean (Bearish), and high Std Dev (Volatility)
    best_bull = day_stats['mean'].idxmax()
    best_bear = day_stats['mean'].idxmin()
    best_vol = day_stats['std'].idxmax()

    c_ai1, c_ai2, c_ai3 = st.columns(3)

    with c_ai1:
        st.success(f"📈 **Top Bullish Day: {best_bull}**")
        st.write(f"Average Return: **{day_stats.loc[best_bull, 'mean']*100:.3f}%**")
        st.markdown(f"""
        **Recommended Strategies:**
        1. **Bull Put Spread**: High {day_stats.loc[best_bull, 'Win_Rate']:.1f}% Win Prob.
        2. **Buy Call**: Best for trend following on {best_bull}.
        3. **Sell Put**: Capitalize on the positive drift.
        """)

    with c_ai2:
        st.error(f"📉 **Top Bearish Day: {best_bear}**")
        st.write(f"Average Return: **{day_stats.loc[best_bear, 'mean']*100:.3f}%**")
        st.markdown(f"""
        **Recommended Strategies:**
        1. **Bear Call Spread**: Protects against minor bounces.
        2. **Buy Put**: High gamma potential on {best_bear}.
        3. **Sell Call**: Effective if the day shows consistent weakness.
        """)

    with c_ai3:
        st.warning(f"⚡ **Top Volatility Day: {best_vol}**")
        st.write(f"Std Dev (Risk): **{day_stats.loc[best_vol, 'std']*100:.3f}%**")
        st.markdown(f"""
        **Recommended Strategies:**
        1. **Long Straddle**: Exploits wide price swings on {best_vol}.
        2. **Iron Condor**: Only if premiums are over-inflated.
        3. **Long Strangle**: Low-cost bet on a major breakout.
        """)

with tab4:
    st.header(f"🔬 Micro-Insights: Last 8 Weeks")
    recent_8_weeks = trade_results.tail(8).copy()
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(px.line(recent_8_weeks, x="Date", y="Exit_Price", title="Price Momentum", markers=True), width='stretch')
    with col_b:
        st.plotly_chart(px.bar(recent_8_weeks, x="Date", y="PnL", color="PnL", title=f"Strategy PnL: {choice}", color_continuous_scale='RdYlGn'), width='stretch')
    
    st.divider()
    st.write("**Full Weekly PnL Matrix**")
    weekly_matrix = trade_results.pivot_table(index='Year', columns='Week', values='PnL', aggfunc='mean')
    st.dataframe(weekly_matrix.style.map(lambda v: f'background-color: {"#c7f9cc" if v > 0 else "#ffccd5"}' if pd.notna(v) else '').format("{:.0f}"), width='stretch')

with tab5:
    st.header("🔮 MCP Command Center")
    recent = data.tail(500)
    mu, sigma, last_p = recent['Daily_Return'].mean(), recent['Daily_Return'].std(), data['Close'].iloc[-1]
    sims, days = 2000, 5
    results = np.zeros((days, sims))
    for s in range(sims):
        p = [last_p]
        for d in range(days): p.append(p[-1] * (1 + np.random.normal(mu, sigma)))
        results[:, s] = p[1:]

    # Visuals
    upper_95 = np.percentile(results, 95, axis=1)
    lower_5 = np.percentile(results, 5, axis=1)
    prob_up = (results[-1, :] > last_p).mean() * 100
    
    fig_f = go.Figure()
    fig_f.add_trace(go.Scatter(x=["Mon", "Tue", "Wed", "Thu", "Fri"], y=upper_95, name="Ceiling", line=dict(dash='dash', color='green')))
    fig_f.add_trace(go.Scatter(x=["Mon", "Tue", "Wed", "Thu", "Fri"], y=lower_5, name="Floor", fill='tonexty', line=dict(dash='dash', color='red')))
    fig_f.add_trace(go.Scatter(x=["Mon", "Tue", "Wed", "Thu", "Fri"], y=np.median(results, axis=1), name="Expected", line=dict(width=4, color='white')))
    st.plotly_chart(fig_f, width='stretch')

    # Strategy Calculations
    atm_strike = int(round(last_p / 50) * 50)
    bull_strike = int(round(upper_95[-1] / 50) * 50)
    bear_strike = int(round(lower_5[-1] / 50) * 50)

    st.subheader(f"🚀 Best Action (Prob Up: {prob_up:.1f}%)")
    ac1, ac2 = st.columns(2)
    with ac1:
        if prob_up > 55:
            st.success(f"**Bullish:** Buy {atm_strike} CE / Sell {bull_strike} CE")
        elif prob_up < 45:
            st.error(f"**Bearish:** Buy {atm_strike} PE / Sell {bear_strike} PE")
        else:
            st.warning("**Neutral:** Iron Condor (Sell {bear_strike} PE & {bull_strike} CE)")
    with ac2:
        st.metric("Support Floor", bear_strike)
        st.metric("Resistance Ceiling", bull_strike)

with tab6:
    st.header("⚡ Tactical Strategy Lab: Performance & Win Rate")
    
    # --- 1. GLOBAL FILTERS ---
    all_years = sorted(data['Year'].unique(), reverse=True)
    sel_years = st.multiselect("Select Backtest Years", options=all_years, default=all_years[:2])

    t1, t2, t3 = st.columns(3)
    with t1:
        sel_day = st.selectbox("Execution Day", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'], index=1)
        st_action = st.selectbox("Strategy", ["Sell Call", "Sell Put", "Buy Call", "Buy Put"])
        vix_lim = st.slider("VIX Filter (Skip if VIX > X)", 10.0, 30.0, 19.0)
    
    with t2:
        hedge_pts = st.number_input("Hedge Distance (0 = Naked)", value=150, step=50)
        lot_size = 65  # NSE Jan 2026 Revision
        num_lots = st.number_input("Number of Lots", value=1, min_value=1)
        
    with t3:
        sl_pct = st.slider("Hard Stop Loss (%)", 0.1, 3.0, 1.2)
        # Margin Logic (NSE 2026)
        cap_hedged = 42000
        cap_naked = 135000
        margin_per_lot = cap_naked if (hedge_pts == 0 or "Buy" in st_action) else cap_hedged
        initial_capital = margin_per_lot * num_lots

    # --- 2. THE ENGINE ---
    df_lab = data[data['Year'].isin(sel_years)].copy()
    df_lab['VIX_Proxy'] = df_lab['Vol'] * 100 
    df_lab = df_lab.sort_values('Date')

    def run_calculation(row, target_day):
        if row['Day'] != target_day or row['VIX_Proxy'] > vix_lim:
            return None, "Skip"
        
        move = row['Close'] - row['Open']
        move_pct = (abs(move) / row['Open']) * 100
        s_prem = row['Open'] * 0.005 
        b_prem = row['Open'] * 0.001 if hedge_pts > 0 else 0
        
        if move_pct > sl_pct:
            return -(row['Open'] * (sl_pct/100)), "SL Hit"
        
        if st_action == "Sell Call":
            p = (s_prem - b_prem) - (max(0, move) if hedge_pts == 0 else min(max(0, move), hedge_pts))
        elif st_action == "Sell Put":
            p = (s_prem - b_prem) - (max(0, -move) if hedge_pts == 0 else min(max(0, -move), hedge_pts))
        elif st_action == "Buy Call":
            p = max(0, move) - s_prem
        elif st_action == "Buy Put":
            p = max(0, -move) - s_prem
        return p, "Success"

    # --- 3. DATA PROCESSING ---
    results = df_lab.apply(lambda x: run_calculation(x, sel_day), axis=1)
    df_lab['Pts'], df_lab['Status'] = zip(*results)
    
    # ISOLATE EXECUTED TRADES ONLY (Fixes Win Rate & Sharpe)
    exec_df = df_lab[df_lab['Status'] != "Skip"].copy()

    if not exec_df.empty:
        exec_df['PnL_Cash'] = exec_df['Pts'] * lot_size * num_lots
        exec_df['Running_Cap'] = initial_capital + exec_df['PnL_Cash'].cumsum()
        
        # Performance Metrics
        total_pnl = exec_df['PnL_Cash'].sum()
        win_rate = (len(exec_df[exec_df['PnL_Cash'] > 0]) / len(exec_df)) * 100
        sharpe = (exec_df['PnL_Cash'].mean() / exec_df['PnL_Cash'].std()) * np.sqrt(52) if exec_df['PnL_Cash'].std() != 0 else 0
        cagr = (((exec_df['Running_Cap'].iloc[-1]) / initial_capital) ** (1/len(sel_years)) - 1) * 100
        
        # Drawdown
        exec_df['Peak'] = exec_df['Running_Cap'].cummax()
        exec_df['Drawdown'] = (exec_df['Running_Cap'] - exec_df['Peak']) / exec_df['Peak'] * 100

        # --- 4. TOP LEVEL METRICS ---
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

        # --- 5. EQUITY CURVE (MAIN CHART) ---
        st.subheader(f"📈 Equity Curve: {st_action} on {sel_day}s")
        fig_equity = px.area(exec_df, x='Date', y='Running_Cap', 
                             title="Portfolio Growth Over Time",
                             color_discrete_sequence=['#00CC96'])
        st.plotly_chart(fig_equity, width='stretch')

        # --- 6. HEATMAPS & DAY COMPARISON ---
        st.subheader("🗓️ Performance Insights")
        h1, h2 = st.columns(2)
        with h1:
            pivot_pnl = exec_df.groupby(['Year', 'Month'])['PnL_Cash'].sum().unstack().fillna(0)
            st.plotly_chart(px.imshow(pivot_pnl, text_auto=True, title="Monthly Profit Heatmap", color_continuous_scale='RdYlGn'), width='stretch')
        with h2:
            day_comp = []
            for d in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
                d_pnl = df_lab[df_lab['Day'] == d].apply(lambda x: run_calculation(x, d)[0], axis=1).fillna(0).sum() * lot_size * num_lots
                day_comp.append({"Day": d, "Total PnL": d_pnl})
            st.plotly_chart(px.bar(pd.DataFrame(day_comp), x='Day', y='Total PnL', title="Strategy Comparison by Day", color='Total PnL', color_continuous_scale='RdYlGn'), width='stretch')

        # --- 7. THE TRADE LOG ---
        st.subheader(f"📋 Trade Log ({sel_day} Execution Only)")
        st.dataframe(
            exec_df[['Date', 'Status', 'VIX_Proxy', 'PnL_Cash', 'Running_Cap', 'Drawdown']].sort_values('Date', ascending=False),
            column_config={
                "PnL_Cash": st.column_config.NumberColumn("PnL (₹)", format="₹%.2f"),
                "Running_Cap": st.column_config.NumberColumn("Account Balance", format="₹%.0f"),
                "Drawdown": st.column_config.NumberColumn("Drawdown %", format="%.2f%%")
            },
            width='stretch'
        )
    else:
        st.warning("No trades match these filters. Adjust VIX or Day selection.")

with tab7:
    st.header("📅 Weekly Strategy Game Plan")
    st.markdown("Institutional Suite: Compounding, MTM tracking, and Post-Tax Net Profit analysis.")

    # --- 1. SETTINGS & RISK ---
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

    # --- 2. THE WEEKLY PLAYBOOK ---
    st.subheader("🛠️ Step 1: Define Your Weekly Playbook")
    day_cols = st.columns(5)
    days_list = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    playbook = {}
    for i, day in enumerate(days_list):
        with day_cols[i]:
            playbook[day] = st.radio(f"Action: {day}", ["No Trade", "Sell Call", "Sell Put", "Buy Call", "Buy Put"], index=0, key=f"p_fin_{day}")

    # --- 3. BACKTEST ENGINE ---
    def run_professional_backtest(row):
        current_day = row['Day']
        if current_day not in playbook or playbook[current_day] == "No Trade":
            return 0, "Skip", 0, 0
            
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
            pnl, m_h, m_l = max(0, move)-s_prem, max(0, h_move)-s_prem, max(0, l_move)-s_prem
        elif strat == "Buy Put":
            pnl, m_h, m_l = max(0, -move)-s_prem, max(0, -l_move)-s_prem, max(0, -h_move)-s_prem

        if (abs(move)/row['Open']*100) > sl_val_t7:
            pnl, status = -(row['Open'] * (sl_val_t7/100)), "SL Hit"
        else: status = "Success"
        return pnl, status, m_l, m_h

    # --- 4. EXECUTION & COMPOUNDING ---
    df_t7 = data[data['Year'].isin(sel_yrs_t7)].copy().sort_values('Date')
    active_days = [d for d, s in playbook.items() if s != "No Trade"]
    
    if active_days:
        res = df_t7.apply(run_professional_backtest, axis=1)
        df_t7['Pts'], df_t7['Status'], df_t7['MTM_L_Pts'], df_t7['MTM_H_Pts'] = zip(*res)
        exec_df = df_t7[df_t7['Status'].isin(["Success", "SL Hit"])].copy()
        
        if not exec_df.empty:
            current_cap = capital_per_lot
            results = []
            for i, row in exec_df.iterrows():
                lots = max(1, int(current_cap // capital_per_lot))
                gross_pnl = row['Pts'] * lot_size_base * lots
                charges = (2 * brokerage_per_order) + (row['Open'] * lot_size_base * lots * tax_stt_pct)
                net_pnl = gross_pnl - charges
                current_cap += net_pnl
                results.append({
                    'Net_PnL': net_pnl,
                    'Gross_PnL': gross_pnl,
                    'Charges': charges,
                    'Running_Cap': current_cap,
                    'Lots': lots,
                    'MTM_L': row['MTM_L_Pts'] * lot_size_base * lots,
                    'MTM_H': row['MTM_H_Pts'] * lot_size_base * lots
                })
            
            res_df = pd.DataFrame(results)
            exec_df = pd.concat([exec_df.reset_index(drop=True), res_df], axis=1)

        # --- 5. PERFORMANCE CALCULATIONS ---
        net_wins = exec_df[exec_df['Net_PnL'] > 0]['Net_PnL']
        net_losses = exec_df[exec_df['Net_PnL'] <= 0]['Net_PnL']
        total_net = exec_df['Net_PnL'].sum()
        
        # Drawdown logic
        exec_df['Peak'] = exec_df['Running_Cap'].cummax()
        exec_df['Drawdown_Pct'] = (exec_df['Running_Cap'] - exec_df['Peak']) / exec_df['Peak'] * 100
        
        # Streak Logic
        is_win = (exec_df['Net_PnL'] > 0).astype(int)
        streaks = is_win.groupby((is_win != is_win.shift()).cumsum()).cumcount() + 1
        max_w_s = streaks[is_win == 1].max() if any(is_win == 1) else 0
        max_l_s = streaks[is_win == 0].max() if any(is_win == 0) else 0

        # --- 6. DISPLAY METRICS ---
        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Net Profit", f"₹{total_net:,.0f}")
        m2.metric("CAGR (Net)", f"{(((exec_df['Running_Cap'].iloc[-1]/capital_per_lot)**(1/max(1,len(sel_yrs_t7))))-1)*100:.2f}%")
        m3.metric("Win Rate", f"{(len(net_wins)/len(exec_df))*100:.1f}%")
        m4.metric("Sharpe Ratio", f"{(exec_df['Net_PnL'].mean()/exec_df['Net_PnL'].std())*np.sqrt(252):.2f}" if exec_df['Net_PnL'].std() != 0 else "0.00")

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Max Profit Day", f"₹{exec_df['Net_PnL'].max():,.0f}")
        m6.metric("Max Loss Day", f"₹{exec_df['Net_PnL'].min():,.0f}")
        m7.metric("Max Intraday MTM Loss", f"₹{exec_df['MTM_L'].min():,.0f}")
        m8.metric("Max Intraday MTM Gain", f"₹{exec_df['MTM_H'].max():,.0f}")

        m9, m10, m11, m12 = st.columns(4)
        m9.metric("Max Drawdown (%)", f"{exec_df['Drawdown_Pct'].min():.2f}%")
        m10.metric("Recovery Factor", f"{abs(total_net / (exec_df['Peak'] - exec_df['Running_Cap']).max()):.2f}" if not (exec_df['Peak'] - exec_df['Running_Cap']).max() == 0 else "0.00")
        m11.metric("Max Win Streak", f"{max_w_s} Days")
        m12.metric("Max Loss Streak", f"{max_l_s} Days")

        # --- 7. YEAR-OVER-YEAR TABLE ---
        st.subheader("📈 Institutional Yearly Performance")
        yoy = exec_df.groupby('Year').agg(
            Net_Profit=('Net_PnL', 'sum'),
            Total_Trades=('Net_PnL', 'count'),
            Profit_Trades=('Net_PnL', lambda x: (x > 0).sum()),
            Win_Rate=('Net_PnL', lambda x: (x > 0).mean() * 100),
        )
        # Add Yearly Drawdown
        yoy['Max_DD_%'] = exec_df.groupby('Year').apply(lambda x: ((x['Running_Cap'] - x['Running_Cap'].cummax()) / x['Running_Cap'].cummax()).min() * 100).values

        st.table(yoy.style.format({
            "Net_Profit": "₹{:,.0f}", "Total_Trades": "{:,.0f}", 
            "Profit_Trades": "{:,.0f}", "Win_Rate": "{:.1f}%", "Max_DD_%": "{:.2f}%"
        }))

        # --- 8. MONTHLY HEATMAP ---
        st.write("**Monthly Breakdown (Balanced at ₹0)**")
        mom = exec_df.groupby(['Year', 'MonthName'])['Net_PnL'].sum().unstack().fillna(0)
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        mom = mom.reindex(columns=[m for m in month_order if m in mom.columns])
        limit = max(abs(mom.values.min()), abs(mom.values.max()))
        st.plotly_chart(px.imshow(mom, text_auto=",.0f", color_continuous_scale='RdYlGn', range_color=[-limit, limit], aspect="auto").update_layout(coloraxis_showscale=False), width='stretch')

        st.area_chart(exec_df.set_index('Date')['Running_Cap'])
        st.subheader("📋 Detailed Trade Log")
        st.dataframe(
                exec_df[['Date', 'Day', 'Open', 'Close', 'Lots', 'Net_PnL', 'Gross_PnL', 'Charges', 'MTM_L', 'MTM_H', 'Running_Cap', 'Status']].sort_values('Date', ascending=False), 
                column_config={
                    "Open": st.column_config.NumberColumn("Nifty Open", format="%.2f"),
                    "Close": st.column_config.NumberColumn("Nifty Close", format="%.2f"),
                    "Net_PnL": st.column_config.NumberColumn("Net PnL", format="₹%.2f"),
                    "Gross_PnL": st.column_config.NumberColumn("Gross PnL", format="₹%.2f"),
                    "Charges": st.column_config.NumberColumn("Charges", format="₹%.2f"),
                    "Running_Cap": st.column_config.NumberColumn("Account Balance", format="₹%.0f"),
                    "MTM_L": st.column_config.NumberColumn("Max MTM Loss", format="₹%.0f"),
                    "MTM_H": st.column_config.NumberColumn("Max MTM Gain", format="₹%.0f")
                },
                width='stretch'
            )
    else:
        st.info("💡 **MCP Note:** Complete your Strategy Playbook in the 'Deep-Dive Calendar' tab to see AI Performance Insights here.")
