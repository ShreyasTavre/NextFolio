"""
streamlit_app.py
────────────────
Financial Decision Intelligence Engine — Streamlit Dashboard
Run: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from typing import List, Optional

# ── Local modules ──────────────────────────────────────────────────────────────
from data_layer import MarketDataPipeline
from quant_engine import QuantEngine
from portfolio_health import PortfolioHealthAnalyzer
from scenario_simulator import MonteCarloSimulator
from sentiment_xai import SentimentAnalyzer, XAILayer

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Intelligence Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
  .metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #0f3460;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 0.4rem 0;
  }
  .health-score {
    font-size: 3rem;
    font-weight: 700;
    color: #00d4ff;
  }
  .risk-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 2px;
  }
  .risk-high   { background: #ff4b4b33; color: #ff4b4b; border: 1px solid #ff4b4b; }
  .risk-medium { background: #ffa50033; color: #ffa500; border: 1px solid #ffa500; }
  .risk-low    { background: #00d46433; color: #00d464; border: 1px solid #00d464; }
  .stButton > button {
    background: linear-gradient(135deg, #0f3460, #00d4ff22);
    color: #00d4ff;
    border: 1px solid #00d4ff;
    border-radius: 8px;
  }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧠 Portfolio Intelligence Engine")
    st.markdown("---")

    default_tickers = "RELIANCE.NS, TCS.NS, AAPL, NVDA, MSFT"
    ticker_input = st.text_area(
        "📋 Portfolio Tickers (comma-separated)",
        value=default_tickers,
        height=100,
    )
    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

    period = st.selectbox(
        "📅 Historical Period",
        ["6mo", "1y", "2y", "5y"],
        index=1,
    )

    pv = st.number_input(
        "💰 Portfolio Value (USD/INR)",
        value=1_000_000,
        step=100_000,
        format="%d",
    )

    equal_weight = st.checkbox("Equal-weight portfolio", value=True)
    weights = None
    if not equal_weight:
        st.write("Enter weights (must sum to 1):")
        weights_input = []
        for t in tickers:
            w = st.number_input(f"  {t}", min_value=0.0, max_value=1.0, value=round(1/len(tickers), 3), step=0.01)
            weights_input.append(w)
        total_w = sum(weights_input)
        if abs(total_w - 1.0) > 0.01:
            st.warning(f"Weights sum to {total_w:.2f}, not 1.0. Will normalise.")
        weights = weights_input

    st.markdown("---")
    run_btn = st.button("🚀 Run Analysis", use_container_width=True)

    st.markdown("---")
    st.markdown("#### ⚡ Stress Test")
    stress_sector = st.selectbox(
        "Shock Sector",
        ["Technology", "Energy", "Financials", "Healthcare", "Consumer Discretionary",
         "Consumer Staples", "Industrials", "Materials", "Information Technology"],
    )
    stress_pct = st.slider("Shock Magnitude (%)", min_value=-50, max_value=-1, value=-10)
    stress_btn = st.button("🧪 Run Stress Test", use_container_width=True)


# ── Main content ───────────────────────────────────────────────────────────────

st.title("📊 Financial Decision Intelligence Engine")
st.caption("Portfolio Intelligence & Risk Simulation Agent · Decision Support System · Not Financial Advice")

if not run_btn and not stress_btn:
    st.info("👈 Configure your portfolio in the sidebar and click **Run Analysis** to begin.")
    st.stop()


# ── Data Loading ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_data(tickers, period):
    pipeline = MarketDataPipeline(tickers, period)
    prices   = pipeline.get_prices()
    info     = pipeline.get_fundamentals()
    news_all = {t: pipeline.get_news(t, 8) for t in tickers}
    return prices, info, news_all

with st.spinner("📡 Fetching market data..."):
    try:
        prices, info, news_all = load_data(tuple(tickers), period)
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        st.stop()

if prices.empty:
    st.error("No price data returned. Check your ticker symbols.")
    st.stop()

valid_tickers = list(prices.columns)
quant    = QuantEngine(prices)
analyzer = PortfolioHealthAnalyzer(prices, info, weights)


# ── STRESS TEST MODE ───────────────────────────────────────────────────────────

if stress_btn:
    st.markdown("---")
    st.subheader(f"🧪 Stress Test: '{stress_sector}' sector drops {abs(stress_pct)}%")

    with st.spinner("Running Monte Carlo simulation (10,000 paths)..."):
        ticker_sectors = {t: info.get(t, {}).get("sector", "Unknown") for t in valid_tickers}
        simulator = MonteCarloSimulator(prices, weights, portfolio_value=pv)
        result    = simulator.run(
            sector=stress_sector,
            shock_pct=stress_pct / 100,
            simulations=10_000,
            ticker_sectors=ticker_sectors,
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean P&L", f"${result['mean_pnl']:,.0f}")
    c2.metric("Worst Case (5%)", f"${result['worst_case_pnl_5pct']:,.0f}")
    c3.metric("VaR-95", f"${result['value_at_risk_95']:,.0f}")
    c4.metric("P(Loss > 10%)", f"{result['prob_loss_gt_10pct']*100:.1f}%")

    st.caption(f"ℹ️ {result['scenario_description']}")
    st.caption(f"Affected tickers: {', '.join(result['affected_tickers']) or 'None in portfolio'}")

    # Distribution histogram
    dist = result["distribution_sample"]
    fig_hist = px.histogram(
        x=dist,
        nbins=40,
        title="Monte Carlo P&L Distribution",
        labels={"x": "P&L (Currency Units)"},
        color_discrete_sequence=["#00d4ff"],
        template="plotly_dark",
    )
    fig_hist.add_vline(x=result["value_at_risk_95"], line_dash="dash", line_color="red",
                       annotation_text="VaR-95")
    fig_hist.add_vline(x=result["mean_pnl"], line_dash="dot", line_color="lime",
                       annotation_text="Mean")
    st.plotly_chart(fig_hist, use_container_width=True)

    # Sensitivity table
    st.subheader("Sensitivity Table")
    with st.spinner("Computing sensitivity table..."):
        sens_df = simulator.sensitivity_table(stress_sector)
    st.dataframe(sens_df, use_container_width=True)

    st.markdown("---")


# ── MAIN ANALYSIS ──────────────────────────────────────────────────────────────

if run_btn:
    # ── TAB Layout ─────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Health Dashboard",
        "🔥 Correlation Analysis",
        "📰 Hype Meter",
        "🔄 Substitution Engine",
        "🤖 AI Insights",
    ])

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 1: Health Dashboard
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab1:
        with st.spinner("Computing portfolio health..."):
            health = analyzer.health_score()
            sector_exp = analyzer.sector_exposure()

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Portfolio Health Score")
            score = health["overall_score"]
            grade = health["grade"]
            color = "#00d464" if score >= 65 else "#ffa500" if score >= 50 else "#ff4b4b"
            st.markdown(
                f'<div style="text-align:center; font-size:5rem; color:{color}; font-weight:700">'
                f'{score}</div>'
                f'<div style="text-align:center; font-size:1.5rem; color:{color}">Grade: {grade}</div>',
                unsafe_allow_html=True,
            )
            st.caption(health["interpretation"])
            st.metric("Portfolio Sharpe",  f"{health['portfolio_sharpe']:.2f}")
            st.metric("Avg Correlation",   f"{health['avg_correlation']:.2f}")
            st.metric("Sector HHI",        f"{health['hhi_sector']:.3f}")

        with col2:
            st.subheader("Sector Allocation")
            fig_pie = px.pie(
                values=list(sector_exp.values()),
                names=list(sector_exp.keys()),
                hole=0.4,
                template="plotly_dark",
                color_discrete_sequence=px.colors.sequential.Plasma_r,
            )
            fig_pie.update_traces(textposition="outside", textinfo="percent+label")
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("---")
        # Risk-Return Scatter
        st.subheader("Risk-Return Scatter")
        rr = quant.risk_return_table()
        rr["Sharpe"] = rr["Sharpe"].abs()  # Ensure positive values for bubble size
        fig_scatter = px.scatter(
            rr.reset_index(),
            x="Volatility", y="Annual_Return",
            size="Sharpe", color="RSI",
            text="Ticker",
            title="Risk vs Return (bubble size = |Sharpe Ratio|)",
            template="plotly_dark",
            color_continuous_scale="RdYlGn",
        )
        fig_scatter.update_traces(textposition="top center")
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Metrics table
        st.subheader("Individual Stock Metrics")
        st.dataframe(rr, use_container_width=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 2: Correlation
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab2:
        corr = quant.correlation_matrix()
        flags = analyzer.hidden_correlation_flags(threshold=0.85)

        # Heatmap
        fig_heat = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            title="Pearson Correlation Matrix (Daily Log Returns)",
            template="plotly_dark",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # Flags
        sev  = flags["severity"]
        sev_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(sev, "⚪")
        st.markdown(f"### {sev_color} Concentration Risk: **{sev}**")
        st.info(flags["message"])

        if flags["high_pairs"]:
            pairs_df = pd.DataFrame(flags["high_pairs"], columns=["Ticker A", "Ticker B", "Correlation r"])
            st.dataframe(pairs_df, use_container_width=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 3: Hype Meter
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab3:
        st.subheader("Hype vs Fundamentals Meter")
        sa = SentimentAnalyzer()
        hype_data = []

        for ticker in valid_tickers:
            score = sa.score(news_all.get(ticker, []))
            label = sa.label(score)
            pe    = info.get(ticker, {}).get("trailingPE")
            fpe   = info.get(ticker, {}).get("forwardPE")
            name  = info.get(ticker, {}).get("shortName", ticker)

            # Hype flag logic
            if score and score > 0.5 and pe and pe > 30:
                flag = "🚨 Hype Overload"
            elif score and score > 0.3:
                flag = "⚠️ Elevated Sentiment"
            elif score and score < -0.4:
                flag = "📉 Extreme Fear"
            else:
                flag = "✅ Balanced"

            hype_data.append({
                "Ticker":        ticker,
                "Name":          name,
                "Sentiment":     score if score else 0.0,
                "Sentiment Label": label,
                "Trailing P/E":  pe,
                "Forward P/E":   fpe,
                "Flag":          flag,
            })

        hype_df = pd.DataFrame(hype_data)

        # Gauge charts
        cols = st.columns(min(len(valid_tickers), 4))
        for idx, row in hype_df.iterrows():
            c = cols[idx % len(cols)]
            with c:
                sent_val = (row["Sentiment"] + 1) / 2 * 100  # map [-1,1] → [0,100]
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=sent_val,
                    title={"text": row["Ticker"]},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar":  {"color": "#00d4ff"},
                        "steps": [
                            {"range": [0, 30], "color": "rgba(255, 75, 75, 0.2)"},
                            {"range": [30, 70], "color": "rgba(255, 165, 0, 0.2)"},
                            {"range": [70, 100], "color": "rgba(0, 212, 100, 0.2)"},
                        ],
                        "threshold": {"line": {"color": "white", "width": 2}, "value": sent_val},
                    },
                ))
                fig_g.update_layout(height=220, margin=dict(t=40, b=10, l=10, r=10), template="plotly_dark")
                st.plotly_chart(fig_g, use_container_width=True)
                st.caption(f"{row['Flag']} · P/E: {row['Trailing P/E']}")

        st.markdown("---")
        st.dataframe(hype_df, use_container_width=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 4: Substitution Engine
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab4:
        st.subheader("🔄 Smart Substitution Engine")
        st.info("Finds a 'Mathematical Twin' — same sector, similar growth, lower volatility.")

        rr_table = quant.risk_return_table()
        if not rr_table.empty:
            # Identify highest-volatility ticker
            highest_vol_ticker = rr_table["Volatility"].idxmax()
            selected_ticker = st.selectbox(
                "Select high-risk ticker to substitute",
                options=valid_tickers,
                index=valid_tickers.index(highest_vol_ticker) if highest_vol_ticker in valid_tickers else 0,
            )

            if st.button("🔍 Find Mathematical Twin"):
                from substitution_engine import SmartSubstitutionEngine
                with st.spinner("Searching for best mathematical twin..."):
                    engine = SmartSubstitutionEngine(valid_tickers)
                    result = engine.find_twin(selected_ticker)

                if "error" in result:
                    st.error(result["error"])
                elif "result" in result:
                    st.warning(result["result"])
                else:
                    best = result.get("best_twin", {})
                    st.success(result.get("interpretation", ""))

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown(f"#### ❌ Current: {selected_ticker}")
                        st.metric("Annual Volatility", f"{result['ref_volatility']*100:.1f}%")
                        st.metric("Annual Return",     f"{result['ref_annual_return']*100:.1f}%")

                    with col_b:
                        st.markdown(f"#### ✅ Twin: {best.get('ticker','')}")
                        st.metric("Annual Volatility", f"{best.get('volatility',0)*100:.1f}%",
                                  delta=f"-{best.get('vol_reduction',0):.1f}%")
                        st.metric("Annual Return",     f"{best.get('annual_return',0)*100:.1f}%")
                        st.metric("Corr to Portfolio", f"{best.get('corr_to_port',0):.2f}")

                    st.markdown("**Top 3 Candidates:**")
                    if result.get("top_3_candidates"):
                        st.dataframe(pd.DataFrame(result["top_3_candidates"]), use_container_width=True)
        else:
            st.warning("Insufficient data to run substitution engine.")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 5: AI Insights
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab5:
        st.subheader("🤖 AI Executive Summary")
        st.caption("Generated by Claude (Anthropic) · Analytical only · Not investment advice")

        health   = analyzer.health_score()
        sector   = analyzer.sector_exposure()
        flags    = analyzer.hidden_correlation_flags(0.85)
        metrics  = {t: {
            "beta":       quant.beta(t),
            "volatility": quant.volatility(t),
            "sharpe":     quant.sharpe_ratio(t),
        } for t in valid_tickers}

        raw_data = {
            "health_score":    health,
            "sector_exposure": sector,
            "hidden_flags":    flags,
            "metrics":         metrics,
        }

        if st.button("🧠 Generate AI Insights"):
            with st.spinner("Generating executive summary via Claude..."):
                xai     = XAILayer()
                summary = xai.generate_summary(raw_data)

            if summary.get("_fallback"):
                st.warning("⚠️ Using rule-based fallback (Claude API key not configured).")

            for key in ["point_1", "point_2", "point_3"]:
                pt  = summary.get(key, {})
                sev = pt.get("severity", "Low")
                icon = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}.get(sev, "⚪")
                with st.expander(f"{icon} {pt.get('title','')}", expanded=True):
                    st.write(pt.get("detail", ""))
                    st.caption(f"Severity: **{sev}**")

            st.markdown("---")
            st.markdown("**Overall Assessment**")
            st.info(summary.get("overall_assessment", ""))

            with st.expander("📄 Raw Risk Data (JSON)"):
                st.json(raw_data)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "⚠️ **Disclaimer**: This tool is a Decision Support System providing objective, quantitative analysis. "
    "It does not constitute financial advice. Always consult a qualified financial advisor before making investment decisions."
)
