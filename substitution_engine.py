"""
substitution_engine.py
──────────────────────
Smart Substitution Engine — finds a "Mathematical Twin":
  A stock in the same sector with similar growth profile
  but lower volatility and lower correlation to the portfolio.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import List, Dict, Any, Optional


# Default candidate universe (extend as needed)
DEFAULT_UNIVERSE = {
    "Technology":     ["MSFT", "GOOGL", "AMD", "INTC", "QCOM", "CSCO", "IBM", "HPQ", "STX"],
    "Information Technology": ["INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", "LTIM.NS"],
    "Energy":         ["XOM", "CVX", "COP", "SLB", "HAL", "MRO", "OXY", "PSX"],
    "Financials":     ["JPM", "BAC", "WFC", "GS", "MS", "AXP", "BLK", "COF"],
    "Financial Services": ["HDFCBANK.NS", "ICICIBANK.NS", "AXISBANK.NS", "KOTAKBANK.NS"],
    "Healthcare":     ["JNJ", "PFE", "ABT", "MRK", "BMY", "AMGN", "GILD", "REGN"],
    "Consumer Discretionary": ["AMZN", "MCD", "NKE", "TGT", "HD", "LOW", "SBUX"],
    "Consumer Staples": ["PG", "KO", "PEP", "WMT", "COST", "MDLZ", "CL"],
    "Industrials":    ["HON", "GE", "MMM", "CAT", "UPS", "FDX", "DE", "LMT"],
    "Materials":      ["LIN", "SHW", "ECL", "APD", "NEM", "FCX", "NUE", "VMC"],
}


class SmartSubstitutionEngine:
    """
    Finds a lower-risk mathematical twin for a given high-risk ticker.

    Parameters
    ----------
    portfolio_tickers : list  – Current portfolio holdings
    period            : str   – Historical data period for calculations
    """

    def __init__(self, portfolio_tickers: List[str], period: str = "1y"):
        self.portfolio = portfolio_tickers
        self.period    = period

    def find_twin(
        self,
        high_risk_ticker: str,
        candidate_universe: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Search for the best Mathematical Twin.

        Scoring:
          60% – Volatility reduction vs target
          20% – Low correlation to existing portfolio
          20% – Similar annual return profile (growth match)
        """
        import yfinance as yf

        # Get sector for the high-risk ticker
        try:
            sector = yf.Ticker(high_risk_ticker).info.get("sector", "Unknown")
        except Exception:
            sector = "Unknown"

        # Build candidate list
        if candidate_universe:
            candidates = [c for c in candidate_universe if c != high_risk_ticker]
        else:
            candidates = DEFAULT_UNIVERSE.get(sector, list({
                v for vals in DEFAULT_UNIVERSE.values() for v in vals
            }))
            candidates = [c for c in candidates if c != high_risk_ticker]

        # Download all prices in one shot
        all_tickers = list(set(self.portfolio + [high_risk_ticker] + candidates))
        try:
            raw = yf.download(
                all_tickers, period=self.period,
                auto_adjust=True, progress=False,
            )
            if len(all_tickers) == 1:
                prices = pd.DataFrame({all_tickers[0]: raw["Close"]})
            else:
                prices = raw["Close"] if "Close" in raw.columns else raw.xs("Close", axis=1, level=0)
        except Exception as e:
            return {"error": f"Data download failed: {e}"}

        prices  = prices.ffill().bfill().dropna(how="all")
        returns = np.log(prices / prices.shift(1)).dropna()

        # Reference metrics for the high-risk ticker
        if high_risk_ticker not in returns.columns:
            return {"error": f"{high_risk_ticker} not found in downloaded data"}

        ref_vol  = returns[high_risk_ticker].std() * np.sqrt(252)
        ref_ret  = (prices[high_risk_ticker].iloc[-1] / prices[high_risk_ticker].iloc[0]) - 1

        # Portfolio average returns (for correlation baseline)
        port_cols = [t for t in self.portfolio if t in returns.columns and t != high_risk_ticker]
        port_ret_series = returns[port_cols].mean(axis=1) if port_cols else returns[high_risk_ticker]

        scored = []
        for c in candidates:
            if c not in returns.columns:
                continue
            c_vol  = returns[c].std() * np.sqrt(252)
            c_ret  = (prices[c].iloc[-1] / prices[c].iloc[0]) - 1

            # Skip if candidate has higher volatility than target
            if c_vol >= ref_vol:
                continue

            vol_reduction  = (ref_vol - c_vol) / ref_vol   # 0→1, higher=better
            ret_similarity = 1 - min(abs(c_ret - ref_ret), 0.5) / 0.5   # penalise huge divergence
            corr_to_port   = abs(returns[c].corr(port_ret_series))
            corr_score     = 1 - corr_to_port   # lower correlation = better

            total_score = 0.60 * vol_reduction + 0.20 * corr_score + 0.20 * ret_similarity
            scored.append({
                "ticker":          c,
                "score":           round(total_score, 4),
                "volatility":      round(c_vol, 4),
                "annual_return":   round(c_ret, 4),
                "corr_to_port":    round(corr_to_port, 4),
                "vol_reduction":   round(vol_reduction * 100, 1),
            })

        if not scored:
            return {
                "high_risk_ticker": high_risk_ticker,
                "sector":           sector,
                "result":           "No suitable twin found in candidate universe.",
                "candidates_checked": len(candidates),
            }

        scored.sort(key=lambda x: -x["score"])
        best = scored[0]

        return {
            "high_risk_ticker":    high_risk_ticker,
            "sector":              sector,
            "ref_volatility":      round(ref_vol, 4),
            "ref_annual_return":   round(ref_ret, 4),
            "best_twin":           best,
            "top_3_candidates":    scored[:3],
            "interpretation": (
                f"'{best['ticker']}' identified as the closest Mathematical Twin for '{high_risk_ticker}'. "
                f"It offers {best['vol_reduction']:.1f}% lower annualised volatility "
                f"with a portfolio correlation of {best['corr_to_port']:.2f}, "
                f"suggesting reduced concentration risk if substituted."
            ),
        }
