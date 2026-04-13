"""
scenario_simulator.py
─────────────────────
Monte Carlo Stress Testing Engine.

Scenario: "What if sector X drops Y%?"
Output:   Distribution of projected portfolio value losses.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any


# Sector Beta sensitivity map (approximate; can be overridden per ticker via yfinance)
SECTOR_SENSITIVITY = {
    "Technology":          1.20,
    "Information Technology": 1.20,
    "Energy":              0.95,
    "Financials":          1.10,
    "Financial Services":  1.10,
    "Healthcare":          0.75,
    "Consumer Discretionary": 1.05,
    "Consumer Staples":    0.60,
    "Industrials":         0.90,
    "Materials":           0.85,
    "Real Estate":         0.80,
    "Communication Services": 1.00,
    "Utilities":           0.55,
    "Unknown":             1.00,
}


class MonteCarloSimulator:
    """
    Monte Carlo stress test for portfolio value under a sector shock.

    Parameters
    ----------
    prices      : pd.DataFrame  – Historical prices (rows=dates, cols=tickers)
    weights     : list[float]   – Portfolio weights (will be normalised)
    portfolio_value : float     – Notional portfolio value in currency units
    """

    def __init__(
        self,
        prices:          pd.DataFrame,
        weights:         Optional[List[float]] = None,
        portfolio_value: float = 1_000_000,
    ):
        self.prices   = prices
        self.tickers  = list(prices.columns)
        n             = len(self.tickers)
        w             = np.array(weights) if weights else np.ones(n)
        self.weights  = w / w.sum()
        self.pv       = portfolio_value
        self.returns  = np.log(prices / prices.shift(1)).dropna()

    # ── Main Runner ───────────────────────────────────────────────────────────

    def run(
        self,
        sector:      str,
        shock_pct:   float,        # e.g. -0.10 for -10%
        simulations: int = 10_000,
        holding_days: int = 21,    # 1 month forward horizon
        ticker_sectors: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation with a sector shock applied.

        Returns
        -------
        dict with:
          - scenario_description
          - mean_loss, median_loss, worst_case (5th pct), best_case (95th pct)
          - var_95, cvar_95
          - distribution (list of 200 sampled outcomes for histogram)
          - affected_tickers
        """
        ticker_sectors = ticker_sectors or {}
        sector_sens    = SECTOR_SENSITIVITY.get(sector, 1.0)

        # Identify tickers affected by the shock
        affected = [
            t for t in self.tickers
            if ticker_sectors.get(t, "Unknown").lower() == sector.lower()
               or sector.lower() in ticker_sectors.get(t, "").lower()
        ]

        # Compute historical covariance matrix (annualised)
        cov_daily = self.returns.cov().values
        means     = self.returns.mean().values

        # Build shock vector: affected tickers absorb a fraction of the sector shock
        # Non-affected tickers absorb spill-over via correlation
        corr  = self.returns.corr().values
        shock = np.zeros(len(self.tickers))
        for i, t in enumerate(self.tickers):
            if t in affected:
                shock[i] = shock_pct * sector_sens
            else:
                # Spill-over: average correlation with affected tickers × 30% of shock
                aff_idxs = [self.tickers.index(a) for a in affected if a in self.tickers]
                if aff_idxs:
                    avg_corr  = np.mean([corr[i, j] for j in aff_idxs])
                    shock[i]  = shock_pct * avg_corr * 0.30

        # Monte Carlo: sample multivariate normal returns + shock
        np.random.seed(42)
        try:
            # Cholesky decomposition for correlated sampling
            L = np.linalg.cholesky(cov_daily + 1e-8 * np.eye(len(cov_daily)))
        except np.linalg.LinAlgError:
            L = np.diag(np.sqrt(np.diag(cov_daily)))

        portfolio_pnl = np.zeros(simulations)
        for _ in range(simulations):
            z           = np.random.standard_normal(len(self.tickers))
            daily_ret   = means + L @ z
            # Apply shock distributed over holding_days (first day gets shock)
            total_ret   = daily_ret * holding_days + shock
            port_ret    = float(self.weights @ total_ret)
            portfolio_pnl[_] = self.pv * port_ret

        # Statistics
        var_95   = float(np.percentile(portfolio_pnl, 5))
        cvar_95  = float(portfolio_pnl[portfolio_pnl <= var_95].mean())
        mean_pnl = float(portfolio_pnl.mean())

        # Downsample for histogram
        sample_idx   = np.random.choice(simulations, 200, replace=False)
        distribution = portfolio_pnl[sample_idx].tolist()

        return {
            "scenario_description": (
                f"Sector '{sector}' drops {abs(shock_pct)*100:.1f}% over {holding_days} trading days. "
                f"Sector sensitivity factor: {sector_sens:.2f}."
            ),
            "affected_tickers":          affected,
            "unaffected_tickers":        [t for t in self.tickers if t not in affected],
            "portfolio_value":           self.pv,
            "simulations":               simulations,
            "mean_pnl":                  round(mean_pnl, 2),
            "median_pnl":                round(float(np.median(portfolio_pnl)), 2),
            "best_case_pnl_95pct":       round(float(np.percentile(portfolio_pnl, 95)), 2),
            "worst_case_pnl_5pct":       round(float(np.percentile(portfolio_pnl, 5)), 2),
            "value_at_risk_95":          round(var_95, 2),
            "conditional_var_95":        round(cvar_95, 2),
            "prob_loss_gt_5pct":         round(float((portfolio_pnl < -0.05 * self.pv).mean()), 4),
            "prob_loss_gt_10pct":        round(float((portfolio_pnl < -0.10 * self.pv).mean()), 4),
            "distribution_sample":       [round(x, 2) for x in distribution],
            "shock_applied":             {t: round(shock[i], 4) for i, t in enumerate(self.tickers)},
        }

    # ── Quick Sensitivity Table ───────────────────────────────────────────────

    def sensitivity_table(
        self,
        sector: str,
        shocks: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """
        Run multiple shock scenarios and return a summary DataFrame.
        shocks defaults to [-5%, -10%, -20%, -30%].
        """
        shocks  = shocks or [-0.05, -0.10, -0.20, -0.30]
        rows    = []
        for s in shocks:
            result = self.run(sector=sector, shock_pct=s, simulations=5_000)
            rows.append({
                "Shock (%)":        f"{s*100:.0f}%",
                "Mean P&L":         result["mean_pnl"],
                "Worst (5%)":       result["worst_case_pnl_5pct"],
                "VaR-95":           result["value_at_risk_95"],
                "CVaR-95":          result["conditional_var_95"],
                "P(Loss>10%)":      f"{result['prob_loss_gt_10pct']*100:.1f}%",
            })
        return pd.DataFrame(rows)
