"""
portfolio_health.py
───────────────────
Portfolio Health Analyzer.

Computes:
  • Overall health score (0–100)
  • Sector exposure breakdown
  • Hidden correlation flags (concentration risk)
  • Diversification score
  • Weighted risk index
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any

from quant_engine import QuantEngine


class PortfolioHealthAnalyzer:
    """
    Analyses portfolio-level health using price data and fundamental metadata.

    Parameters
    ----------
    prices   : pd.DataFrame   – Adjusted closing prices (rows=dates, cols=tickers)
    info     : dict           – Fundamentals dict from MarketDataPipeline.get_fundamentals()
    weights  : list[float]    – Portfolio weights (must sum to 1); defaults to equal-weight
    """

    def __init__(
        self,
        prices:  pd.DataFrame,
        info:    Dict[str, Any],
        weights: Optional[List[float]] = None,
    ):
        self.prices   = prices
        self.info     = info
        self.tickers  = list(prices.columns)
        n             = len(self.tickers)
        self.weights  = np.array(weights) / np.sum(weights) if weights else np.ones(n) / n
        self.quant    = QuantEngine(prices)

    # ── Sector Exposure ───────────────────────────────────────────────────────

    def sector_exposure(self) -> Dict[str, float]:
        """
        Return a dict mapping each sector to its portfolio weight percentage.
        e.g. {"Technology": 0.45, "Energy": 0.30, "Unknown": 0.25}
        """
        sector_weights: Dict[str, float] = {}
        for ticker, weight in zip(self.tickers, self.weights):
            sector = self.info.get(ticker, {}).get("sector", "Unknown")
            sector_weights[sector] = sector_weights.get(sector, 0.0) + float(weight)
        # Round to 4 dp
        return {k: round(v, 4) for k, v in sorted(sector_weights.items(), key=lambda x: -x[1])}

    # ── Hidden Correlation Flags ──────────────────────────────────────────────

    def hidden_correlation_flags(self, threshold: float = 0.85) -> Dict[str, Any]:
        """
        Detect clusters of stocks with pairwise Pearson r > threshold.
        Returns:
          - high_pairs: list of (ticker_a, ticker_b, r)
          - at_risk_tickers: tickers appearing in ≥1 high-correlation pair
          - severity: "High" / "Medium" / "Low"
          - message: human-readable summary
        """
        corr   = self.quant.correlation_matrix()
        pairs  = self.quant.high_correlation_pairs(threshold)

        at_risk = set()
        for a, b, _ in pairs:
            at_risk.update([a, b])

        n_pairs   = len(pairs)
        n_total   = len(self.tickers) * (len(self.tickers) - 1) // 2
        ratio     = n_pairs / n_total if n_total > 0 else 0

        if ratio >= 0.5:
            severity = "High"
        elif ratio >= 0.2:
            severity = "Medium"
        else:
            severity = "Low"

        if n_pairs == 0:
            message = "No hidden concentration risk detected — portfolio appears well-diversified."
        else:
            message = (
                f"Concentration risk detected: {n_pairs} pair(s) exhibit correlation >{threshold}. "
                f"Tickers at risk: {', '.join(sorted(at_risk))}. "
                f"Consider rebalancing to reduce systemic exposure."
            )

        return {
            "high_pairs":      pairs,
            "at_risk_tickers": sorted(at_risk),
            "n_flagged_pairs": n_pairs,
            "severity":        severity,
            "message":         message,
        }

    # ── Individual Risk Scores ────────────────────────────────────────────────

    def _individual_risk_scores(self) -> Dict[str, float]:
        """
        Composite risk score per ticker (0=low risk, 100=extreme risk).
        Formula: 0.3×normalised_volatility + 0.3×normalised_beta + 0.2×RSI_extreme + 0.2×negative_sharpe
        """
        vols   = {t: self.quant.volatility(t)    for t in self.tickers}
        betas  = {t: self.quant.beta(t)           for t in self.tickers}
        rsis   = {t: self.quant.rsi(t)            for t in self.tickers}
        sharps = {t: self.quant.sharpe_ratio(t)   for t in self.tickers}

        def safe(arr):
            vals = [v for v in arr.values() if not np.isnan(v)]
            return (min(vals), max(vals)) if vals else (0, 1)

        v_min, v_max = safe(vols)
        b_min, b_max = safe(betas)

        scores = {}
        for t in self.tickers:
            vol_norm  = _norm(vols[t],   v_min, v_max)
            beta_norm = _norm(betas[t],  b_min, b_max)

            rsi_val   = rsis[t] if not np.isnan(rsis[t]) else 50
            rsi_score = max(0, (abs(rsi_val - 50) - 20)) / 30   # extreme RSI (>70 or <30) scores high

            sharpe_val   = sharps[t] if not np.isnan(sharps[t]) else 0
            sharpe_score = max(0, -sharpe_val) / 2               # negative Sharpe is risky

            composite = (
                0.35 * vol_norm +
                0.30 * beta_norm +
                0.20 * rsi_score +
                0.15 * sharpe_score
            )
            scores[t] = round(min(100, composite * 100), 1)

        return scores

    # ── Portfolio Health Score ────────────────────────────────────────────────

    def health_score(self) -> Dict[str, Any]:
        """
        Compute an overall Portfolio Health Score (0–100, higher = healthier).

        Components:
          40% – Weighted average individual risk (inverted)
          25% – Diversification (1 / avg_correlation across all pairs)
          20% – Sector concentration penalty
          15% – Sharpe adequacy
        """
        # Component 1: Weighted risk
        risk_scores   = self._individual_risk_scores()
        weighted_risk = sum(
            risk_scores[t] * w for t, w in zip(self.tickers, self.weights)
            if t in risk_scores
        )
        risk_component = max(0, 100 - weighted_risk)  # invert: low risk → high health

        # Component 2: Diversification
        corr = self.quant.correlation_matrix()
        upper_tri = [
            corr.iloc[i, j]
            for i in range(len(corr))
            for j in range(i + 1, len(corr))
        ]
        avg_corr          = np.nanmean(upper_tri) if upper_tri else 0
        divers_component  = max(0, (1 - avg_corr) * 100)

        # Component 3: Sector concentration (Herfindahl-Hirschman Index style)
        exposures         = list(self.sector_exposure().values())
        hhi               = sum(e ** 2 for e in exposures)   # 0 (perfect div) → 1 (monopoly)
        sector_component  = max(0, (1 - hhi) * 100)

        # Component 4: Sharpe adequacy
        port_sharpe       = self.quant.portfolio_metrics(list(self.weights))["sharpe_ratio"]
        sharpe_component  = min(100, max(0, (port_sharpe + 1) / 3 * 100))  # [-1,2] → [0,100]

        total = (
            0.40 * risk_component   +
            0.25 * divers_component +
            0.20 * sector_component +
            0.15 * sharpe_component
        )
        total = round(total, 1)

        grade = (
            "A" if total >= 80 else
            "B" if total >= 65 else
            "C" if total >= 50 else
            "D" if total >= 35 else "F"
        )

        return {
            "overall_score":       total,
            "grade":               grade,
            "risk_component":      round(risk_component,   1),
            "divers_component":    round(divers_component, 1),
            "sector_component":    round(sector_component, 1),
            "sharpe_component":    round(sharpe_component, 1),
            "avg_correlation":     round(float(avg_corr),  4),
            "hhi_sector":          round(float(hhi),       4),
            "portfolio_sharpe":    round(float(port_sharpe), 3),
            "individual_risk_scores": risk_scores,
            "interpretation": _interpret_health(total),
        }

    # ── Concentration Report ──────────────────────────────────────────────────

    def concentration_report(self) -> Dict[str, Any]:
        """Full concentration report combining weight, sector, and correlation risk."""
        weight_dict     = dict(zip(self.tickers, self.weights.tolist()))
        top_holding     = max(weight_dict, key=weight_dict.get)
        sector_exp      = self.sector_exposure()
        top_sector      = max(sector_exp, key=sector_exp.get)
        corr_flags      = self.hidden_correlation_flags()

        return {
            "top_holding":          top_holding,
            "top_holding_weight":   round(weight_dict[top_holding], 4),
            "top_sector":           top_sector,
            "top_sector_exposure":  sector_exp[top_sector],
            "correlation_risk":     corr_flags,
            "weight_distribution":  {t: round(w, 4) for t, w in weight_dict.items()},
        }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _norm(value, vmin, vmax):
    if np.isnan(value) or vmax == vmin:
        return 0.0
    return max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))


def _interpret_health(score: float) -> str:
    if score >= 80:
        return "Portfolio exhibits strong diversification and controlled risk exposure."
    if score >= 65:
        return "Portfolio health is adequate; moderate concentration risk detected in one or more areas."
    if score >= 50:
        return "Portfolio shows elevated risk. Review sector concentration and high-beta holdings."
    if score >= 35:
        return "Portfolio health is poor. Significant correlation clusters and/or sector overexposure detected."
    return "Critical risk detected. Portfolio is highly concentrated and vulnerable to systemic shocks."
