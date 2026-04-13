"""
quant_engine.py
───────────────
Quantitative metrics engine.
Calculates Beta, Volatility, RSI, Sharpe Ratio, and Correlation Matrix.
All calculations use daily log returns.
"""

import numpy as np
import pandas as pd
from typing import Optional


RISK_FREE_RATE_ANNUAL = 0.065   # ~6.5% (India repo rate approx); change for US markets


class QuantEngine:
    """
    Quantitative metrics calculator.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame of adjusted closing prices (rows=dates, cols=tickers).
    benchmark_ticker : str
        Ticker to use as market benchmark for Beta calculation.
        Defaults to "^NSEI" (Nifty 50); use "^GSPC" for S&P 500.
    """

    def __init__(self, prices: pd.DataFrame, benchmark_ticker: str = "^NSEI"):
        self.prices   = prices
        self.returns  = np.log(prices / prices.shift(1)).dropna()
        self._bm_ticker = benchmark_ticker
        self._bm_returns: Optional[pd.Series] = None

    # ── Benchmark ─────────────────────────────────────────────────────────────

    def _get_benchmark(self) -> pd.Series:
        if self._bm_returns is not None:
            return self._bm_returns
        try:
            import yfinance as yf
            bm = yf.download(
                self._bm_ticker,
                start=self.prices.index[0],
                end=self.prices.index[-1],
                auto_adjust=True,
                progress=False,
            )["Close"]
            self._bm_returns = np.log(bm / bm.shift(1)).dropna()
        except Exception:
            # Fallback: equal-weighted portfolio as pseudo-benchmark
            self._bm_returns = self.returns.mean(axis=1)
        return self._bm_returns

    # ── Core Metrics ──────────────────────────────────────────────────────────

    def volatility(self, ticker: str, annualize: bool = True) -> float:
        """
        Annualised volatility (standard deviation of log returns).
        σ_annual = σ_daily × √252
        """
        if ticker not in self.returns.columns:
            return np.nan
        daily_vol = self.returns[ticker].std()
        return float(daily_vol * np.sqrt(252) if annualize else daily_vol)

    def beta(self, ticker: str) -> float:
        """
        Market Beta: Cov(R_stock, R_market) / Var(R_market).
        β > 1 → more volatile than market.
        """
        if ticker not in self.returns.columns:
            return np.nan
        bm     = self._get_benchmark()
        stock  = self.returns[ticker]
        aligned = pd.concat([stock, bm], axis=1, join="inner").dropna()
        if len(aligned) < 30:
            return np.nan
        cov    = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
        return float(cov[0, 1] / cov[1, 1])

    def sharpe_ratio(self, ticker: str) -> float:
        """
        Annualised Sharpe Ratio: (R_annual - R_f) / σ_annual.
        """
        if ticker not in self.returns.columns:
            return np.nan
        daily_rf     = RISK_FREE_RATE_ANNUAL / 252
        excess       = self.returns[ticker] - daily_rf
        annual_ret   = excess.mean() * 252
        annual_vol   = excess.std() * np.sqrt(252)
        return float(annual_ret / annual_vol) if annual_vol != 0 else np.nan

    # Alias for API compatibility
    def sharpe(self, ticker: str) -> float:
        return self.sharpe_ratio(ticker)

    def rsi(self, ticker: str, window: int = 14) -> float:
        """
        Relative Strength Index (RSI-14).
        RSI = 100 - [100 / (1 + RS)],  RS = avg_gain / avg_loss
        """
        if ticker not in self.prices.columns:
            return np.nan
        delta  = self.prices[ticker].diff()
        gain   = delta.clip(lower=0)
        loss   = -delta.clip(upper=0)
        avg_g  = gain.ewm(span=window, adjust=False).mean()
        avg_l  = loss.ewm(span=window, adjust=False).mean()
        rs     = avg_g / avg_l
        rsi    = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1])

    def annual_return(self, ticker: str) -> float:
        """Compound annual growth rate (CAGR) over the available price history."""
        if ticker not in self.prices.columns:
            return np.nan
        col   = self.prices[ticker].dropna()
        days  = (col.index[-1] - col.index[0]).days
        years = days / 365.25
        cagr  = (col.iloc[-1] / col.iloc[0]) ** (1 / years) - 1
        return float(cagr)

    def drawdown(self, ticker: str) -> dict:
        """Max drawdown and current drawdown from all-time high."""
        if ticker not in self.prices.columns:
            return {}
        col       = self.prices[ticker].dropna()
        roll_max  = col.cummax()
        dd        = (col - roll_max) / roll_max
        return {
            "max_drawdown":     float(dd.min()),
            "current_drawdown": float(dd.iloc[-1]),
        }

    # ── Correlation Matrix ────────────────────────────────────────────────────

    def correlation_matrix(self) -> pd.DataFrame:
        """
        Pearson correlation matrix of daily log returns.
        Returns a symmetric DataFrame (tickers × tickers).
        """
        return self.returns.corr(method="pearson")

    def high_correlation_pairs(self, threshold: float = 0.85) -> list:
        """
        Return list of (ticker_a, ticker_b, correlation) tuples where
        Pearson r > threshold (excluding self-correlation).
        Sorted descending by correlation.
        """
        corr   = self.correlation_matrix()
        pairs  = []
        tickers = corr.columns.tolist()
        for i, a in enumerate(tickers):
            for b in tickers[i + 1:]:
                r = corr.loc[a, b]
                if abs(r) >= threshold:
                    pairs.append((a, b, round(float(r), 4)))
        return sorted(pairs, key=lambda x: -abs(x[2]))

    # ── Portfolio-Level Metrics ───────────────────────────────────────────────

    def portfolio_metrics(self, weights: Optional[list] = None) -> dict:
        """
        Compute portfolio-level return, volatility, and Sharpe Ratio.
        Weights default to equal-weight.
        """
        n = len(self.returns.columns)
        w = np.array(weights) if weights else np.ones(n) / n
        w = w / w.sum()   # normalise

        port_ret   = self.returns @ w
        annual_ret = port_ret.mean() * 252
        annual_vol = port_ret.std() * np.sqrt(252)
        daily_rf   = RISK_FREE_RATE_ANNUAL / 252
        sharpe     = (port_ret - daily_rf).mean() * 252 / (annual_vol if annual_vol else 1)

        return {
            "annual_return":     round(float(annual_ret), 4),
            "annual_volatility": round(float(annual_vol), 4),
            "sharpe_ratio":      round(float(sharpe), 4),
        }

    # ── Risk-Return Table ─────────────────────────────────────────────────────

    def risk_return_table(self) -> pd.DataFrame:
        """
        Returns a DataFrame with columns:
        [Ticker, Annual_Return, Volatility, Sharpe, Beta, RSI]
        Useful for scatter plot visualisation.
        """
        rows = []
        for t in self.returns.columns:
            rows.append({
                "Ticker":         t,
                "Annual_Return":  round(self.annual_return(t) * 100, 2),
                "Volatility":     round(self.volatility(t) * 100, 2),
                "Sharpe":         round(self.sharpe_ratio(t), 3),
                "Beta":           round(self.beta(t), 3),
                "RSI":            round(self.rsi(t), 1),
            })
        return pd.DataFrame(rows).set_index("Ticker")
