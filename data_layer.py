"""
data_layer.py
─────────────
Market Data Pipeline using yfinance + newsapi-python.
Handles price history, fundamentals, and news fetching.
"""

import yfinance as yf
import pandas as pd
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Optional: pip install newsapi-python
try:
    from newsapi import NewsApiClient
    NEWSAPI_AVAILABLE = True
except ImportError:
    NEWSAPI_AVAILABLE = False


# ── Configuration ──────────────────────────────────────────────────────────────

NEWSAPI_KEY = "YOUR_NEWSAPI_KEY_HERE"   # Set via env var in production


# ── Main Pipeline Class ────────────────────────────────────────────────────────

class MarketDataPipeline:
    """
    Unified data retrieval layer.

    Parameters
    ----------
    tickers : list of str
        e.g. ["RELIANCE.NS", "TCS.NS", "AAPL", "NVDA"]
    period  : str
        yfinance period string: "1d","5d","1mo","3mo","6mo","1y","2y","5y","max"
    """

    def __init__(self, tickers: List[str], period: str = "1y"):
        self.tickers = tickers
        self.period  = period
        self._prices: Optional[pd.DataFrame] = None
        self._info:   Dict[str, Dict]         = {}

    # ── Price Data ─────────────────────────────────────────────────────────────

    def get_prices(self, column: str = "Adj Close") -> pd.DataFrame:
        """
        Return a DataFrame of adjusted closing prices.
        Rows = dates, Columns = tickers.
        Missing values are forward-filled then back-filled.
        """
        if self._prices is not None:
            return self._prices

        logger.info(f"Fetching price data for {self.tickers}, period={self.period}")
        raw = yf.download(
            self.tickers,
            period=self.period,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
        )

        if len(self.tickers) == 1:
            prices = raw[["Close"]].rename(columns={"Close": self.tickers[0]})
        else:
            # Multi-ticker download returns MultiIndex columns
            prices = pd.DataFrame()
            for t in self.tickers:
                try:
                    col = raw[(t, "Close")] if (t, "Close") in raw.columns else raw["Close"][t]
                    prices[t] = col
                except Exception as e:
                    logger.warning(f"Could not extract prices for {t}: {e}")

        prices = prices.ffill().bfill().dropna(how="all")
        self._prices = prices
        return prices

    def get_returns(self) -> pd.DataFrame:
        """Daily log returns."""
        import numpy as np
        prices = self.get_prices()
        return np.log(prices / prices.shift(1)).dropna()

    # ── Fundamentals ───────────────────────────────────────────────────────────

    def get_fundamentals(self) -> Dict[str, Dict[str, Any]]:
        """
        Return a dict keyed by ticker with fundamental fields:
        sector, industry, trailingPE, forwardPE, marketCap, beta, etc.
        """
        if self._info:
            return self._info

        for ticker in self.tickers:
            try:
                t = yf.Ticker(ticker)
                info = t.info or {}
                self._info[ticker] = {
                    "shortName":            info.get("shortName", ticker),
                    "sector":               info.get("sector", "Unknown"),
                    "industry":             info.get("industry", "Unknown"),
                    "marketCap":            info.get("marketCap"),
                    "trailingPE":           info.get("trailingPE"),
                    "forwardPE":            info.get("forwardPE"),
                    "priceToBook":          info.get("priceToBook"),
                    "dividendYield":        info.get("dividendYield"),
                    "fiveYearAvgDividendYield": info.get("fiveYearAvgDividendYield"),
                    "beta":                 info.get("beta"),
                    "fiftyTwoWeekHigh":     info.get("fiftyTwoWeekHigh"),
                    "fiftyTwoWeekLow":      info.get("fiftyTwoWeekLow"),
                    "currency":             info.get("currency", "USD"),
                    "country":              info.get("country", "Unknown"),
                    "currentPrice":         info.get("currentPrice") or info.get("regularMarketPrice"),
                }
            except Exception as e:
                logger.error(f"Failed to fetch fundamentals for {ticker}: {e}")
                self._info[ticker] = {"sector": "Unknown", "shortName": ticker}

        return self._info

    # ── News Data ──────────────────────────────────────────────────────────────

    def get_news(self, ticker: str, max_articles: int = 10) -> List[Dict[str, str]]:
        """
        Fetch recent news headlines for a ticker.
        Falls back to yfinance news if NewsAPI key is unavailable.
        """
        articles = []

        # Try NewsAPI first
        if NEWSAPI_AVAILABLE and NEWSAPI_KEY != "YOUR_NEWSAPI_KEY_HERE":
            try:
                client = NewsApiClient(api_key=NEWSAPI_KEY)
                from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
                query     = self._info.get(ticker, {}).get("shortName", ticker)
                resp      = client.get_everything(
                    q=query,
                    from_param=from_date,
                    language="en",
                    sort_by="relevancy",
                    page_size=max_articles,
                )
                for a in resp.get("articles", []):
                    articles.append({
                        "title":       a.get("title", ""),
                        "description": a.get("description", ""),
                        "source":      a.get("source", {}).get("name", ""),
                    })
                return articles
            except Exception as e:
                logger.warning(f"NewsAPI failed for {ticker}: {e}")

        # Fallback: yfinance news
        try:
            t    = yf.Ticker(ticker)
            news = t.news or []
            for item in news[:max_articles]:
                articles.append({
                    "title":       item.get("title", ""),
                    "description": item.get("summary", ""),
                    "source":      item.get("publisher", ""),
                })
        except Exception as e:
            logger.error(f"yfinance news failed for {ticker}: {e}")

        return articles

    # ── Sector Mapping ─────────────────────────────────────────────────────────

    def sector_map(self) -> Dict[str, str]:
        """Return {ticker: sector} mapping."""
        info = self.get_fundamentals()
        return {t: info[t].get("sector", "Unknown") for t in self.tickers}

    # ── Utility ────────────────────────────────────────────────────────────────

    def latest_prices(self) -> Dict[str, float]:
        """Return the most recent price for each ticker."""
        prices = self.get_prices()
        return {col: float(prices[col].iloc[-1]) for col in prices.columns}

    def price_summary(self) -> pd.DataFrame:
        """Quick stats table: last price, 52-wk high/low, YTD return."""
        prices = self.get_prices()
        info   = self.get_fundamentals()
        rows   = []
        for t in self.tickers:
            if t not in prices.columns:
                continue
            series = prices[t].dropna()
            ytd_ret = (series.iloc[-1] / series.iloc[0] - 1) * 100
            rows.append({
                "Ticker":        t,
                "Last Price":    round(series.iloc[-1], 2),
                "YTD Return %":  round(ytd_ret, 2),
                "52W High":      info[t].get("fiftyTwoWeekHigh"),
                "52W Low":       info[t].get("fiftyTwoWeekLow"),
                "Sector":        info[t].get("sector", "Unknown"),
                "Currency":      info[t].get("currency", "USD"),
            })
        return pd.DataFrame(rows).set_index("Ticker")
