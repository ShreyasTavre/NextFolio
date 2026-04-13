"""
AI-Powered Financial Decision Intelligence Engine
FastAPI Backend — Entry Point
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from data_layer import MarketDataPipeline
from quant_engine import QuantEngine
from portfolio_health import PortfolioHealthAnalyzer
from scenario_simulator import MonteCarloSimulator
from substitution_engine import SmartSubstitutionEngine
from sentiment_analyzer import SentimentAnalyzer
from xai_layer import XAILayer

app = FastAPI(
    title="Financial Decision Intelligence Engine",
    description="Portfolio Intelligence & Risk Simulation Agent",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response Models ──────────────────────────────────────────────────

class PortfolioRequest(BaseModel):
    tickers: List[str]
    weights: Optional[List[float]] = None   # must sum to 1 if provided
    period: str = "1y"

class StressTestRequest(BaseModel):
    tickers: List[str]
    weights: Optional[List[float]] = None
    sector: str
    shock_pct: float          # e.g. -0.10  for a 10 % drop
    simulations: int = 10_000

class SubstitutionRequest(BaseModel):
    high_risk_ticker: str
    portfolio_tickers: List[str]
    candidate_universe: Optional[List[str]] = None

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "Financial Intelligence Engine"}


@app.post("/portfolio/health")
def portfolio_health(req: PortfolioRequest):
    """Full portfolio health dashboard: correlation, sector exposure, risk scores."""
    pipeline   = MarketDataPipeline(req.tickers, req.period)
    prices     = pipeline.get_prices()
    info       = pipeline.get_fundamentals()
    quant      = QuantEngine(prices)
    analyzer   = PortfolioHealthAnalyzer(prices, info, req.weights)

    return {
        "correlation_matrix":       quant.correlation_matrix().to_dict(),
        "portfolio_health_score":   analyzer.health_score(),
        "sector_exposure":          analyzer.sector_exposure(),
        "hidden_correlation_flags": analyzer.hidden_correlation_flags(threshold=0.85),
        "metrics": {
            t: {
                "beta":        quant.beta(t),
                "volatility":  quant.volatility(t),
                "rsi":         quant.rsi(t),
                "sharpe":      quant.sharpe(t),
            }
            for t in req.tickers
        }
    }


@app.post("/portfolio/stress-test")
def stress_test(req: StressTestRequest):
    """Monte Carlo stress test with sector-shock scenario."""
    pipeline  = MarketDataPipeline(req.tickers, "2y")
    prices    = pipeline.get_prices()
    simulator = MonteCarloSimulator(prices, req.weights)

    result = simulator.run(
        sector=req.sector,
        shock_pct=req.shock_pct,
        simulations=req.simulations,
    )
    return result


@app.post("/portfolio/substitution")
def substitution(req: SubstitutionRequest):
    """Find a lower-risk mathematical twin for a high-risk holding."""
    universe = req.candidate_universe or []
    engine   = SmartSubstitutionEngine(req.portfolio_tickers)
    twin     = engine.find_twin(req.high_risk_ticker, universe)
    return twin


@app.post("/portfolio/hype-meter")
def hype_meter(req: PortfolioRequest):
    """Compare NLP sentiment vs fundamental P/E to flag hype overload."""
    pipeline  = MarketDataPipeline(req.tickers, req.period)
    info      = pipeline.get_fundamentals()
    analyzer  = SentimentAnalyzer()
    results   = {}

    for ticker in req.tickers:
        news       = pipeline.get_news(ticker, max_articles=10)
        sentiment  = analyzer.score(news)
        pe         = info.get(ticker, {}).get("trailingPE", None)
        pe_5yr_avg = info.get(ticker, {}).get("fiveYearAvgDividendYield", None)
        flag       = _hype_flag(sentiment, pe, pe_5yr_avg)
        results[ticker] = {
            "sentiment_score":  sentiment,
            "trailing_pe":      pe,
            "flag":             flag,
        }

    return results


@app.post("/portfolio/insights")
def insights(req: PortfolioRequest):
    """XAI layer: convert raw risk data into a 3-point executive summary."""
    pipeline = MarketDataPipeline(req.tickers, req.period)
    prices   = pipeline.get_prices()
    info     = pipeline.get_fundamentals()
    quant    = QuantEngine(prices)
    analyzer = PortfolioHealthAnalyzer(prices, info, req.weights)
    xai      = XAILayer()

    raw_data = {
        "health_score":    analyzer.health_score(),
        "sector_exposure": analyzer.sector_exposure(),
        "hidden_flags":    analyzer.hidden_correlation_flags(0.85),
        "metrics": {
            t: {
                "beta":       quant.beta(t),
                "volatility": quant.volatility(t),
                "sharpe":     quant.sharpe(t),
            }
            for t in req.tickers
        }
    }
    summary = xai.generate_summary(raw_data)
    return {"executive_summary": summary, "raw_data": raw_data}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _hype_flag(sentiment: float, pe: Optional[float], pe_avg: Optional[float]) -> str:
    if sentiment is None:
        return "Insufficient Data"
    if sentiment > 0.6:
        if pe and pe_avg and pe > pe_avg * 1.5:
            return "🚨 Hype Overload"
        return "⚠️ Elevated Sentiment"
    if sentiment < -0.4:
        return "📉 Extreme Fear"
    return "✅ Balanced"


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
