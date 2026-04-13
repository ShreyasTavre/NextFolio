# 📊 AI-Powered Financial Decision Intelligence Engine

> **Decision Support System** — Portfolio Intelligence & Risk Simulation Agent  
> Not a trading bot. Objective, quantitative analysis only.

---

## 🏗️ Architecture Overview

```
financial_engine/
├── backend/
│   ├── main.py                # FastAPI REST API
│   ├── data_layer.py          # yfinance + NewsAPI data pipeline
│   ├── quant_engine.py        # Beta, Volatility, RSI, Sharpe, Correlation
│   ├── portfolio_health.py    # Health score, sector exposure, hidden correlation flags
│   ├── scenario_simulator.py  # Monte Carlo stress testing
│   ├── substitution_engine.py # Mathematical Twin finder
│   └── sentiment_xai.py       # NLP sentiment + Claude XAI layer
├── streamlit_app.py           # Full Streamlit dashboard UI
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API keys (optional but recommended)
```bash
export NEWSAPI_KEY="your_newsapi_key"       # https://newsapi.org/
export ANTHROPIC_API_KEY="your_claude_key"  # https://console.anthropic.com/
```

### 3a. Launch the Streamlit Dashboard (recommended)
```bash
cd financial_engine
streamlit run streamlit_app.py
```

### 3b. Launch the FastAPI Backend (for API-first usage)
```bash
cd financial_engine/backend
uvicorn main:app --reload --port 8000
# API docs: http://localhost:8000/docs
```

---

## 🎯 Feature Modules

### 1. Portfolio Health Dashboard (`portfolio_health.py`)
- **Health Score** (0–100 with letter grade A–F)
  - Weighted risk component (volatility × beta × RSI)
  - Diversification component (inverse average correlation)
  - Sector concentration (HHI index)
  - Sharpe adequacy
- **Sector Exposure**: Weighted allocation per GICS sector
- **Hidden Correlation Flags**: Detects pairs with Pearson r > 0.85 — flags as concentration risk

### 2. Quantitative Engine (`quant_engine.py`)
| Metric | Formula |
|--------|---------|
| Volatility | `σ_daily × √252` (annualised) |
| Beta | `Cov(R_stock, R_market) / Var(R_market)` |
| RSI | 14-period EWM RSI |
| Sharpe | `(R_annual - R_f) / σ_annual` (Rf = 6.5% default) |
| Correlation | Pearson on daily log returns |

### 3. Monte Carlo Stress Tester (`scenario_simulator.py`)
- **Input**: Sector, shock magnitude (e.g., -10%), simulations (default 10,000)
- **Output**: P&L distribution, VaR-95, CVaR-95, probability of loss > 5%/10%
- Sector sensitivity factors applied per GICS sector
- Cross-portfolio spill-over calculated via correlation matrix

### 4. Smart Substitution Engine (`substitution_engine.py`)
- Identifies highest-risk ticker in portfolio
- Searches candidate universe (same/related sector) for **Mathematical Twin**
- Scoring: 60% vol reduction + 20% low portfolio correlation + 20% return similarity
- Returns top-3 candidates ranked by composite score

### 5. Hype vs Fundamentals Meter (`sentiment_xai.py`)
- **NLP**: VADER with financial domain lexicon (no API key required)
- **Flags**: Extreme Bullish sentiment + trailing P/E > 30 → "🚨 Hype Overload"
- Visualised as gauge charts per ticker

### 6. XAI Layer — Executive Summary (`sentiment_xai.py`)
- Passes raw risk JSON to **Claude (claude-sonnet-4-20250514)**
- Returns structured 3-point executive summary with severity ratings
- Graceful rule-based fallback if API key is not set

---

## 📡 REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Service health check |
| `POST` | `/portfolio/health` | Full health dashboard |
| `POST` | `/portfolio/stress-test` | Monte Carlo stress test |
| `POST` | `/portfolio/substitution` | Mathematical twin finder |
| `POST` | `/portfolio/hype-meter` | Sentiment vs fundamentals |
| `POST` | `/portfolio/insights` | AI executive summary |

### Example Request
```bash
curl -X POST http://localhost:8000/portfolio/health \
  -H "Content-Type: application/json" \
  -d '{"tickers": ["RELIANCE.NS", "TCS.NS", "AAPL", "NVDA"], "period": "1y"}'
```

---

## 📊 Streamlit Dashboard Tabs

| Tab | Content |
|-----|---------|
| 🏠 Health Dashboard | Score, sector pie, risk-return scatter, metrics table |
| 🔥 Correlation Analysis | Heatmap, high-correlation pair flags |
| 📰 Hype Meter | Per-ticker sentiment gauges, P/E comparison |
| 🔄 Substitution Engine | Mathematical twin search UI |
| 🤖 AI Insights | Claude-generated executive summary |

---

## ⚠️ Disclaimer

This system is a **Decision Support System** providing objective, quantitative portfolio analysis.  
It **does not** constitute financial advice. All outputs are analytical observations, not buy/sell recommendations.  
Past performance and simulated scenarios do not guarantee future results.  
Always consult a qualified financial advisor before making investment decisions.
