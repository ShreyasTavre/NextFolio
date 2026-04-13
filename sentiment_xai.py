"""
sentiment_analyzer.py
─────────────────────
NLP Sentiment Analysis on financial news headlines.
Uses VADER (rule-based, no API key needed) as the primary engine,
with optional transformer-based analysis via HuggingFace pipeline.
"""

from typing import List, Dict, Optional
import re

try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
    nltk.download("vader_lexicon", quiet=True)
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

# Financial domain word boosts
FINANCIAL_LEXICON = {
    "beat":        0.6,
    "miss":       -0.6,
    "surge":       0.7,
    "plunge":     -0.8,
    "rally":       0.5,
    "crash":      -0.9,
    "outperform":  0.6,
    "downgrade":  -0.5,
    "upgrade":     0.5,
    "default":    -0.9,
    "bankruptcy": -1.0,
    "record":      0.4,
    "profit":      0.5,
    "loss":       -0.5,
    "revenue":     0.2,
    "buyback":     0.4,
    "dividend":    0.3,
    "layoff":     -0.4,
    "acquisition": 0.3,
    "lawsuit":    -0.4,
}


class SentimentAnalyzer:
    """
    Financial news sentiment scorer.

    Returns a compound score in [-1, 1]:
      > 0.3  → Bullish
      < -0.3 → Bearish
      else   → Neutral
    """

    def __init__(self):
        if VADER_AVAILABLE:
            self._vader = SentimentIntensityAnalyzer()
            # Inject financial lexicon
            self._vader.lexicon.update(FINANCIAL_LEXICON)
        else:
            self._vader = None

    def score(self, articles: List[Dict[str, str]]) -> Optional[float]:
        """
        Score a list of news articles.
        Each article: {"title": str, "description": str, ...}
        Returns average compound score, or None if no articles.
        """
        if not articles:
            return None

        texts = []
        for a in articles:
            t = f"{a.get('title', '')} {a.get('description', '')}".strip()
            if t:
                texts.append(t)

        if not texts:
            return None

        if self._vader:
            scores = [self._vader.polarity_scores(t)["compound"] for t in texts]
        else:
            scores = [self._simple_score(t) for t in texts]

        avg = sum(scores) / len(scores)
        return round(float(avg), 4)

    def label(self, score: Optional[float]) -> str:
        if score is None:
            return "No Data"
        if score > 0.5:
            return "Extreme Bullish"
        if score > 0.2:
            return "Bullish"
        if score > -0.2:
            return "Neutral"
        if score > -0.5:
            return "Bearish"
        return "Extreme Bearish"

    def _simple_score(self, text: str) -> float:
        """Rule-based fallback when VADER is unavailable."""
        text_lower = text.lower()
        score = 0.0
        for word, val in FINANCIAL_LEXICON.items():
            if word in text_lower:
                score += val
        return max(-1.0, min(1.0, score / max(1, len(text.split()))))


# ─────────────────────────────────────────────────────────────────────────────


"""
xai_layer.py
────────────
Explainable AI (XAI) Layer.
Converts raw portfolio risk JSON into a 3-point executive summary
using the Anthropic Claude API (claude-sonnet-4-20250514).
"""

import json
from typing import Dict, Any


class XAILayer:
    """
    Generates human-readable 3-point executive summaries from raw risk data.
    Uses the Anthropic Claude API via the standard /v1/messages endpoint.
    """

    SYSTEM_PROMPT = """You are a senior portfolio risk analyst providing objective, data-driven insights.
Your role is to interpret quantitative risk data and provide clear, analytical observations.
You do NOT provide direct investment advice or tell users to buy/sell specific assets.
Instead, you flag risks, patterns, and exposures in an objective, professional tone.
Always frame insights as "Risk of X detected" or "Exposure analysis indicates Y" rather than "You should Z".
Respond ONLY with a JSON object in this exact format:
{
  "point_1": {"title": "...", "detail": "...", "severity": "Low|Medium|High|Critical"},
  "point_2": {"title": "...", "detail": "...", "severity": "Low|Medium|High|Critical"},
  "point_3": {"title": "...", "detail": "...", "severity": "Low|Medium|High|Critical"},
  "overall_assessment": "..."
}"""

    def generate_summary(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call Claude API to generate executive summary from risk data.
        Falls back to rule-based summary if API call fails.
        """
        try:
            import urllib.request
            payload = json.dumps({
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1000,
                "system": self.SYSTEM_PROMPT,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Analyse the following portfolio risk data and provide a 3-point "
                            "executive summary with severity ratings:\n\n"
                            f"{json.dumps(raw_data, indent=2)}"
                        )
                    }
                ]
            }).encode()

            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data    = json.loads(resp.read())
                content = data["content"][0]["text"].strip()
                content = content.replace("```json", "").replace("```", "").strip()
                return json.loads(content)

        except Exception as e:
            return self._rule_based_summary(raw_data, str(e))

    def _rule_based_summary(self, data: Dict[str, Any], error: str = "") -> Dict[str, Any]:
        """Deterministic fallback when LLM is unavailable."""
        health  = data.get("health_score", {})
        flags   = data.get("hidden_flags", {})
        sector  = data.get("sector_exposure", {})
        metrics = data.get("metrics", {})

        score   = health.get("overall_score", 50)
        top_sec = max(sector, key=sector.get) if sector else "Unknown"
        top_exp = sector.get(top_sec, 0)

        # Point 1: Concentration risk
        n_pairs = flags.get("n_flagged_pairs", 0)
        p1_sev  = "High" if n_pairs >= 3 else "Medium" if n_pairs >= 1 else "Low"
        p1 = {
            "title":    "Correlation Concentration Risk",
            "detail":   (
                f"{n_pairs} highly-correlated pair(s) detected (r > 0.85). "
                f"Affected tickers: {', '.join(flags.get('at_risk_tickers', [])[:4]) or 'None'}."
                if n_pairs else "No significant correlation concentration detected."
            ),
            "severity": p1_sev,
        }

        # Point 2: Sector exposure
        p2_sev = "High" if top_exp > 0.5 else "Medium" if top_exp > 0.3 else "Low"
        p2 = {
            "title":    "Sector Overexposure Analysis",
            "detail":   (
                f"Portfolio carries {top_exp*100:.1f}% exposure to the '{top_sec}' sector. "
                f"{'Significant concentration risk.' if top_exp > 0.4 else 'Moderate exposure, within acceptable range.'}"
            ),
            "severity": p2_sev,
        }

        # Point 3: Portfolio health & Sharpe
        port_sharpe = health.get("portfolio_sharpe", 0)
        p3_sev      = "High" if port_sharpe < 0 else "Medium" if port_sharpe < 0.5 else "Low"
        p3 = {
            "title":    "Risk-Adjusted Return Quality",
            "detail":   (
                f"Portfolio Health Score: {score}/100 (Grade {health.get('grade','?')}). "
                f"Sharpe Ratio: {port_sharpe:.2f}. "
                f"{'Sub-optimal risk-adjusted returns detected.' if port_sharpe < 0.5 else 'Adequate risk-adjusted performance.'}"
            ),
            "severity": p3_sev,
        }

        overall = (
            f"Portfolio scored {score}/100. "
            f"Primary concern: {['sector overexposure', 'correlation clusters', 'low Sharpe ratio'][0]}. "
            f"Recommend reviewing holdings in {top_sec} sector."
        )

        return {
            "point_1": p1,
            "point_2": p2,
            "point_3": p3,
            "overall_assessment": overall,
            "_fallback": True,
            "_error":    error,
        }
