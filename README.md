<div align="center">

<img src="https://img.shields.io/badge/Aurelius-AI%20Stock%20Intelligence-blue?style=for-the-badge&logo=chart-line&logoColor=white" />

# 🔮 Aurelius — AI Stock Intelligence

### *Think like an investment desk. Powered by Deep Learning + Multi-Agent AI.*

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Railway-purple?style=for-the-badge)](https://web-production-2f71.up.railway.app)
[![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-green?style=for-the-badge)](https://langchain-ai.github.io/langgraph/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-teal?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-TFT%20Model-red?style=for-the-badge&logo=pytorch)](https://pytorch.org)

</div>



## 🧠 What is Aurelius?

Aurelius is a **full-stack AI stock analysis system** built for Indian markets. You type a stock ticker — Aurelius runs a multi-agent pipeline that combines deep learning forecasts, real-time news sentiment, and fundamental screener data to produce a single, well-reasoned investment decision.

No more switching between 5 tabs. No more gut-feeling trades. Just data.

> ⚠️ **Disclaimer:** This is for educational purposes only. Not financial advice.

---

## ⚙️ How It Works — Agent Architecture

```
User Query (e.g. "Analyse TCS")
         │
         ▼
  ┌─────────────────┐
  │  Primary Router  │  ← Gemini 2.5 Flash (fallback: Cerebras Qwen 235B)
  │  (LangGraph)     │    Decides which tools to invoke
  └────────┬────────┘
           │ parallel execution
    ┌──────┼──────┐
    ▼      ▼      ▼
┌──────┐ ┌──────┐ ┌──────────┐
│ TFT  │ │ News │ │Fundament-│
│ Tool │ │ Tool │ │ als Tool │
└──┬───┘ └──┬───┘ └────┬─────┘
   ▼         ▼          ▼
┌──────┐ ┌──────┐ ┌──────────┐
│Tech  │ │News  │ │Fundament-│
│ LLM  │ │ LLM  │ │ als LLM  │
└──┬───┘ └──┬───┘ └────┬─────┘
   └────────┼───────────┘
            ▼
   ┌─────────────────┐
   │  Decision Node  │  ← Chief Investment Officer Synthesis
   │  (Aurelius      │    BUY / SELL / WAIT + Conviction Score
   │   Fusion)       │
   └─────────────────┘
```

All 3 tools execute **in parallel** via LangGraph's conditional edges — so you get a full analysis in seconds, not minutes.

---

## 🔬 The TFT Model — Technical Forecasting Engine

The heart of Aurelius is a **Temporal Fusion Transformer (TFT)** — a state-of-the-art deep learning architecture purpose-built for time-series forecasting.

### Training Data
| Property | Detail |
|---|---|
| 📈 Stocks covered | **331 Indian stocks** (NSE listed) |
| 📅 Historical depth | **10 years** of daily OHLCV data |
| 🔢 Total data points | ~1.2 million rows |
| 🧮 Features | Close, High, Low, Open, Volume + Technical Indicators |

### What it outputs
- **7-day price forecast** (base case)
- **Quantile confidence bands** — 2%, 10%, 25%, 50%, 75%, 90%, 98%
- Tells you not just *where* the price is going, but *how confident* the model is

### Why TFT over LSTM/ARIMA?
- Handles **multiple time horizons** simultaneously
- Built-in **attention mechanism** — learns which past days matter most
- Naturally outputs **probabilistic forecasts** (uncertainty-aware)
- Far better at capturing **regime changes** in volatile markets

---

## 📰 News Intelligence — Powered by Tavily

Aurelius scrapes and scores **real-time news** across 3 dimensions for every analysis:

| Layer | What it captures |
|---|---|
| 🏢 Stock-level news | Earnings, management changes, deals, legal issues |
| 🏭 Sector news | IT sector outlook, regulatory updates, macro sector trends |
| 🌍 Macro news | Fed rates, FII flows, oil prices, USD/INR impact |

**Tavily** is used as the search backbone — it returns high-quality, relevant news articles that are then passed through the LLM for sentiment scoring and signal extraction.

Output: A structured sentiment score + key market-moving signals for the LLM to reason over.

---

## 📊 Fundamental Screener — Powered by Screener.in

Aurelius scrapes **Screener.in** directly for fundamental data:

- **Valuation:** P/E, P/B, EV/EBITDA
- **Profitability:** ROE, ROCE, Net Margin
- **Cash Flow:** Free Cash Flow, Operating Cash Flow
- **Balance Sheet:** Debt/Equity, Current Ratio
- **Growth:** Revenue growth, Earnings growth (3Y/5Y)

A RAG (Retrieval-Augmented Generation) pipeline processes the scraped data and feeds it to the Fundamentals LLM node for structured analysis.

---

## ⚡ Aurelius Synthesis — The Final Decision

After all 3 agents complete, the **Decision Node** synthesizes everything:

```
Technical Score (TFT forecast + indicators)
        +
Sentiment Score (News + Macro signals)
        +
Fundamental Score (Screener data)
        =
BUY / SELL / WAIT  +  Conviction % + Entry / SL levels
```

You can also **chat directly with Aurelius** post-analysis to ask follow-up questions — "Why WAIT and not BUY?", "What's the key risk here?", "How did TCS fundamentals compare to last quarter?"

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| 🧠 Primary LLM | Gemini 2.5 Flash |
| 🔄 Fallback LLM | Cerebras — Qwen 3 235B (auto-switches on quota limits) |
| 🕸️ Agent Orchestration | LangGraph |
| 🔗 LLM Framework | LangChain |
| 📈 Forecasting Model | PyTorch + pytorch-forecasting (TFT) |
| 🌐 News Search | Tavily |
| 📊 Fundamentals | Screener.in (custom scraper) |
| 📉 Market Data | yfinance |
| 🚀 Backend | FastAPI + Uvicorn |
| 🎨 Frontend | Jinja2 + Vanilla JS |
| ☁️ Deployment | Railway |

---

## 🚀 Running Locally

### Prerequisites
- Python 3.10+
- API keys: Gemini, Cerebras, Tavily

### Setup

```bash
# Clone the repo
git clone https://github.com/Premsai100/ai_stock_prediction.git
cd ai_stock_prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Add your API keys to .env
```

### Environment Variables

```env
GEMINI_API=your_gemini_api_key
CEREBRAS_API=your_cerebras_api_key
TAVILY_API=your_tavily_api_key
```

### Run

```bash
python app.py
# Open http://127.0.0.1:6969
```

---

## 📁 Project Structure

```
aurelius/
├── agent/
│   ├── model.py          # LangGraph agent graph definition
│   ├── tools.py          # TFT, News, Fundamental tools
│   └── prompts.py        # All LLM system prompts
├── ml_models/
│   └── technical_prediction/  # Trained TFT checkpoint
├── pipelines/            # TFT data pipeline & inference
├── rag/                  # Fundamental RAG pipeline
├── scraping/             # Screener.in scraper
├── templates/            # Jinja2 HTML templates
├── config/
│   └── settings.py       # API key config
├── app.py                # FastAPI entrypoint
└── requirements.txt
```


## 🤝 Contributing

PRs are welcome! If you find a bug or want to add a feature, open an issue first so we can discuss it.

---

<div align="center">

Built with 🤍 by [Premsai](https://github.com/Premsai100)

*If this helped you, a ⭐ on the repo would mean a lot!*

</div>
