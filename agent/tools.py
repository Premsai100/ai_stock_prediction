from langchain_core.tools import tool
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from scraping.news.news_scraper import scrape_stock_news
from scraping.nse.structured_numbers_scraper import scrape_screener
from pipelines.technical_pipeline import TechnicalPredictor

@tool
def news_scraper_tool(queries: list[str]) -> dict:
    """
    Fetch recent stock-related news for the given search queries.

    Args:
        queries (list[str]): A list of search queries such as stock names,
        tickers, or company names.

    Returns:
        dict: A dictionary containing scraped news articles including
        titles, sources, timestamps, and summaries related to the queries.
    """
    print("-------------news tool called-----------------")
    news = scrape_stock_news(queries)
    return news


@tool
def fundamental_data_tool(symbol: str) -> dict:
    """
    Retrieve fundamental financial data for a given stock symbol.

    Args:
        symbol (str): Stock ticker symbol (e.g., 'AAPL', 'RELIANCE', 'TCS').

    Returns:
        dict: A dictionary containing key fundamental metrics such as
        P/E ratio, market capitalization, revenue, profit, debt, and other
        financial indicators scraped from Screener or financial data sources.
    """
    print("-------------fundamental tool called-----------------")
    fundamentals = scrape_screener(symbol)
    if not fundamentals:
        return {
            "supported": False,
            "symbol":    symbol,
            "error":     f"No fundamental data found for {symbol} — may not be an Indian equity",
            "data":      None
        }
    fundamentals["supported"] = True
    return fundamentals


@tool
def tft_technicals_tool(symbol: str) -> dict:
    """
    Generate technical analysis and 7-day price forecast for a stock
    using a Temporal Fusion Transformer (TFT) model.

    Args:
        symbol (str): Stock ticker symbol (e.g. "RELIANCE", "TCS")

    Returns:
        dict: {
            "forecast":   7-day quantile predictions (pessimistic/base/optimistic),
            "indicators": compressed indicator snapshot from 60-day data
        }
    """
    print("-------------tft tool called-----------------")
    predictor = TechnicalPredictor(symbol)
    if not predictor.check_symbol():
        return {
            "supported":  False,
            "symbol":     symbol,
            "error":      f"{symbol} is not supported by the TFT model",
            "forecast":   None,
            "indicators": None
        }
    result = predictor.predict()
    result["supported"] = True
    
    return result

