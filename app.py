import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from agent.model import stock_graph
from fastapi.concurrency import run_in_threadpool
import anthropic
import yfinance as yf
import uvicorn
from typing import List
from config.settings import CEREBRAS_API
from openai import OpenAI


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "your-anthropic-key")


app = FastAPI(title="Aurelius — AI Stock Intelligence")
templates = Jinja2Templates(directory="templates")



class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    ticker:str
    final_decision:       str
    technical_analysis:   str | None = None
    news_analysis:        str | None = None
    fundamental_analysis: str | None = None

class ChatMessage(BaseModel):
    role:    str
    content: str

class ChatRequest(BaseModel):
    system:   str
    messages: List[ChatMessage]

class ChatResponse(BaseModel):
    reply: str

class QuoteResponse(BaseModel):
    symbol:     str
    price:      float | None = None
    change:     float | None = None
    change_pct: float | None = None
    currency:   str = "INR"



@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the Aurelius dashboard"""
    return templates.TemplateResponse("index.html", {"request": request})

def _extract_ticker_via_llm(query: str) -> str:
    client = OpenAI(
        base_url="https://api.cerebras.ai/v1",
        api_key=CEREBRAS_API,
    )
    resp = client.chat.completions.create(
        model="qwen-3-235b-a22b-instruct-2507",
        max_tokens=10,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a stock ticker extractor. "
                    "Given a user query about a stock, reply with ONLY the exchange ticker symbol "
                    "(e.g. RELIANCE, TCS, HDFCBANK, AAPL). "
                    "No punctuation, no explanation, just the symbol."
                ),
            },
            {"role": "user", "content": query},
        ],
    )
    ticker = resp.choices[0].message.content or ""
    import re
    ticker = re.sub(r"[^A-Za-z]", "", ticker).upper().strip()
    return ticker or "STOCK"


@app.post("/predict", response_model=QueryResponse)
async def predict(request: QueryRequest):
    """Run full stock analysis; ticker extracted via LLM in parallel with the graph."""
    import asyncio
 
    try:
        graph_task  = asyncio.create_task(
            run_in_threadpool(stock_graph.invoke, {"query": request.query})
        )
        ticker_task = asyncio.create_task(
            run_in_threadpool(_extract_ticker_via_llm, request.query)
        )
 
        response, ticker = await asyncio.gather(graph_task, ticker_task)
 
        technical = response.get("technical_analysis")
        if technical and any(x in technical for x in [
            "not currently supported",
            "not in supported",
            "ERROR:",
            "not supported by the TFT",
            "Action Required",
            "Please select a supported",
        ]):
            technical = None
 
        final = response.get("final_decision") or ""
        if not final:
            raise ValueError(
                "Analysis produced no final decision. "
                f"technical={bool(technical)}, "
                f"news={bool(response.get('news_analysis'))}, "
                f"fundamentals={bool(response.get('fundamental_analysis'))}"
            )
 
        return QueryResponse(
            ticker               = ticker,
            final_decision       = final,
            technical_analysis   = technical,
            news_analysis        = response.get("news_analysis"),
            fundamental_analysis = response.get("fundamental_analysis"),
        )
 
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        print("i am triggered")
        client = OpenAI(
            base_url="https://api.cerebras.ai/v1",
            api_key=CEREBRAS_API,
        )


        api_messages = [
            {"role": m.role, "content": m.content}
            for m in request.messages
        ]

        if api_messages and api_messages[0]["role"] == "assistant":
            api_messages = api_messages[1:]

        response = await run_in_threadpool(
            lambda: client.chat.completions.create(
                model="qwen-3-235b-a22b-instruct-2507",
                messages=[
                    {"role": "system", "content": request.system},
                    *api_messages
                ],
                max_tokens=3000
            )
        )

        reply = response.choices[0].message.content if response.choices[0].message.content else "I Couldn't Process That"
        return ChatResponse(reply=reply)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/quote/{symbol}", response_model=QuoteResponse)
async def get_quote(symbol: str):
    """Fetch live price for a stock symbol"""
    try:

        def _fetch():
            t = yf.Ticker(symbol.upper() + ".NS")
            info = t.fast_info
            if hasattr(info, 'last_price') and info.last_price:
                return {
                    "symbol":     symbol.upper(),
                    "price":      round(float(info.last_price), 2),
                    "change":     round(float(info.last_price - info.previous_close), 2),
                    "change_pct": round(((info.last_price - info.previous_close) / info.previous_close) * 100, 2),
                    "currency":   "INR",
                }
            t2 = yf.Ticker(symbol.upper())
            info2 = t2.fast_info
            if hasattr(info2, 'last_price') and info2.last_price:
                return {
                    "symbol":     symbol.upper(),
                    "price":      round(float(info2.last_price), 2),
                    "change":     round(float(info2.last_price - info2.previous_close), 2),
                    "change_pct": round(((info2.last_price - info2.previous_close) / info2.previous_close) * 100, 2),
                    "currency":   "USD",
                }
            return None

        data = await run_in_threadpool(_fetch)
        if not data:
            return QuoteResponse(symbol=symbol.upper())
        return QuoteResponse(**data)

    except Exception:
        return QuoteResponse(symbol=symbol.upper())



if __name__ == "__main__":
    uvicorn.run("app:app", port=6969, reload=True)