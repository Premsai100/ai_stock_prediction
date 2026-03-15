from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from agent.tools import (
    news_scraper_tool,
    fundamental_data_tool,
    tft_technicals_tool
)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from config.settings import GEMINI_API,GROQ_API,CEREBRAS_API
from langchain_groq import ChatGroq
from typing import TypedDict, Optional, List
from typing_extensions import Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from agent.prompts import fundamental_analysis_llm_prompt,technical_analysis_llm_prompt,primary_llm_prompt,news_analysis_llm_prompt,final_decision_llm_prompt


router_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API,
    temperature=0,
    max_retries=0
)

router_fallback_llm = ChatOpenAI(
    model="qwen-3-235b-a22b-instruct-2507",   
    api_key=CEREBRAS_API,
    base_url="https://api.cerebras.ai/v1",
    temperature=0,
    model_kwargs={"parallel_tool_calls": True}
)


router_llm_with_tools = router_llm.bind_tools(
    [news_scraper_tool, fundamental_data_tool, tft_technicals_tool]
)

router_fallback_llm_with_tools =  router_fallback_llm.bind_tools(
    [news_scraper_tool, fundamental_data_tool, tft_technicals_tool]
)

class AgentState(TypedDict):

    messages: Annotated[List[BaseMessage], add_messages]

    query: str

    tool_calls: Optional[List[dict]]

    technical_data: Optional[dict]
    news_data: Optional[dict]
    fundamental_data: Optional[dict]

    technical_analysis: Optional[str]
    news_analysis: Optional[str]
    fundamental_analysis: Optional[str]

    final_decision: Optional[str]


def primary_router(state: AgentState):
    try:
        response = router_llm_with_tools.invoke([
            SystemMessage(content=primary_llm_prompt),
            HumanMessage(content=state["query"])
        ])

    except Exception as re:
        print("--------------retrying with groq-------------")
        response = router_fallback_llm_with_tools.invoke([
            SystemMessage(content=primary_llm_prompt),
            HumanMessage(content=state["query"])
        ])

    tool_calls = response.tool_calls or []

    return {
        "messages": [response],
        "tool_calls": tool_calls
    }




def run_technical_tool(state: AgentState):
    tool_call = next(
        (t for t in state["tool_calls"] if t["name"] == "tft_technicals_tool"),
        None
    )

    if not tool_call:
        return {}

    result = tft_technicals_tool.invoke(tool_call["args"])

    return {
        "technical_data": result
    }


def run_news_tool(state: AgentState):

    print(state["tool_calls"])
    tool_call = next(
        (t for t in state["tool_calls"] if t["name"] == "news_scraper_tool"),
        None
    )

    if not tool_call:
        return {}

    result = news_scraper_tool.invoke(tool_call["args"])

    return {
        "news_data": result
    }


def run_fundamental_tool(state: AgentState):
    tool_call = next(
        (t for t in state["tool_calls"] if t["name"] == "fundamental_data_tool"),
        None
    )

    if not tool_call:
        return {}

    result = fundamental_data_tool.invoke(tool_call["args"])

    return {
        "fundamental_data": result
    }



def technical_analysis_node(state: AgentState):
    try:
        response = router_llm.invoke([
            SystemMessage(content=technical_analysis_llm_prompt),
            HumanMessage(content=str(state.get("technical_data")))
        ])
    except Exception as re:
        print("--------------retrying with groq-------------")
        response = router_fallback_llm.invoke([
            SystemMessage(content=technical_analysis_llm_prompt),
            HumanMessage(content=str(state.get("technical_data")))
        ])

    return {
        "messages": [response],
        "technical_analysis": response.content
    }


def news_analysis_node(state: AgentState):
    try:
        response = router_llm.invoke([
            SystemMessage(content=news_analysis_llm_prompt),
            HumanMessage(content=str(state.get("news_data")))
        ])
    except Exception as re:
        print("--------------retrying with groq-------------")
        response = router_fallback_llm.invoke([
            SystemMessage(content=news_analysis_llm_prompt),
            HumanMessage(content=str(state.get("news_data")))
        ])

    return {
        "messages": [response],
        "news_analysis": response.content
    }


def fundamental_analysis_node(state: AgentState):
    try:
        response = router_llm.invoke([
            SystemMessage(content=fundamental_analysis_llm_prompt),
            HumanMessage(content=str(state.get("fundamental_data")))
        ])
    except Exception as re:
        print("--------------retrying with groq-------------")
        response = router_fallback_llm.invoke([
            SystemMessage(content=fundamental_analysis_llm_prompt),
            HumanMessage(content=str(state.get("fundamental_data")))
        ])

    return {
        "messages": [response],
        "fundamental_analysis": response.content
    }




def decision_node(state: AgentState):

    analysis = f"""
Technical Analysis:
{state.get("technical_analysis")}

News Analysis:
{state.get("news_analysis")}

Fundamental Analysis:
{state.get("fundamental_analysis")}
"""
    try:
        response = router_llm.invoke([
            SystemMessage(content=final_decision_llm_prompt),
            HumanMessage(content=analysis)
        ])
    except Exception as re:
        print("--------------retrying with groq-------------")
        response = router_fallback_llm.invoke([
            SystemMessage(content=final_decision_llm_prompt),
            HumanMessage(content=analysis)
        ])

    return {
        "messages": [response],
        "final_decision": response.content
    }



def tool_router(state: AgentState):
    tools = state.get("tool_calls", [])
    
    tool_names = {t["name"] for t in tools}
    
    routes = []
    if "tft_technicals_tool" in tool_names:
        routes.append("technical_tool")
    if "news_scraper_tool" in tool_names:
        routes.append("news_tool")
    if "fundamental_data_tool" in tool_names:
        routes.append("fundamental_tool")
    
    return routes if routes else ["decision_llm"]



graph = StateGraph(AgentState)

graph.add_node("primary_router", primary_router)

graph.add_node("technical_tool", run_technical_tool)
graph.add_node("news_tool", run_news_tool)
graph.add_node("fundamental_tool", run_fundamental_tool)

graph.add_node("technical_llm", technical_analysis_node)
graph.add_node("news_llm", news_analysis_node)
graph.add_node("fundamental_llm", fundamental_analysis_node)

graph.add_node("decision_llm", decision_node)



graph.add_edge(START, "primary_router")


graph.add_conditional_edges(
    "primary_router",
    tool_router
)



graph.add_edge("technical_tool", "technical_llm")
graph.add_edge("news_tool", "news_llm")
graph.add_edge("fundamental_tool", "fundamental_llm")


graph.add_edge("technical_llm", "decision_llm")
graph.add_edge("news_llm", "decision_llm")
graph.add_edge("fundamental_llm", "decision_llm")


graph.add_edge("decision_llm", END)


stock_graph = graph.compile()




