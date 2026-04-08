
import os
from typing import TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Mock tools
def mock_tool(query: str):
    return "Mock result for " + query

class GraphState(TypedDict):
    messages: List[BaseMessage]

def agent_node(state: GraphState):
    print(f"\n[Agent] Current messages count: {len(state['messages'])}")
    for i, m in enumerate(state['messages']):
        print(f"  {i}: {type(m).__name__}")
    
    # Simulate an AI message with a tool call if it's the first turn
    if len(state['messages']) == 2:
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "mock_tool", "args": {"query": "test"}, "id": "call_1"}]
        )
        print("[Agent] Returning tool call")
        return {"messages": state["messages"] + [msg]}
    else:
        print("[Agent] Returning final response")
        return {"messages": state["messages"] + [AIMessage(content="Final response")]}

def router(state: GraphState):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END

def call_tools(state):
    print(f"\n[ToolsNode] Executing tools...")
    # Simulate ToolNode behavior
    # ToolNode returns ONLY the new ToolMessages
    out = {"messages": [ToolMessage(content="Tool result", tool_call_id="call_1", name="mock_tool")]}
    print(f"[ToolsNode] Returning: {out}")
    return out

workflow = StateGraph(GraphState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", call_tools)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", router, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")

app = workflow.compile()

initial_state = {"messages": [SystemMessage(content="System"), HumanMessage(content="Hi")]}
print("Starting graph...")
app.invoke(initial_state)
