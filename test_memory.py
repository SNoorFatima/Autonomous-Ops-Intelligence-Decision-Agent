from graph import build_graph
from langchain_core.messages import HumanMessage, SystemMessage
import os
from graph import SYSTEM_PROMPT

def test():
    app = build_graph()
    state = {"messages": [SystemMessage(content=SYSTEM_PROMPT)]}
    
    query = "Analyze 'data/superstore_us_clean.csv' and check if we have any historical analysis for comparison. If not, just store this new analysis summary."
    print(f"Running query: {query}")
    state["messages"].append(HumanMessage(content=query))
    
    try:
        final_state = app.invoke(state, config={"recursion_limit": 15})
        print("\n--- Final Response ---")
        print(final_state["messages"][-1].content)
        
        print("\n--- Message Sequence ---")
        for m in final_state["messages"]:
            print(f"{type(m).__name__}: {str(m.content)[:100]}...")
            if hasattr(m, 'tool_calls'):
                print(f"  Tool Calls: {[t['name'] for t in m.tool_calls]}")
                
    except Exception as e:
        print(f"\n[Error] {e}")

if __name__ == "__main__":
    test()
