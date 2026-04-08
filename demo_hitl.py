
from graph import build_graph, SYSTEM_PROMPT
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
import os

def run_hitl_demo():
    app = build_graph()
    thread_id = "lab_5_refinement_thread"
    config = {"configurable": {"thread_id": thread_id}}

    print("--- HITL Rejection Demo ---")
    state = {"messages": [SystemMessage(content=SYSTEM_PROMPT)]}
    query = "Analyze 'data/superstore_us_clean.csv' and store the analysis."
    state["messages"].append(HumanMessage(content=query))

    print(f"Sending query: {query}")
    
    for event in app.stream(state, config, stream_mode="values"):
        pass

    snapshot = app.get_state(config)
    
    if snapshot.next:
        last_msg = snapshot.values["messages"][-1]
        print(f"\nGraph interrupted before tool: {[t['name'] for t in last_msg.tool_calls]}")
        
        print("\n[HITL] Human is REJECTING the tool call...")
        
        rejection_messages = []
        for tc in last_msg.tool_calls:
            rejection_messages.append(
                ToolMessage(
                    tool_call_id=tc["id"],
                    content=f"User rejected the following tool call: {tc['name']}. Do not execute it and provide a final response based on previous data."
                )
            )
        
        # Task: Human approval flow - Update state and resume
        app.update_state(config, {"messages": rejection_messages}, as_node="tools")
        print("[HITL] State updated (Rejected as 'tools' node). Resuming execution...")

        for event in app.stream(None, config, stream_mode="values"):
            pass

    final_snapshot = app.get_state(config)
    print("\n--- Final Agent Response after Rejection ---")
    print(final_snapshot.values["messages"][-1].content)

    print("\n✅ HITL Rejection refinement demonstration complete.")

if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY not found in environment.")
    else:
        run_hitl_demo()
