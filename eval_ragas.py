import json
import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_groq import ChatGroq
from graph import build_graph
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage

def run_evaluation():
    # Load test data
    with open("eval_tests.json", "r") as f:
        test_cases = json.load(f)

    # Initialize Graph
    app = build_graph()
    config = {"configurable": {"thread_id": "eval_thread"}}

    # Prepare lists for Ragas
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    # Scoring LLM (using the project's model for consistency)
    # We must ensure GROQ_API_KEY is in the environment
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("ERROR: GROQ_API_KEY not found in environment.")
        return

    eval_llm = ChatGroq(api_key=groq_api_key, model="llama-3.3-70b-versatile", temperature=0)

    print(f"Starting evaluation on {len(test_cases)} cases...")

    for case in test_cases:
        question = case["question"]
        ground_truth = case["ground_truth"]
        
        print(f"\nEvaluating: {question}")
        
        state = {"messages": [HumanMessage(content=question)]}
        
        # Run graph
        result_state = None
        # We use app.invoke or app.stream. To handle HITL, we might need to loop.
        try:
            # Simple run. If HITL triggers, we need to approve it.
            # In ABIDA, HITL triggers before 'tools' node if store_analysis_result is called.
            # Our eval_tests shouldn't trigger it, but let's be safe.
            
            for event in app.stream(state, config, stream_mode="values"):
                result_state = event
                
                # Check for interrupt
                snapshot = app.get_state(config)
                if snapshot.next:
                    print(f"  [HITL] Auto-approving interrupt: {snapshot.next}")
                    # Resume with None to approve the current tool calls
                    for sub_event in app.stream(None, config, stream_mode="values"):
                        result_state = sub_event
        except Exception as e:
            print(f"  [Error] Failed to run graph for this query: {e}")
            continue
        
        if not result_state:
            print("  [Error] No result state returned.")
            continue

        # Extract Answer and Contexts
        messages = result_state["messages"]
        final_answer = messages[-1].content
        
        # Extract contexts from 'grounding_search' ToolMessages
        retrieved_docs = []
        for msg in messages:
            if isinstance(msg, ToolMessage) and msg.name == "grounding_search":
                try:
                    # The content is a JSON string of hits
                    import json as py_json
                    tool_res = py_json.loads(msg.content)
                    if isinstance(tool_res, list):
                        for hit in tool_res:
                            if "text" in hit:
                                retrieved_docs.append(hit["text"])
                            else:
                                retrieved_docs.append(str(hit))
                    else:
                        retrieved_docs.append(str(msg.content))
                except Exception as e:
                    print(f"  [Warning] Could not parse ToolMessage content: {e}")
                    retrieved_docs.append(str(msg.content))
        
        # If no RAG was used, contexts will be empty, which is fine for evaluation
        questions.append(question)
        answers.append(final_answer)
        contexts.append(retrieved_docs if retrieved_docs else ["No context retrieved"])
        ground_truths.append(ground_truth)

    # Create local Dataset
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data)

    # Run Ragas evaluation
    print("\nRunning Ragas metrics (using Groq Llama 3.3)...")
    try:
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=eval_llm
        )

        # Output results
        print("\n" + "="*50)
        print("RAGAS EVALUATION SUMMARY")
        print("="*50)
        print(result)
        
        df = result.to_pandas()
        df.to_csv("eval_results.csv", index=False)
        print("\nDetailed results saved to eval_results.csv")
    except Exception as e:
        print(f"\n[Error] Ragas evaluation failed: {e}")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    run_evaluation()
