import os
import json
import sys
import nest_asyncio

from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseLLM
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, ToolMessage
from graph import build_graph

# Fix for async event loops in notebook/script environments
nest_asyncio.apply()

# 1. Define Groq Judge for CI Headless Environment
class GroqDeepEvalLLM(DeepEvalBaseLLM):
    def __init__(self, model_name="llama-3.3-70b-versatile"):
        self.model_name = model_name
        # Fails securely if GROQ_API_KEY is not set in environment
        self.llm = ChatGroq(model=model_name, temperature=0)

    def load_model(self):
        return self.llm

    def generate(self, prompt: str) -> str:
        return self.llm.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        res = await self.llm.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return self.model_name

def run_ci_evaluation():
    # Enforce Headless / CI Rules: Read purely from env
    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY environment variable is not set.")
        sys.exit(1)
        
    print("Starting Automated Quality Gate Evaluation...")

    # Load configured thresholds
    try:
        with open("eval_thresholds.json", "r") as f:
            thresholds = json.load(f)
            print(f"Loaded thresholds: {thresholds}")
    except Exception as e:
        print(f"ERROR: Failed to load eval_thresholds.json: {e}")
        sys.exit(1)

    # Load test cases
    if not os.path.exists("eval_tests.json"):
        print("ERROR: eval_tests.json not found. Aborting.")
        sys.exit(1)
        
    with open("eval_tests.json", "r") as f:
        test_cases = json.load(f)

    # Initialize Graph and Judge
    app = build_graph()
    judge = GroqDeepEvalLLM()

    # Define Metrics dynamically based on thresholds
    faith_thresh = thresholds.get("faithfulness", 0.7)
    rel_thresh = thresholds.get("answer_relevancy", 0.7)

    metrics = [
        FaithfulnessMetric(threshold=faith_thresh, model=judge),
        AnswerRelevancyMetric(threshold=rel_thresh, model=judge)
    ]

    all_passed = True
    results_output = []

    for i, case in enumerate(test_cases):
        question = case["question"]
        ground_truth = case["ground_truth"]
        
        print(f"\nEvaluating Case {i+1}/{len(test_cases)}: '{question}'")
        
        state = {"messages": [HumanMessage(content=question)]}
        current_config = {"configurable": {"thread_id": f"ci_eval_{hash(question)}"}}
        result_state = None
        
        try:
            for event in app.stream(state, current_config, stream_mode="values"):
                result_state = event
                snapshot = app.get_state(current_config)
                # Auto-approve any interrupts for headless execution
                if snapshot.next:
                    for sub_event in app.stream(None, current_config, stream_mode="values"):
                        result_state = sub_event
        except Exception as e:
            print(f"  [Error] Agent execution failed: {e}")
            all_passed = False
            continue
        
        if not result_state:
            continue

        messages = result_state["messages"]
        final_answer = messages[-1].content
        
        # Extract RAG Context
        retrieved_docs = []
        for msg in messages:
            if isinstance(msg, ToolMessage) and msg.name == "grounding_search":
                try:
                    import json as py_json
                    tool_res = py_json.loads(msg.content)
                    if isinstance(tool_res, list):
                        for hit in tool_res:
                            retrieved_docs.append(hit.get("text", str(hit)))
                except:
                    retrieved_docs.append(str(msg.content))

        test_case = LLMTestCase(
            input=question,
            actual_output=final_answer,
            retrieval_context=retrieved_docs if retrieved_docs else ["No context retrieved"],
            expected_output=ground_truth
        )

        case_results = {
            "question": question,
            "actual_output": final_answer,
            "metrics": []
        }
        case_passed = True

        for m in metrics:
            m_name = m.__class__.__name__.lower().replace("metric", "")
            print(f"  - Measuring {m_name}...")
            try:
                m.measure(test_case)
                metric_passed = m.is_successful()
                print(f"    Score: {m.score} (Threshold: {m.threshold}) -> {'PASS' if metric_passed else 'FAIL'}")
                case_results["metrics"].append({
                    "name": m_name,
                    "score": m.score,
                    "threshold": m.threshold,
                    "passed": metric_passed,
                    "reason": getattr(m, 'reason', "N/A")
                })
                if not metric_passed:
                    case_passed = False
                    all_passed = False
            except Exception as e:
                print(f"    [Error] Metric calculation failed: {e}")
                case_results["metrics"].append({
                    "name": m_name,
                    "error": str(e),
                    "passed": False
                })
                case_passed = False
                all_passed = False

        results_output.append(case_results)

    # Write machine-readable output
    with open("eval_results.json", "w") as f:
        json.dump(results_output, f, indent=2)

    print("\n==============================")
    print("QUALITY GATE EVALUATION REPORT")
    print("==============================")
    
    if not all_passed:
        print("RESULT: FAILED")
        print("One or more metrics fell below the required threshold. Deployment blocked.")
        sys.exit(1)
    else:
        print("RESULT: PASSED")
        print("All metrics met or exceeded thresholds. Code is safe for deployment.")
        sys.exit(0)

if __name__ == "__main__":
    run_ci_evaluation()
