import time
import requests
import json
import os
import pandas as pd
import nest_asyncio
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseLLM
from langchain_groq import ChatGroq
from graph import build_graph
from langchain_core.messages import HumanMessage, ToolMessage
from dotenv import load_dotenv

# Fix for async issues
nest_asyncio.apply()

# 1. Define Local Judge for DeepEval (Throttled & Resilient)
class LocalDeepEvalLLM(DeepEvalBaseLLM):
    def __init__(self, model_name="llama3.1:8b", endpoint="http://localhost:11435/api/generate"):
        self.model_name = model_name
        self.endpoint = endpoint

    def load_model(self):
        return self

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0}
        }
        
        # Exponential Backoff for local server stability
        base_delay = 5
        for attempt in range(5):
            try:
                response = requests.post(self.endpoint, json=payload, timeout=300)
                response.raise_for_status()
                content = response.json().get("response", "")
                
                # Sanitize Output
                if "```" in content:
                    import re
                    match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
                    if match:
                        content = match.group(1)
                
                return content.strip()
            except Exception as e:
                delay = base_delay * (2 ** attempt)
                print(f"  [Local LLM Warning] Connection failed ({e}). Retrying in {delay}s...")
                time.sleep(delay)
        
        return "Error: Local LLM failed after 5 retries."

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name

def run_evaluation():
    load_dotenv()
    
    if not os.path.exists("eval_tests.json"):
        print("ERROR: eval_tests.json not found.")
        return
        
    with open("eval_tests.json", "r") as f:
        test_cases = json.load(f)

    # Initialize Graph (Uses LocalOllamaChat internally now)
    app = build_graph()
    custom_judge = LocalDeepEvalLLM()

    # Metrics
    metrics = [
        FaithfulnessMetric(threshold=0.7, model=custom_judge),
        AnswerRelevancyMetric(threshold=0.7, model=custom_judge),
        ContextualPrecisionMetric(threshold=0.7, model=custom_judge),
        ContextualRecallMetric(threshold=0.7, model=custom_judge)
    ]

    print(f"Starting Robust Local Evaluation on {len(test_cases)} cases...")
    csv_data = []

    for i, case in enumerate(test_cases):
        question = case["question"]
        ground_truth = case["ground_truth"]
        
        print(f"\n[{i+1}/{len(test_cases)}] Case: {question}")
        
        # 1. RUN AGENT (Local)
        state = {"messages": [HumanMessage(content=question)]}
        current_config = {"configurable": {"thread_id": f"eval_{hash(question)}"}}
        result_state = None
        
        try:
            for event in app.stream(state, current_config, stream_mode="values"):
                result_state = event
                snapshot = app.get_state(current_config)
                if snapshot.next:
                    for sub_event in app.stream(None, current_config, stream_mode="values"):
                        result_state = sub_event
        except Exception as e:
            print(f"  [Error] Agent phase failed: {e}")
            continue
        
        if not result_state:
            continue

        messages = result_state["messages"]
        final_answer = messages[-1].content
        
        # Context Extraction
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
        
        # 2. RUN METRICS (Local)
        test_case = LLMTestCase(
            input=question,
            actual_output=final_answer,
            retrieval_context=retrieved_docs if retrieved_docs else ["No context retrieved"],
            expected_output=ground_truth
        )

        row = {
            "input": question,
            "actual_output": final_answer,
            "expected_output": ground_truth
        }

        for m in metrics:
            m_name = m.__class__.__name__.lower().replace("metric", "")
            print(f"    - Measuring {m_name}...")
            try:
                # Add tiny delay between metrics
                time.sleep(5)
                m.measure(test_case)
                row[f"{m_name}_score"] = m.score
                row[f"{m_name}_reason"] = getattr(m, 'reason', "N/A")
            except Exception as e:
                print(f"      [Metric Error] {e}")
                row[f"{m_name}_score"] = None
                row[f"{m_name}_error"] = str(e)
        
        csv_data.append(row)
        
        # Save intermediate results
        pd.DataFrame(csv_data).to_csv("deepeval_results.csv", index=False)
        
        # 3. COOL DOWN (To prevent connection resets)
        print("  Cooldown for 15 seconds...")
        time.sleep(15)

    print("\nRobust Evaluation complete. Results in deepeval_results.csv")

if __name__ == "__main__":
    run_evaluation()
