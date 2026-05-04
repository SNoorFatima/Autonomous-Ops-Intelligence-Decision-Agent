# Automated Quality Gates & CI/CD Report

## 1. CI-Ready Evaluation Script
The original evaluation script (`eval_deepeval.py`) has been successfully adapted into a CI-ready script (`run_eval.py`). 
- **Headless Execution:** The new script runs completely headlessly. It handles graph streaming and automatically accepts interruptions (such as HITL tool checks) to ensure it does not block waiting for user input.
- **Exit Codes:** It guarantees an exit code of `0` when all DeepEval metrics pass according to their thresholds, and `1` if any metric fails. This enables CI platforms like GitHub Actions to properly mark the pipeline status.
- **Credential Management:** It no longer uses a hardcoded local LLM wrapper. Instead, it relies on `GROQ_API_KEY` mapped directly from the host environment variables, ensuring no sensitive credentials leak into the source code.
- **Machine-Readable Artifacts:** All execution results, including metric names, scores, threshold values, pass/fail status, and explanations, are written to a standard `eval_results.json` file.

## 2. Pipeline Configuration
A GitHub Actions workflow (`.github/workflows/main.yml`) has been established to automate the Quality Gate.
- **Trigger:** The workflow triggers on every push and pull request against the `main` branch.
- **Workflow:** It checks out the source code, provisions a Python 3.11 environment (with pip caching), and installs the required packages from `requirements.txt`.
- **Secret Store Integration:** The script accesses `GROQ_API_KEY` securely populated from GitHub Secrets (`${{ secrets.GROQ_API_KEY }}`), satisfying the requirement that no secret may appear in any committed file.
- **Surfacing Results:** The pipeline runs `python run_eval.py` and strictly relies on its exit code. Regardless of success or failure, the workflow archives the `eval_results.json` artifact for developer review.

## 3. Versioned Threshold Configuration
Metric threshold values are strictly version-controlled within `eval_thresholds.json`.

**Threshold Justification:**
- **Faithfulness (0.7):** We demand a minimum score of 0.7 to ensure that the agent's responses are derived strictly from the provided RAG context and database tools without hallucination. Setting this 10% higher (0.77) would likely cause excessive pipeline failures for minor, acceptable paraphrasing by the LLM. Setting it 10% lower (0.63) increases the risk of factual inaccuracies (hallucinations) slipping through into production.
- **Answer Relevancy (0.7):** We demand a score of 0.7 to ensure the agent directly answers the user's question without unnecessary digressions. Setting this 10% higher (0.77) risks failing perfectly valid responses that simply include extra helpful context. Setting it 10% lower (0.63) might allow the agent to ignore the core query entirely and output tangential information, degrading the user experience.

*Note: The breaking change demonstration will be verified via Walkthrough.*
