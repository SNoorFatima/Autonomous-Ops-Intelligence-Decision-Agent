# PRD — Autonomous Business Intelligence & Decision Agent (ABIDA)

> **Course/Lab:** Agentic Architecture (LangGraph-focused)  
> **Deliverable:** PRD.md + Architecture_Diagram.png  
> **Industry Vertical:** **Supply Chain / Operations Analytics**

## 1) Problem Statement (Bottleneck)
Operations teams (warehouse, delivery, procurement) receive data from multiple sources (shipment logs, inventory sheets, vendor lead times, incident notes). Today, analysis is manual and fragmented:

- **Multiple steps** are required: cleaning messy CSV/Excel, merging sources, computing KPIs, investigating anomalies, and writing reports.
- A single LLM prompt cannot reliably do this end-to-end because it requires **tool use**, **multi-step planning**, **validation**, and **action execution**.

**ABIDA** is an agentic system that continuously turns raw ops data into decision-ready insights and actions:
- Detects delays, stock-out risks, and abnormal cost spikes
- Explains likely root causes
- Generates reports/dashboards
- Sends alerts and creates follow-up tasks

## 2) Use Case Selection (Why it needs an Agent)
**Use Case:** “Weekly Operations Review + Exception Handling”  
Given the latest datasets, ABIDA must:
1. Ingest multiple files (orders, inventory, shipments, vendor SLAs)
2. Clean and normalize (types, missing values, duplicates)
3. Compute KPIs (OTD, lead-time, fill-rate, stock-out risk)
4. Detect anomalies (late lanes, cost outliers, underperforming vendors)
5. Compare with previous weeks (memory)
6. Produce an executive report + charts
7. Execute actions (send alerts/email/Slack; create a ticket/task)

This is inherently **multi-step** and requires **planning + tools + validation**, so it cannot be solved with a single response.

## 3) User Personas
### Primary Persona — Operations Manager
- Needs quick, trustworthy insights and prioritized actions
- Cares about OTD, backlogs, vendor performance, cost spikes

### Secondary Persona — Data Analyst (Ops)
- Wants automated cleaning + reproducible analysis runs
- Needs exportable charts and a consistent report structure

### Secondary Persona — Procurement / Vendor Manager
- Wants vendor-level KPIs, SLA breaches, and negotiation evidence

## 4) Success Metrics
### Accuracy & Reliability
- **KPI correctness:** ≥ 95% match against a reference pipeline on test datasets
- **Tool success rate:** ≥ 98% successful tool calls per run (no crash / silent failure)
- **Hallucination rate:** ≤ 2% unsupported claims in reports (checked via source references)

### Business Impact
- **Time saved:** ≥ 60% reduction in manual weekly reporting time
- **Issue detection:** Detect ≥ 80% of labeled incidents (late shipments / stock-outs) in evaluation set
- **Actionability:** ≥ 70% of runs produce at least 1 concrete recommended action

## 5) Tool & Data Inventory (External World)
### Knowledge Sources (Grounding)
- **Structured data:** CSV/Excel (orders, shipments, inventory, vendor SLAs)
- **Optional DB:** SQL (Postgres/MySQL) for historical ops tables
- **Unstructured docs:** Policy PDFs / SOPs (e.g., escalation rules, SLA definitions)
- **Optional wiki:** Confluence/Notion pages with process notes

### Action Tools (Python/APIs)
**Core Python tools**
- `load_dataset(file_path)`
- `profile_schema(df)`
- `clean_data(df, rules)`
- `join_sources(dfs, keys)`
- `compute_kpis(df)`
- `detect_anomalies(df)`
- `generate_charts(df, kpis)`
- `generate_report_pdf(results)` / `generate_report_ppt(results)`
- `store_run_memory(embedding, metadata)` / `retrieve_similar_runs(query)`

**External integrations (optional but recommended)**
- `send_email(to, subject, attachment)` (SMTP/Gmail API)
- `post_slack(channel, message, attachment)` (Slack API)
- `create_ticket(title, description)` (Jira API) or `update_notion_page(page_id, content)`

## 6) LangGraph-Orchestrated Agent Design (High Level)
ABIDA is implemented as a **LangGraph** workflow with:
- **State**: datasets, inferred schema, KPI results, anomalies, citations, action plan
- **Routing**: which agent runs next based on state and checks
- **Retries/Guardrails**: validate outputs before moving forward

### Agent Roles (Nodes)
- **Planner Agent**: decomposes goal into steps, selects tools, defines checks
- **Data Cleaning Agent**: fixes types, missing data, duplicates, joins sources
- **Analysis Agent**: KPIs, anomaly detection, segmentation, comparisons
- **Visualization Agent**: charts/dashboard artifacts + captions
- **Report Agent**: executive summary + recommendations + export
- **Memory Agent**: stores/retrieves historical runs for comparisons

## 7) Scope (MVP vs Extensions)
### MVP (Must-have)
- Multi-file ingestion (CSV/Excel)
- Cleaning + KPI computation
- Anomaly detection + explanations
- PDF report + chart export
- Run memory (store + compare last run)
- Email alert (basic)

### Extensions (Nice-to-have)
- Full Slack/Jira/Notion integration
- Real-time monitoring (scheduled runs)
- Forecasting and what-if simulations
- Role-based dashboards

## 8) Risks & Mitigations
- **Messy data breaks pipeline** → strict schema inference + cleaning rules + fallback prompts
- **Incorrect insights** → KPI validation checks + citations + “confidence” + human review mode
- **Tool failures** → retries + safe defaults + error logs per step
- **Privacy** → local processing, redaction, access control for exports

---
**Repo Checklist**
- `PRD.md` (this file)
- `Architecture_Diagram.png` (diagram of LangGraph system)
