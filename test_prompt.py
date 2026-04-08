from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, HumanMessage
from graph import _base_llm, GUARDRAIL_PROMPT
llm = _base_llm()
prompt = SystemMessage(content=GUARDRAIL_PROMPT)
msg = HumanMessage(content="Analyze 'data/superstore_us_clean.csv' and store the result.")
ans = llm.invoke([prompt, msg])
print('Raw answer:', ans.content)
