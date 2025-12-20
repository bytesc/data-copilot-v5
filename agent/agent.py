from agent.utils.json_query_t import run_opensearch_demo
from agent.utils.llm_access.LLM import get_llm
from agent.utils.llm_access.call_llm import call_llm

llm = get_llm()

def query_agent(question, tables=None,  retries=2):

    run_opensearch_demo()

    ans = call_llm(question, llm).content

    return ans


