from langgraph_engineer.loader import load_github_file
from langgraph_engineer.model import _get_model
from langgraph_engineer.state import AgentState
from langchain_core.messages import AIMessage
from langchain_core.pydantic_v1 import BaseModel

critique_prompt = """You are tasked with critiquing a junior developer's first attempt at building a LangGraph application. \
Here is a long unit test file for LangGraph. This should contain a lot (but possibly not all) \
relevant information on how to use LangGraph.

<unit_test_file>
{file}
</unit_test_file>

Based on the conversation below, attempt to critique the developer. If it seems like the written solution is fine, then provide a JSON object conforming to the following schema:

{{ "logic": "<string reasoning>", "accept": true or false }}


If you accept, ensure "accept" is true. Otherwise, set "accept" to false.
Do not include extra text outside the JSON.
"""

class Accept(BaseModel):
    logic: str
    accept: bool

def _swap_messages(messages):
    new_messages = []
    for m in messages:
        if isinstance(m, AIMessage):
            new_messages.append({"role": "assistant", "content": m.content})
        else:
            new_messages.append({"role": "user", "content": m.content})
    return new_messages

def critique(state: AgentState, config):
    github_url = "https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/tests/test_pregel.py"
    file_contents = load_github_file(github_url)
    messages = [
        {"role": "user", "content": critique_prompt.format(file=file_contents)},
        {"role": "assistant", "content": state.get('requirements')}
    ] + _swap_messages(state['messages'])

    model = _get_model(config, "gemini", "critique_model").with_structured_output(Accept)
    response = model.invoke(messages)
    accepted = response.accept

    if accepted:
        return {
            "messages": [
                {"role": "user", "content": response.logic},
                {"role": "assistant", "content": "okay, sending to user"}
            ],
            "accepted": True
        }
    else:
        return {
            "messages": [
                {"role": "user", "content": response.logic}
            ],
            "accepted": False
        }
