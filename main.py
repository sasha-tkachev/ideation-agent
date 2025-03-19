import os
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict 
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

PERPLEXITY_TO_XMIND ="""
You will be given a markdown output from perplexity.

Please take it and convert it into a xmind research tree.

The tree represents insights to a question.

For each insight create a single subnode with more detail about the insight. Do not add the prefix "detail"
For each insight with a reference. Create a subnode with the text "Why?" and to that node create a new subnode with the URL.
The why node should be added only to leafs on the graph.

Return the response in a code block so it can be copied into xMind.
Use tabs for indent instead of spaces.
Don't use xml.
Don't use empty newlines
"""


def parse_xmind_to_dict(content):
    """Convert XMind formatted string to OrderedDict
    
This is the format of the xmind tree:
Question1?
    Answer1
        Question1.1?
            Answer1.1
        Question1.2?
            
    Fact2
    Fact3
Question2?
    Answer2
Question3?
Should be converted to:
{
    "Question1": {
        "Answer1": {
            "Question1.1": {
                "Answer1.1": {}
            },
            "Question1.2": {
                "Answer1.2": {}
            }
        },
        "Fact2": {},
        "Fact3": {}
    },
    "Question2": {
        "Answer2": {}
    },
    "Question3": {}
}
"""
    lines = content.strip().split('\n')
    root = OrderedDict()
    stack = [(root, -1)]
    
    for line in lines:
        if line.strip() == "":
            continue
        indent = len(line) - len(line.lstrip()) 
        content = line.strip()
        
        while stack and indent <= stack[-1][1]:
            stack.pop()
            
        current_dict = stack[-1][0]
        new_dict = OrderedDict()
        current_dict[content] = new_dict
        stack.append((new_dict, indent))
        
    return root
    

def dict_to_xmind(d, level=0):
    """Convert OrderedDict back to XMind formatted string"""
    result = []
    for key, value in d.items():
        result.append('\t' * level + key)
        if value:
            result.extend(dict_to_xmind(value, level + 1))
    return result

def _convert_response_to_xmind_graph(response: str, openai_client: OpenAI) -> str:
    return openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": PERPLEXITY_TO_XMIND}, {"role": "user", "content": response}]
    ).choices[0].message.content.replace("```plaintext", "").strip().strip("`").strip()
    


def _answer_question(question: str, perplexity_client: OpenAI, openai_client: OpenAI) -> str:
    perplexity_response = perplexity_client.chat.completions.create(
        model="sonar",          
        messages=[{"role": "user", "content": question}]
    )
    citations = ""
    for i, citation in enumerate(perplexity_response.model_extra.get("citations", [])):
        citations += f"[{i+1}] {citation}\n"     
    result = perplexity_response.choices[0].message.content
    if citations:
        result += "\n\nCitations:\n" + citations
    return question, _convert_response_to_xmind_graph(result, openai_client)





def _answer_questions_in_parallel(questions: set[str], perplexity_client: OpenAI, openai_client: OpenAI) -> dict[str, str]:
    with ThreadPoolExecutor(max_workers=10) as executor:
        return list(executor.map(_answer_question, questions, [perplexity_client] * len(questions), [openai_client] * len(questions)))


def _find_all_questions(xmind_tree: dict, keys_so_far: tuple[str, ...] = tuple()) -> dict[str, tuple[str, ...]]:
    """
    Find all questions in the xmind tree, return the list of keys needed to find the question in the tree.
    """
    result = {}
    for key, value in xmind_tree.items():
        if key.endswith("?") and not value:
            result[key] = keys_so_far
        if value:
            result.update(_find_all_questions(value, keys_so_far + (key,)))
    return result

def _remove_first_layer(xmind_tree: dict) -> dict:
    if len(xmind_tree) == 1 and isinstance(xmind_tree, dict):
        return xmind_tree[list(xmind_tree.keys())[0]]
    return xmind_tree

def _answer_xmind_questions(xmind_tree: dict, perplexity_client: OpenAI, openai_client: OpenAI) -> dict:
    questions = _find_all_questions(xmind_tree)
    answers = _answer_questions_in_parallel(set(questions.keys()), perplexity_client, openai_client)
    for question, answer in answers:
        question_path = questions[question]
        current_node = xmind_tree
        for key in question_path:
            current_node = current_node[key]
        current_node[question] = _remove_first_layer(parse_xmind_to_dict(answer))
    return xmind_tree


def _research_xmind_tree(xmind_tree: str, perplexity_client: OpenAI, openai_client: OpenAI) -> str:
    result = _answer_xmind_questions(parse_xmind_to_dict(xmind_tree), perplexity_client, openai_client)
    return "\n".join(dict_to_xmind(result))

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

class ClipboardData(BaseModel):
    content: str

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

@app.post("/research")
async def research_clipboard(data: ClipboardData):
    try:
        perplexity_client = OpenAI(
            api_key=os.getenv("PERPLEXITY_API_KEY"), 
            base_url="https://api.perplexity.ai"
        )
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        result = _research_xmind_tree(
            data.content,
            perplexity_client,
            openai_client
        )
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 