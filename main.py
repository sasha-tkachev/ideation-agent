import os
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict 
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
from sse_starlette.sse import EventSourceResponse
import asyncio
import json

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
    


async def _answer_question(question: str, perplexity_client: OpenAI, openai_client: OpenAI) -> str:
    await send_progress(f"🔍 {question}")
    
    # Run OpenAI calls in thread pool
    loop = asyncio.get_event_loop()
    perplexity_response = await loop.run_in_executor(
        None,
        lambda: perplexity_client.chat.completions.create(
            model="sonar",          
            messages=[{"role": "user", "content": question}]
        )
    )
    
    citations = ""
    for i, citation in enumerate(perplexity_response.model_extra.get("citations", [])):
        citations += f"[{i+1}] {citation}\n"     
    result = perplexity_response.choices[0].message.content
    if citations:
        result += "\n\nCitations:\n" + citations
        
    xmind_result = await loop.run_in_executor(
        None,
        lambda: _convert_response_to_xmind_graph(result, openai_client)
    )
    
    await send_progress(f"✅ {question}")
    return question, xmind_result





async def _answer_questions_in_parallel(questions: set[str], perplexity_client: OpenAI, openai_client: OpenAI) -> dict[str, str]:
    return await asyncio.gather(*[
        _answer_question(q, perplexity_client, openai_client) 
        for q in questions
    ])


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

async def _answer_xmind_questions(xmind_tree: dict, perplexity_client: OpenAI, openai_client: OpenAI) -> dict:
    questions = _find_all_questions(xmind_tree)
    answers = await _answer_questions_in_parallel(set(questions.keys()), perplexity_client, openai_client)
    for question, answer in answers:
        question_path = questions[question]
        current_node = xmind_tree
        for key in question_path:
            current_node = current_node[key]
        current_node[question] = _remove_first_layer(parse_xmind_to_dict(answer))
    return xmind_tree


async def _research_xmind_tree(xmind_tree: str, perplexity_client: OpenAI, openai_client: OpenAI) -> str:
    result = await _answer_xmind_questions(parse_xmind_to_dict(xmind_tree), perplexity_client, openai_client)
    return "\n".join(dict_to_xmind(result))

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

class ClipboardData(BaseModel):
    content: str

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

# Add these global variables
progress_updates = asyncio.Queue()
active_connections = set()

# Add these new functions
async def send_progress(message: str, type: str = "progress"):
    await progress_updates.put({"message": message, "type": type})

async def progress_generator():
    try:
        while True:
            if progress_updates.empty():
                await asyncio.sleep(0.1)
                continue
            
            data = await progress_updates.get()
            yield {
                "event": "message",
                "data": json.dumps(data)
            }
    except asyncio.CancelledError:
        raise

@app.post("/research")
async def research_clipboard(data: ClipboardData):
    try:
        if not any(line.strip().endswith('?') for line in data.content.split('\n')):
            await send_progress(f"Got {repr(data.content)}")
            raise HTTPException(
                status_code=400,
                detail="The XMind tree must contain at least one question (line ending with '?')"
            )

        # Parse the tree and find questions
        tree = parse_xmind_to_dict(data.content)
        questions = _find_all_questions(tree)
        
        await send_progress(f"<code>{data.content}</code>")
        # Send progress updates
        await send_progress(f"Found {len(questions)} questions to research")
        await send_progress("Starting research process...")

        perplexity_client = OpenAI(
            api_key=os.getenv("PERPLEXITY_API_KEY"),
            base_url="https://api.perplexity.ai"
        )
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        result = await _research_xmind_tree(
            data.content,
            perplexity_client,
            openai_client
        )
        await send_progress(f"<code>{result}</code>")
        await send_progress("Research completed!", type="complete")
        return {"result": result}
    except Exception as e:
        await send_progress(f"Error: {str(e)}", type="error")
        raise HTTPException(status_code=500, detail=str(e))

# Add this new endpoint
@app.get('/progress')
async def progress_endpoint():
    return EventSourceResponse(progress_generator())


@app.get("/favicon.ico")
async def favicon():
    return FileResponse("static/favicon.ico")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 