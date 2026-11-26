# Design and Implementation of a Multidocument Retrieval Agent Using LlamaIndex

## AIM
To design and implement a multidocument retrieval agent using LlamaIndex to extract and synthesize information from multiple research articles, and to evaluate its performance by testing it with diverse queries, analyzing its ability to deliver concise, relevant, and accurate responses.

## PROBLEM STATEMENT
Finding and comparing information across several research articles by hand is slow and error-prone. The task is to build an agent that accepts a question in natural language, chooses the right document and retrieval method, and returns an answer based on the articles. It must handle specific factual questions, summaries, and questions that involve more than one paper.

## DESIGN STEPS

### STEP 1: Set up the environment and documents
Load the OpenAI API key, apply `nest_asyncio` for asynchronous calls in the notebook, and place the three research papers in the working folder.

### STEP 2: Build two tools for each paper
Use `get_doc_tools` (from `utils.py`) to create:
- a **vector tool** for specific questions about a paper
- a **summary tool** for summarizing a whole paper

Three papers give six tools.

### STEP 3: Create the agent and test it
Pass the six tools to a `FunctionCallingAgentWorker`, run it with an `AgentRunner`, and ask different types of questions. The agent decides which tools to call. Check each answer against the source paper.

## PROGRAM

Requires `helper.py`, `utils.py`, the three PDFs and an OpenAI API key.

```python
# Register gpt-4o-mini with the older llama-index-llms-openai package
from llama_index.llms.openai import utils as openai_utils

openai_utils.ALL_AVAILABLE_MODELS["gpt-4o-mini"] = 128000
openai_utils.CHAT_MODELS["gpt-4o-mini"] = 128000

from pathlib import Path

import nest_asyncio
from helper import get_openai_api_key

OPENAI_API_KEY = get_openai_api_key()
nest_asyncio.apply()

papers = [
    "12_Optimizing_the_Landscape_of.pdf",
    "34_The_Use_of_AI_Tools_to_Deve.pdf",
    "18_Automated_Item_Generation_f.pdf",
]

missing = [p for p in papers if not Path(p).exists()]
if missing:
    raise FileNotFoundError(f"Missing PDF files: {missing}")

from utils import get_doc_tools

paper_to_tools_dict = {}
for paper in papers:
    print(f"Getting tools for paper: {paper}")
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]

initial_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]
print(f"Number of tools: {len(initial_tools)}")

from llama_index.llms.openai import OpenAI
from llama_index.core.agent import FunctionCallingAgentWorker, AgentRunner

llm = OpenAI(model="gpt-4o-mini")

agent_worker = FunctionCallingAgentWorker.from_tools(
    initial_tools,
    llm=llm,
    verbose=True,
)
agent = AgentRunner(agent_worker)

queries = [
    # specific facts
    "Tell me about the evaluation dataset used in Optimizing_the_Landscape_of, "
    "and then tell me about the evaluation results",
    # summaries of two papers
    "Give me a summary of both Automated_Item_Generation_f "
    "and The_Use_of_AI_Tools_to_Deve",
    # comparison
    "Compare how Automated_Item_Generation_f and The_Use_of_AI_Tools_to_Deve "
    "use AI",
]

for query in queries:
    print(f"\nQUERY: {query}")
    response = agent.query(query)
    print(str(response))
```

## OUTPUT

Tool creation (from an actual run):

```
Getting tools for paper: 12_Optimizing_the_Landscape_of.pdf
Getting tools for paper: 34_The_Use_of_AI_Tools_to_Deve.pdf
Getting tools for paper: 18_Automated_Item_Generation_f.pdf
Number of tools: 6
```

Agent responses:

> _Paste the output of the three queries here after running the program._

Evaluation (compare each answer with the paper):

| Query type | Correct tool used? | Concise? | Relevant? | Accurate? |
|---|---|---|---|---|
| Specific facts | | | | |
| Summaries | | | | |
| Comparison | | | | |

## RESULT
A multidocument retrieval agent was designed and implemented using LlamaIndex, with a vector tool and a summary tool for each of three research papers.

> _After testing, add one or two sentences on how concise, relevant and accurate the responses were._
