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

```text
Getting tools for paper: 12_Optimizing_the_Landscape_of.pdf
Getting tools for paper: 34_The_Use_of_AI_Tools_to_Deve.pdf
Getting tools for paper: 18_Automated_Item_Generation_f.pdf
Number of tools: 6
Added user message to memory: Tell me about the evaluation dataset used in Optimizing_the_Landscape_of, and then tell me about the evaluation results
=== Calling Function ===
Calling function: vector_tool_12_Optimizing_the_Landscape_of with args: {"query": "evaluation dataset"}
=== Function Output ===
The provided information does not specify an evaluation dataset. It focuses on the methodology and results of embedding models and simulations, including metrics like Normalized Mutual Information (NMI) and Total Entropy Fit Index (TEFI), but does not mention any specific dataset used for evaluation.
=== Calling Function ===
Calling function: vector_tool_12_Optimizing_the_Landscape_of with args: {"query": "evaluation results"}
=== Function Output ===
The evaluation results indicate that optimized solutions consistently achieve higher accuracy across various item counts, particularly benefiting moderate to large item pools. The comparison between original and optimized methods shows that the optimized approach, which utilizes a dynamic embedding generation algorithm, leads to significant improvements in structural recovery. This suggests that the composite optimization strategy enhances the accuracy of dimensional organization in psychometric assessments. Additionally, the findings highlight the importance of semantic structure in embeddings as substantive validity evidence, reinforcing the notion that these structures are foundational to construct definition rather than mere approximations awaiting empirical validation.
=== LLM Response ===
The paper "Optimizing the Landscape of" does not specify a particular evaluation dataset. Instead, it focuses on the methodology and results of embedding models and simulations, utilizing metrics like Normalized Mutual Information (NMI) and Total Entropy Fit Index (TEFI) without mentioning any specific dataset used for evaluation.

Regarding the evaluation results, the findings indicate that optimized solutions consistently achieve higher accuracy across various item counts, especially benefiting moderate to large item pools. The comparison between original and optimized methods demonstrates that the optimized approach, which employs a dynamic embedding generation algorithm, leads to significant improvements in structural recovery. This suggests that the composite optimization strategy enhances the accuracy of dimensional organization in psychometric assessments. Furthermore, the results emphasize the importance of semantic structure in embeddings as substantive validity evidence, reinforcing the idea that these structures are foundational to construct definition rather than mere approximations awaiting empirical validation.
assistant: The paper "Optimizing the Landscape of" does not specify a particular evaluation dataset. Instead, it focuses on the methodology and results of embedding models and simulations, utilizing metrics like Normalized Mutual Information (NMI) and Total Entropy Fit Index (TEFI) without mentioning any specific dataset used for evaluation.

Regarding the evaluation results, the findings indicate that optimized solutions consistently achieve higher accuracy across various item counts, especially benefiting moderate to large item pools. The comparison between original and optimized methods demonstrates that the optimized approach, which employs a dynamic embedding generation algorithm, leads to significant improvements in structural recovery. This suggests that the composite optimization strategy enhances the accuracy of dimensional organization in psychometric assessments. Furthermore, the results emphasize the importance of semantic structure in embeddings as substantive validity evidence, reinforcing the idea that these structures are foundational to construct definition rather than mere approximations awaiting empirical validation.
Added user message to memory: Give me a summary of both Automated_Item_Generation_f and The_Use_of_AI_Tools_to_Deve
=== Calling Function ===
Calling function: summary_tool_18_Automated_Item_Generation_f with args: {"input": "Automated Item Generation (AIG) refers to the process of using algorithms and software to create assessment items automatically. This approach leverages various techniques, including natural language processing and machine learning, to generate questions and tasks that can be used in educational settings. The paper discusses the methodologies involved in AIG, the benefits of using automated systems for item generation, such as efficiency and scalability, and the challenges that come with ensuring the quality and validity of the generated items. It also explores the potential applications of AIG in different educational contexts and the future directions for research in this area."}
=== Function Output ===
Automated Item Generation (AIG) is a process that utilizes algorithms and software to automatically create assessment items. This method incorporates various techniques, including natural language processing and machine learning, to produce questions and tasks suitable for educational environments. The methodologies involved in AIG focus on structured prompt design, algorithmic filtering, and expert review to ensure the quality of the generated items. Benefits of AIG include increased efficiency and scalability in item development, addressing the growing demand for high-quality assessment items. However, challenges remain in maintaining the validity and quality of these items, particularly in aligning them with established educational frameworks. The potential applications of AIG span various educational contexts, and future research directions aim to refine these methodologies and explore their effectiveness across diverse settings.
=== Calling Function ===
Calling function: summary_tool_34_The_Use_of_AI_Tools_to_Deve with args: {"input": "The paper discusses the integration of AI tools in the development of educational resources and assessments. It highlights how AI can enhance the creation process by providing personalized learning experiences, automating repetitive tasks, and improving the quality of educational materials. The authors examine various AI technologies, including machine learning, natural language processing, and data analytics, and their applications in educational settings. The paper also addresses the ethical considerations and challenges associated with the use of AI in education, such as data privacy and the need for transparency in AI algorithms. Overall, it emphasizes the transformative potential of AI tools in shaping the future of education."}
=== Function Output ===
The study focuses on the application of AI tools specifically in the development and validation of Q-matrices for cognitive diagnostic modeling in educational assessments. It explores how generative AI can assist in creating Q-matrices, which are essential for mapping test items to the skills required for answering them. The research compares the performance of various AI models in generating Q-matrices for a reading comprehension test, assessing their agreement with a validated Q-matrix. While some AI tools demonstrated promising results, particularly in identifying comprehension skills, the study also highlights the variability in performance across different models and time points. Ethical considerations, such as the reliability of AI outputs and the importance of expert judgment in the Q-matrix construction process, are also discussed, emphasizing that AI should complement rather than replace human expertise in educational assessments.
=== LLM Response ===
### Summary of Automated Item Generation (AIG)

Automated Item Generation (AIG) is a process that utilizes algorithms and software to automatically create assessment items. This method incorporates various techniques, including natural language processing and machine learning, to produce questions and tasks suitable for educational environments. The methodologies involved in AIG focus on structured prompt design, algorithmic filtering, and expert review to ensure the quality of the generated items. Benefits of AIG include increased efficiency and scalability in item development, addressing the growing demand for high-quality assessment items. However, challenges remain in maintaining the validity and quality of these items, particularly in aligning them with established educational frameworks. The potential applications of AIG span various educational contexts, and future research directions aim to refine these methodologies and explore their effectiveness across diverse settings.

### Summary of The Use of AI Tools in Educational Development

The study focuses on the application of AI tools specifically in the development and validation of Q-matrices for cognitive diagnostic modeling in educational assessments. It explores how generative AI can assist in creating Q-matrices, which are essential for mapping test items to the skills required for answering them. The research compares the performance of various AI models in generating Q-matrices for a reading comprehension test, assessing their agreement with a validated Q-matrix. While some AI tools demonstrated promising results, particularly in identifying comprehension skills, the study also highlights the variability in performance across different models and time points. Ethical considerations, such as the reliability of AI outputs and the importance of expert judgment in the Q-matrix construction process, are also discussed, emphasizing that AI should complement rather than replace human expertise in educational assessments.
assistant: ### Summary of Automated Item Generation (AIG)

Automated Item Generation (AIG) is a process that utilizes algorithms and software to automatically create assessment items. This method incorporates various techniques, including natural language processing and machine learning, to produce questions and tasks suitable for educational environments. The methodologies involved in AIG focus on structured prompt design, algorithmic filtering, and expert review to ensure the quality of the generated items. Benefits of AIG include increased efficiency and scalability in item development, addressing the growing demand for high-quality assessment items. However, challenges remain in maintaining the validity and quality of these items, particularly in aligning them with established educational frameworks. The potential applications of AIG span various educational contexts, and future research directions aim to refine these methodologies and explore their effectiveness across diverse settings.

### Summary of The Use of AI Tools in Educational Development

The study focuses on the application of AI tools specifically in the development and validation of Q-matrices for cognitive diagnostic modeling in educational assessments. It explores how generative AI can assist in creating Q-matrices, which are essential for mapping test items to the skills required for answering them. The research compares the performance of various AI models in generating Q-matrices for a reading comprehension test, assessing their agreement with a validated Q-matrix. While some AI tools demonstrated promising results, particularly in identifying comprehension skills, the study also highlights the variability in performance across different models and time points. Ethical considerations, such as the reliability of AI outputs and the importance of expert judgment in the Q-matrix construction process, are also discussed, emphasizing that AI should complement rather than replace human expertise in educational assessments.

```

## RESULT
A multidocument retrieval agent was designed and implemented using LlamaIndex, with a vector tool and a summary tool for each of three research papers.


