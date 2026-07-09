---
title: LangChain Common Practices
date: 2024-12-25 05:04:50
tags: knowledge
---

This is a collection of some common useful LangChain (v.0.3.3) practices based on my coding experience so far. 

## LLM Application Development Landscape

Nowadays, LLM applications can be classified into the following categories.

1. **Simple LLM Calls**

   Applications where LLMs are used directly to answer questions or perform tasks without additional layers of complexity. The focus is on generating responses to prompts or queries. These are straightforward implementations, often used for tasks like content generation, question answering, or summarization.

   Real-world examples:

   * ChatGPT for Q&A: Users input questions, and the model directly generates answers.
   * Copywriting Tools: Applications like Jasper AI create marketing content, blogs, or product descriptions based on user inputs.

2. **Vectorstores (RAG)**

   Vectorstores are used in Retrieval-Augmented Generation (RAG) applications, where relevant information is retrieved from a database of embeddings (vectorized representations of text) to enhance the LLM's responses. This allows the LLM to work with domain-specific or proprietary knowledge not contained in its training data.

   Real-world examples:

   * Chatbots for Enterprises: A customer support chatbot retrieves relevant product documentation or FAQs stored in a vectorstore to provide accurate responses.
   * Search-Augmented Systems: Google Bard integrates real-time information retrieval to provide up-to-date and contextually relevant responses.

3. **Agents**

   Agents are LLM-driven systems that execute tasks autonomously or semi-autonomously based on input instructions. They can make decisions, interact with APIs, and manage workflows. Agents often use reasoning frameworks like ReAct (Reasoning and Acting) to decide what steps to take next.

   Real-world examples:

   - Zapier AI Assistant: Automates workflows by taking instructions, analyzing data, and executing API calls or actions across platforms.
   - LangChain Agents: Used for multi-step tasks such as filling out forms, managing databases, or performing calculations.

4. **Agents + Vectorstores**

   This combines the reasoning and decision-making capabilities of agents with the data retrieval abilities of vectorstores. These systems can autonomously fetch relevant knowledge from vectorstores and execute tasks, enabling advanced applications like AutoGPT. The integration provides both reasoning depth and domain-specific accuracy.

   Real-world examples:

   - **AutoGPT:** An open-source agent that can generate business plans by researching topics, retrieving relevant information, and autonomously completing subtasks.
   - **GPT Engineer:** Helps developers by retrieving relevant programming resources and autonomously generating code, debugging, or improving software projects.

## Chaining a Simple Prompt

```python
from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# define information to be incorporated to the prompt template
information = """
    Elon Reeve Musk (/ˈiːlɒn/; EE-lon; born June 28, 1971) is a businessman and investor. 
    He is the founder, chairman, CEO, and CTO of SpaceX; angel investor, CEO, product architect and former chairman of Tesla, Inc.; owner, chairman and CTO of X Corp.; founder of the Boring Company and xAI; co-founder of Neuralink and OpenAI; and president of the Musk Foundation. 
    He is the wealthiest person in the world, with an estimated net worth of US$232 billion as of December 2023, according to the Bloomberg Billionaires Index, and $254 billion according to Forbes, primarily from his ownership stakes in Tesla and SpaceX.
    A member of the wealthy South African Musk family, Elon was born in Pretoria and briefly attended the University of Pretoria before immigrating to Canada at age 18, acquiring citizenship through his Canadian-born mother. 
    Two years later, he matriculated at Queen's University at Kingston in Canada. Musk later transferred to the University of Pennsylvania, and received bachelor's degrees in economics and physics. 
    He moved to California in 1995 to attend Stanford University. However, Musk dropped out after two days and, with his brother Kimbal, co-founded online city guide software company Zip2. 
    The startup was acquired by Compaq for $307 million in 1999, and, that same year Musk co-founded X.com, a direct bank. X.com merged with Confinity in 2000 to form PayPal.
    In October 2002, eBay acquired PayPal for $1.5 billion, and that same year, with $100 million of the money he made, Musk founded SpaceX, a spaceflight services company. 
    In 2004, he became an early investor in electric vehicle manufacturer Tesla Motors, Inc. (now Tesla, Inc.). He became its chairman and product architect, assuming the position of CEO in 2008. 
    In 2006, Musk helped create SolarCity, a solar-energy company that was acquired by Tesla in 2016 and became Tesla Energy. In 2013, he proposed a hyperloop high-speed vactrain transportation system. 
    In 2015, he co-founded OpenAI, a nonprofit artificial intelligence research company. 
    The following year, Musk co-founded Neuralink—a neurotechnology company developing brain–computer interfaces—and the Boring Company, a tunnel construction company. 
    In 2022, he acquired Twitter for $44 billion. He subsequently merged the company into newly created X Corp. and rebranded the service as X the following year. 
    In March 2023, he founded xAI, an artificial intelligence company.
"""

# create a prompt template
template = """
Given the information {information} about a person, please create:
1. A short summary
2. Two interesting facts about the person.
"""

# incorporate information into prompt
summary_prompt_template = PromptTemplate(
    input_variables=["information"], template=template
)

# create an llm
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
# llm = ChatOllama(model="llama3")

# create a chain
chain = summary_prompt_template | llm | StrOutputParser()

# prompt the model
response = chain.invoke(input={"information": information})

print(response)

```

## Parsing the Output with a Customized Format

Using `PydanticOutputParser` and user defined output data structure.

```python
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain_openai import OpenAI

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")


# define your desired data structure
class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")

    # add custom validation logic easily with Pydantic
    @validator("setup")
    def question_ends_with_question_mark(cls, field):
        if field[-1] != "?":
            raise ValueError("Badly formed question!")
        return field


# set up a parser + inject instructions into the prompt template.
parser = PydanticOutputParser(pydantic_object=Joke)

# create a prompt with query and instruction
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

#a query intended to prompt a language model to populate the data structure
prompt_and_model = prompt | llm
output = prompt_and_model.invoke({"query": "Tell me a joke."})
parser.invoke(output)
```

## Data Ingestion to Pinecone Vectorstore (RAG)

Using `TextLoader`, `CharacterTextSplitter`, `OpenAIEmbeddings`, and [Pinecone](https://app.pinecone.io/) vector database.

Please refer to [LangChain text splitter techniques](https://python.langchain.com/docs/how_to/#text-splitters) ;[text Split by character](https://python.langchain.com/docs/how_to/character_text_splitter/); [text embedding models](https://python.langchain.com/docs/how_to/embed_text/) for more details.

```python
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# load data
loader = TextLoader("doc1.txt")
document = loader.load()

# split data
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(document)

# create embedding
embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

# ingest data to vector db
PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ['INDEX_NAME'])print("finish")

```

## Data Retrieval from Pinecone Vectorestore (RAG)

`langchain-ai/retrieval-qa-chat` is a [ChatPromptTemplate](https://smith.langchain.com/hub/langchain-ai/retrieval-qa-chat?organizationId=9f3e510d-ce6b-4f98-98a3-ff383fda2d96) ensuring answers are based solely on the context. 

```python
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI()

# build user query prompt
query = "what is Pinecone in machine learning?"
chain = PromptTemplate.from_template(template=query) | llm

# store query prompt to vector db
vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], embedding=embeddings
)

# create a retrieval qa prompt
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

# create a prompt chain
combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

# create retrieval chain
retrival_chain = create_retrieval_chain(
    retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
)

# execute the retrieval
result = retrival_chain.invoke(input={"input": query})

print(result)

```

Customized retrieval prompt:

`RunnablePassthrough` is used to pass through arguments [from one step to the next](https://python.langchain.com/docs/how_to/passthrough/). It allows us to pass on the user's question to the prompt and model. 

```python
import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI()

query = "what is Pinecone in machine learning?"
chain = PromptTemplate.from_template(template=query) | llm

vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], embedding=embeddings
)


template = """
Use the following pieces of context to answer the question at the end. 
If you don't know the answer, you can say "I don't know". Don't make up an answer.
Use three sentences maximum and keep the answer short and to the point.
Always say "thanks for the question" before answering the question.

{context}

Question: {question}

Answer:
"""

custom_rag_prompt = PromptTemplate.from_template(template=template)
rag_chain = (
    {"context": vectorstore.as_retriever() | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
)

res = rag_chain.invoke(query)
print(res)

```

## Chat with a PDF (RAG with FAISS)

Using `PyPDFLoader`, `CharacterTextSplitter`, `OpenAIEmbeddings`, and `FAISS` local vector database.

Please refer to [PDF loader](https://python.langchain.com/docs/how_to/document_loader_pdf/) ; [Langchain FAISS vectorstore](https://python.langchain.com/docs/integrations/vectorstores/faiss/); [FAISS](https://faiss.ai/) for more details.

```python
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub

pdf_path = "react.pdf"
loader = PyPDFLoader(file_path=pdf_path)
documents = loader.load()
text_splitter = CharacterTextSplitter(
    chunk_size=1000, chunk_overlap=30, separator="\n"
)
docs = text_splitter.split_documents(documents=documents)

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local("faiss_index_react")

new_vectorstore = FAISS.load_local(
    "faiss_index_react", embeddings, allow_dangerous_deserialization=True
)

retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
combine_docs_chain = create_stuff_documents_chain(
    OpenAI(), retrieval_qa_chat_prompt
)
retrieval_chain = create_retrieval_chain(
    new_vectorstore.as_retriever(), combine_docs_chain
)

res = retrieval_chain.invoke({"input": "Give me the gist of ReAct in 3 sentences"})
print(res["answer"])

```

## Create a ReAct Agent

Using `langchain-ai/react-agent-template` to build a [ReAct prompt](https://smith.langchain.com/hub/langchain-ai/react-agent-template?organizationId=9f3e510d-ce6b-4f98-98a3-ff383fda2d96).

```python
from dotenv import load_dotenv
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonREPLTool

load_dotenv()

# create an instruction
instructions = """You are an agent designed to write and execute python code to answer questions.
You have access to a python REPL, which you can use to execute python code.
If you get an error, debug your code and try again.
Only use the output of your code to answer the question. 
You might know the answer without running any code, but you should still run the code to get the answer.
If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
"""

# use an ReAct prompt template
base_prompt = hub.pull("langchain-ai/react-agent-template")
prompt = base_prompt.partial(instructions=instructions)

# make use a tool to execute python code
tools = [PythonREPLTool()]

# define a ReAct agent
agent = create_react_agent(
    prompt=prompt,
    llm=ChatOpenAI(temperature=0, model="gpt-4o-mini"),
    tools=tools,
)

# create the ReAct agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# execute the ReAct agent
agent_executor.invoke(
    input={
        "input": """generate and save in current working directory a QRcode
                    that point to https://jokerdii.github.io/di-blog, you have qrcode package installed already"""
    }
)

agent_executor.invoke(
    input={
        "input": """generate and save in current working directory a synthetic csv dataset 
                    with 1000 rows and 2 columns that is about Amazon product description and price."""
    }
)

```

## Using an LangChain Agent for Tasks

`create_csv_agent` is an `AgentExecutor` object able to perform operations in CSVs.

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_csv_agent


load_dotenv()

# make use a CSV agent from langchain
csv_agent = create_csv_agent(
    llm=ChatOpenAI(temperature=0, model="gpt-4o-mini"),
    path="episode_info.csv",
    verbose=True,
)

# execute the agent
csv_agent.invoke(
    input={"input": "how many columns are there in file episode_info.csv"}
)
csv_agent.invoke(
    input={
        "input": "print the seasons by ascending order of the number of episodes they have."
    }
)

```

## Creating an ReAct Agent with Multiple Agents Provided as Tools

```python
from typing import Any

from dotenv import load_dotenv
from langchain import hub
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.agents import (
    create_react_agent,
    AgentExecutor,
)
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.agents.agent_toolkits import create_csv_agent


load_dotenv()


instructions = """You are an agent designed to write and execute python code to answer questions.
You have access to a python REPL, which you can use to execute python code.
You have qrcode package installed
If you get an error, debug your code and try again.
Only use the output of your code to answer the question. 
You might know the answer without running any code, but you should still run the code to get the answer.
If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
    """

base_prompt = hub.pull("langchain-ai/react-agent-template")
prompt = base_prompt.partial(instructions=instructions)

# define python agent
tools = [PythonREPLTool()]
python_agent = create_react_agent(
    prompt=prompt,
    llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"),
    tools=tools,
)

python_agent_executor = AgentExecutor(agent=python_agent, tools=tools, verbose=True)

# define CSV agent
csv_agent_executor: AgentExecutor = create_csv_agent(
    llm=ChatOpenAI(temperature=0, model="gpt-4"),
    path="episode_info.csv",
    verbose=True,
)

#### router grand agent

# list agent tools
def python_agent_executor_wrapper(original_prompt: str) -> dict[str, Any]:
    return python_agent_executor.invoke({"input": original_prompt})

tools = [
    Tool(
        name="Python Agent",
        func=python_agent_executor_wrapper,
        description="""useful when you need to transform natural language to python and execute the python code,
                        returning the results of the code execution
                        DOES NOT ACCEPT CODE AS INPUT""",
    ),
    Tool(
        name="CSV Agent",
        func=csv_agent_executor.invoke,
        description="""useful when you need to answer question over episode_info.csv file,
                        takes an input the entire question and returns the answer after running pandas calculations""",
    ),
]

# create grand ReAct agent
prompt = base_prompt.partial(instructions="")
grand_agent = create_react_agent(
    prompt=prompt,
    llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"),
    tools=tools,
)
grand_agent_executor = AgentExecutor(agent=grand_agent, tools=tools, verbose=True)

# execute grand ReAct agent and print output
print(
    grand_agent_executor.invoke(
        {
            "input": "which season has the most episodes?",
        }
    )
)

print(
    grand_agent_executor.invoke(
        {
            "input": "Generate and save in current working directory 15 qrcodes that point to `www.udemy.com/course/langchain`",
        }
    )
)

```

## Function / Tool Calling 

LangChain provides a standardized interface for connecting tools to models. 

* `ChatModel.bind_tools()`: a method for attaching tool definitions to model calls.
* `AIMessage.tool_calls`: an attribute on the `AIMessage` returned from the model for easily accessing the tool aclls the model decided to make.
* `create_tool_calling_agent`: an agent constsructor that works with ANY model that implements `bind_tools` and returns `tool_calls`.

Directly using `PythonREPLTool` which is already a tool object. Use with caution because [Python REPL](https://python.langchain.com/docs/integrations/tools/python/) can execute arbitrary code on the host machine (e.g., delete files, make network requests). 

```python
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()


@tool
def multiply(x: float, y: float) -> float:
    """Multiply 'x' times 'y'."""
    return x * y


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "you're a helpful assistant"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

tools = [TavilySearchResults(), multiply]
llm = ChatOpenAI(model="gpt-4o-mini")

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

response = agent_executor.invoke(
    {
        "input": "what is the weather in dubai right now? compare it with San Fransisco, output should in in celsious",
    }
)

print(response)

```

## Tool Calling with Langchain

```python
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

llm_with_tools = tool_calling_model.bind_tools([multiply])

result = llm_with_tools.invoke("What is 2 multiplied by 3?")
```

## Token Limitation Handling Strategies

when passing documents into the LLM context window, there are three approaches for handling context window limitations:

1. [Stuffing](https://python.langchain.com/docs/tutorials/summarization/): suff all documents into a single prompt

```python
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

# define prompt
prompt_template = """Write a concise summary of the following:
"{text}"
CONCISE SUMMARY:"""
prompt = PromptTemplate.from_template(prompt_template)

# define LLM chain
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
llm_chain = LLMChain(llm=llm, prompt=prompt)

# define StuffDocumentsChain
stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
docs = loader.load()
print(stuff_chain.run(docs))
```

2. [Map-reduce](https://python.langchain.com/docs/tutorials/summarization/): summarize each document on its own in parallel and put them into a final summary.

```python
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain_text_splitters import CharacterTextSplitter

# Map
map_template = """The following is a set of documents
{docs}
Based on this list of docs, please identify the main themes 
Helpful Answer:"""
map_prompt = PromptTemplate.from_template(map_template)
map_chain = LLMChain(llm=llm, prompt=map_prompt)
# Reduce
reduce_template = """The following is set of summaries:
{docs}
Take these and distill it into a final, consolidated summary of the main themes. 
Helpful Answer:"""
reduce_prompt = PromptTemplate.from_template(reduce_template)
reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)
# Combine documents by mapping a chain over them, then combining results
map_reduce_chain = MapReduceDocumentsChain(
    llm_chain=map_chain,
    reduce_documents_chain=reduce_documents_chain,
    document_variable_name="docs",
    return_intermediate_steps=False,
)
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=0)
split_docs = text_splitter.split_documents(docs)
print(map_reduce_chain.run(split_docs))
```

3. Refine: The refine documents chain constructs a response by looping over the input documents and iteratively updating its answer. 

```python
from langchain.chains.summarize import load_summarize_chain
prompt = """
                  Please provide a summary of the following text.
                  TEXT: {text}
                  SUMMARY:
                  """

question_prompt = PromptTemplate(
    template=question_prompt_template, input_variables=["text"]
)

refine_prompt_template = """
              Write a concise summary of the following text delimited by triple backquotes.
              Return your response in bullet points which covers the key points of the text.
              ```{text}```
              BULLET POINT SUMMARY:
              """

refine_template = PromptTemplate(
    template=refine_prompt_template, input_variables=["text"]

# load refine chain
chain = load_summarize_chain(
    llm=llm,
    chain_type="refine",
    question_prompt=question_prompt,
    refine_prompt=refine_prompt,
    return_intermediate_steps=True,
    input_key="input_documents",
    output_key="output_text",
)
result = chain({"input_documents": split_docs}, return_only_outputs=True)
```

## Coreference Resolution

[Adding memory to chatbots.](https://python.langchain.com/docs/how_to/chatbots_memory/)

LangChain provides a way to build applications that have memory using LangGraph's [persistence](https://langchain-ai.github.io/langgraph/concepts/persistence/). You can [enable persistence](https://langchain-ai.github.io/langgraph/how-tos/persistence/) in LangGraph applications by providing a `checkpointer` when compiling the graph. Every iteration,  LangGraph takes the information and saves it in a DB (PostgreSQL, MySQL, Redis, and MongoDB saver). 

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

workflow = StateGraph(state_schema=MessagesState)

# define the function that calls the model
def call_model(state: MessagesState):
    system_prompt = (
        "You are a helpful assistant. "
        "Answer all questions to the best of your ability."
    )
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = model.invoke(messages)
    return {"messages": response}


# define the node and edge
workflow.add_node("model", call_model)
workflow.add_edge(START, "model")

# add simple in-memory checkpointer
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
```

Langchain has three main strategies to manage state:

1. Simply stuffing previous messages into a chat model prompt.

```python
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="You are a helpful assistant. Answer all questions to the best of your ability."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | model

ai_msg = chain.invoke(
    {
        "messages": [
            HumanMessage(
                content="Translate from English to French: I love programming."
            ),
            AIMessage(content="J'adore la programmation."),
            HumanMessage(content="What did you just say?"),
        ],
    }
)
print(ai_msg.content)
```

2. The above, but trimming old messages to reduce the amount of distracting information the model has to deal with.

```python
from langchain_core.messages import trim_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

# define trimmer
# count each message as 1 "token" (token_counter=len) and keep only the last two messages
trimmer = trim_messages(strategy="last", max_tokens=2, token_counter=len)

workflow = StateGraph(state_schema=MessagesState)


# define the function that calls the model
def call_model(state: MessagesState):
    trimmed_messages = trimmer.invoke(state["messages"])
    system_prompt = (
        "You are a helpful assistant. "
        "Answer all questions to the best of your ability."
    )
    messages = [SystemMessage(content=system_prompt)] + trimmed_messages
    response = model.invoke(messages)
    return {"messages": response}


# define the node and edge
workflow.add_node("model", call_model)
workflow.add_edge(START, "model")

# ddd simple in-memory checkpointer
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
```

3. More complex modifications like synthesizing summaries for long running conversations.

```python
from langchain_core.messages import HumanMessage, RemoveMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

workflow = StateGraph(state_schema=MessagesState)


# Define the function that calls the model
def call_model(state: MessagesState):
    system_prompt = (
        "You are a helpful assistant. "
        "Answer all questions to the best of your ability. "
        "The provided chat history includes a summary of the earlier conversation."
    )
    system_message = SystemMessage(content=system_prompt)
    message_history = state["messages"][:-1]  # exclude the most recent user input
    # Summarize the messages if the chat history reaches a certain size
    if len(message_history) >= 4:
        last_human_message = state["messages"][-1]
        # Invoke the model to generate conversation summary
        summary_prompt = (
            "Distill the above chat messages into a single summary message. "
            "Include as many specific details as you can."
        )
        summary_message = model.invoke(
            message_history + [HumanMessage(content=summary_prompt)]
        )

        # Delete messages that we no longer want to show up
        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"]]
        # Re-add user message
        human_message = HumanMessage(content=last_human_message.content)
        # Call the model with summary & response
        response = model.invoke([system_message, summary_message, human_message])
        message_updates = [summary_message, human_message, response] + delete_messages
    else:
        message_updates = model.invoke([system_message] + state["messages"])

    return {"messages": message_updates}


# Define the node and edge
workflow.add_node("model", call_model)
workflow.add_edge(START, "model")

# Add simple in-memory checkpointer
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
```

## Tracing Application with LangSmith

[LangSmith](https://www.langchain.com/langsmith) traces LLM calls, tool usage, LLM model latency, token count, and cost.

To integrate LangSmith to our application, we need to generate an API key, add it as "LANGCHAIN_API_KEY" in environment variables, install langsmith dependency, setup our environment. Please refer to [set up tracing](https://smith.langchain.com/o/9f3e510d-ce6b-4f98-98a3-ff383fda2d96/?paginationState=%7B%22pageIndex%22%3A0%2C%22pageSize%22%3A5%7D) for detailed steps.

## LangChain Hub

[LangChain Hub](https://smith.langchain.com/hub?organizationId=9f3e510d-ce6b-4f98-98a3-ff383fda2d96) is a comprehensive platform that serves as a repository for pre-built components, tools, and configurations designed to accelerate the development of LLM applications. It simplifies the integration of various building blocks—models, prompts, chains, and agents—enabling developers to create robust and scalable applications without starting from scratch.

## LangChain Text Splitter Playground

[Text Splitter Playground](https://langchain-text-splitter.streamlit.app/) is a user-friendly interface designed to help developers experiment with and fine-tune text-splitting strategies. In many LLM applications, particularly those involving large documents or retrieval-augmented generation (RAG), it is essential to divide text into manageable chunks while preserving context. This tool allows users to optimize the chunking process for their specific needs.
