---
title: Persisting Agent State
date: 2025-01-24 22:25:22
tags: knowledge
---

## Persistence in LangGraph

[Persistence](https://langchain-ai.github.io/langgraph/concepts/persistence/) is a cornerstone for building robust and production-grade applications. LandGraph introduces a game-changing feature that ensures application states are stored and retrievable at any point. This redefines reliability and scalability in workflow management. This capability is especially vital when executing workflows involving interruptions, user inputs, or debugging. Whether you're building a simple app or an enterprise-grade system, persistence ensures your application is always ready to handle interruptions and user interactions gracefully.

The "Persisting Agent Stage" enables seamless workflows, especially in user-facing applications. Here’s why this feature is critical:

1. **Human-in-the-Loop Workflows:** Many applications rely on user input to make decisions or advance processes. With persistence, LandGraph allows the graph execution to pause, checkpoint the state into persistent storage, and resume later. This means the application can wait for user input and continue without losing context.
2. **Debugging and History**: Persistence creates a robust mechanism for saving the application state after every step. This makes debugging easier and enables the creation of detailed execution histories.
3. **Support for Multi-Session Scenarios**: Applications often require users to switch between sessions while maintaining their progress. Persistence ensures continuity by saving states into persistent storage.

At the heart of this feature is the **[CheckPointer](https://langchain-ai.github.io/langgraph/reference/checkpoints/)** object, a persistence layer implemented by LandGraph. Here’s how it works:

- **Integration with Databases**
  The CheckPointer can save states into various database types, including:

  - **Document databases**: Firestore, MongoDB
  - **Relational databases**: PostgreSQL, SQLite, MySQL
  - **Graph databases**: Neo4j, AWS Neptune

  For example, the following section will focus on persisting states into an SQLite database, a popular choice for local environments. The process can also be extended to managed cloud databases like Google Cloud SQL or AWS RDS.

- **State Management**
  As each node in the graph executes, the CheckPointer saves the updated state into the database. This ensures that states are recoverable after interruptions, enabling the graph to resume execution from exactly where it left off.

To implement persistence, follow these simple steps:

1. Import the **CheckPointer** object from LandGraph.
2. Create an instance of CheckPointer and configure it with a connection string (local or cloud-based database).
3. Pass the CheckPointer instance to your graph during creation. LandGraph will handle state persistence automatically after each node execution.

```python
from langgraph.checkpoint.sqlite import SqliteSaver

memory = SqliteSaver.from_conn_string(":checkpoints.sqlite:")
graph = workflow.complie(checkpointer=memory)
```

The result is that you can pause the graph, fetch user input, and continue execution seamlessly, all while ensuring states are securely stored in your chosen database.

## MemorySaver + Interrupts = Human In The Loop

Human-in-the-loop systems are essential to modern applications, allowing seamless integration of human feedback into automated workflows. With the help of the **MemorySaver** feature, you can build applications using LangGraph that pause, capture user input, and resume execution effortlessly. 

In workflows involving human interaction, there are moments where the application needs to pause, gather feedback from the user, and then continue processing. For instance, consider a sequence of tasks where:

1. A process executes its initial steps.
2. The system pauses to collect human input.
3. The workflow resumes, incorporating the user’s feedback.

This type of flow requires **interrupts** to halt the execution and **persistence** to save the current state of the workflow. Langraph provides the tools to manage both seamlessly.

### Implementation

To illustrate, let’s build a straightforward graph with the following steps:

1. Start with a simple initial node.
2. Execute a task and pause for human feedback.
3. Resume execution with the updated state and complete the workflow.

We use Langraph's **MemorySaver**, a checkpointing tool that saves the workflow’s state in memory after each node’s execution. This ephemeral storage method is perfect for local testing and prototyping. Here’s a simplified version of the setup:

```python
from dotenv import load_dotenv

load_dotenv()
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


class State(TypedDict):
    input: str
    user_feedback: str


def step_1(state: State) -> None:
    print("---Step 1---")


def human_feedback(state: State) -> None:
    print("---human_feedback---")


def step_3(state: State) -> None:
    print("---Step 3--")


builder = StateGraph(State)
builder.add_node("step_1", step_1)
builder.add_node("human_feedback", human_feedback)
builder.add_node("step_3", step_3)
builder.add_edge(START, "step_1")
builder.add_edge("step_1", "human_feedback")
builder.add_edge("human_feedback", "step_3")
builder.add_edge("step_3", END)


memory = MemorySaver()

graph = builder.compile(checkpointer=memory, interrupt_before=["human_feedback"])

graph.get_graph().draw_mermaid_png(output_file_path="graph.png")

```

The graph visualization by using [Mermaid.ink](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.graph.MermaidDrawMethod.html) is here:

![hitl-graph](Persisting-Agent-State/hitl-graph.png)

## MemorySaver Implementations

Integrating human feedback into automated systems is a growing trend in AI development. It bridges the gap between machine automation and human judgment, enabling better decision-making, improved accuracy, and adaptability. In this section, we explore how to incorporate human-in-the-loop functionality into a graph-based system while leveraging memory storage to track execution states. This walkthrough showcases the process from initialization to final execution.

```python
from dotenv import load_dotenv

load_dotenv()
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


class State(TypedDict):
    input: str
    user_feedback: str


def step_1(state: State) -> None:
    print("### Step 1 ###")


def human_feedback(state: State) -> None:
    print("### Human Feedback ###")


def step_3(state: State) -> None:
    print("### Step 3 ###")


builder = StateGraph(State)
builder.add_node("step_1", step_1)
builder.add_node("human_feedback", human_feedback)
builder.add_node("step_3", step_3)
builder.add_edge(START, "step_1")
builder.add_edge("step_1", "human_feedback")
builder.add_edge("human_feedback", "step_3")
builder.add_edge("step_3", END)


memory = MemorySaver()

graph = builder.compile(checkpointer=memory, interrupt_before=["human_feedback"])

graph.get_graph().draw_mermaid_png(output_file_path="graph.png")

if __name__ == "__main__":
    thread = {"configurable": {"thread_id": "1"}}

    initial_input = {"input": "hello world"}

    for event in graph.stream(initial_input, thread, stream_mode="values"):
        print(event)

    print(graph.get_state(thread).next)

    user_input = input("How do you want to update the state? ")

    graph.update_state(thread, {"user_feedback": user_input}, as_node="human_feedback")

    print("### State after update ###")
    print(graph.get_state(thread))

    print(graph.get_state(thread).next)

    for event in graph.stream(None, thread, stream_mode="values"):
        print(event)
```

The graph’s execution is tied to a `thread` variable, a dictionary initialized with a `thread_id`. This serves as a session or conversation identifier, distinguishing various graph runs. For simplicity, the `thread_id` is set to `1`, though a more robust implementation would use a UUID. The graph processes events using `graph.stream()`, which accepts the initial input and thread details. Events are streamed in value mode, and each event is printed for transparency. 

During execution:

* Input is processed.
* Node executions are logged. 
* Interruptions allow for dynamic human input.

Running the graph in debug mode provides insights into:

- Memory storage (`memory.storage`) containing nested objects that log the graph state.
- Transition logs for each node, showing updates or lack thereof.

At an interrupt, human feedback is solicited using Python's built-in `input()` function. This input updates the state dynamically. Once human input is integrated, the graph resumes execution. Subsequent steps process the updated state, leading to the graph’s completion.

## SqliteSaver

Switching from an ephemeral memory-based state saver to a persistent database saver can significantly enhance the durability and traceability of your graph’s execution. In this section, we’ll explore how to replace the in-memory `MemorySaver` with an `SQLiteSaver` for long-term storage and easy debugging. 

The `MemorySaver` is transient, meaning all state information vanishes after the program stops. By using an SQLite database, you can:

- Persist graph states across runs.
- Debug and troubleshoot using a structured database.
- Resume executions exactly where they were interrupted.

```python
import sqlite3

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    input: str
    user_feedback: str


def step_1(state: State) -> None:
    print("### Step 1 ###")


def human_feedback(state: State) -> None:
    print("### Human Feedback ###")


def step_3(state: State) -> None:
    print("### Step 3 ###")


builder = StateGraph(State)
builder.add_node("step_1", step_1)
builder.add_node("human_feedback", human_feedback)
builder.add_node("step_3", step_3)
builder.add_edge(START, "step_1")
builder.add_edge("step_1", "human_feedback")
builder.add_edge("human_feedback", "step_3")
builder.add_edge("step_3", END)

conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
memory = SqliteSaver(conn)
graph = builder.compile(checkpointer=memory, interrupt_before=["human_feedback"])

graph.get_graph().draw_mermaid_png(output_file_path="graph.png")

if __name__ == "__main__":
    thread = {"configurable": {"thread_id": "1"}}

    initial_input = {"input": "hello world"}

    for event in graph.stream(initial_input, thread, stream_mode="values"):
        print(event)

    print(graph.get_state(thread).next)

    user_input = input("How do you want to update the state: ")

    graph.update_state(thread, {"user_feedback": user_input}, as_node="human_feedback")

    print("### State after update ###")
    print(graph.get_state(thread))

    print(graph.get_state(thread).next)

    for event in graph.stream(None, thread, stream_mode="values"):
        print(event)
```

We start by importing the required modules. Then Initialize a connection to your SQLite database. The `check_same_thread=False` flag ensures thread-safe database operations, essential for stopping and restarting execution across different threads. After that we create an instance of `SQLiteSaver` and pass it the SQLite connection. This saver integrates seamlessly with the graph execution pipeline, persisting states to the SQLite database.

1. **Initial Execution**: Run the graph with the `SQLiteSaver`. After execution, you’ll see a new file, `checkpoints.sqlite`, created in your project directory.
2. **Inspect the Database**: Use your IDE’s database tools (e.g. SQLite3 Editor for VS Code) to load and inspect the `checkpoints.sqlite` file. You’ll find a table storing graph states, similar to what you’d see with `MemorySaver`, but now it’s persistent.

![screenshot_sqlite_ide](Persisting-Agent-State/screenshot_sqlite_ide.png)

Changing the `thread_id` allows you to simulate a new session while retaining access to previous runs. When resuming, the graph starts from the last recorded state. You can verify this by inspecting the database entries for the new `thread_id`. 

For enhanced traceability, integrate Langsmith for tracking and debugging. Langsmith provides detailed insights, including thread metadata and execution traces.
