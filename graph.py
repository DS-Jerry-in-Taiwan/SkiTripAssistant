from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from state import AgentState
from nodes import rag_retrieval_node, call_model_node

#TODO: Add node workflow below

def create_graph() -> StateGraph:
    # initialize graph
    workflow = StateGraph(AgentState)
    
    # register nodes
    workflow.add_node("rag", rag_retrieval_node)
    workflow.add_node("llm", call_model_node)
    
    # define workflow edge
    workflow.add_edge(START, "rag")
    workflow.add_edge("rag", "llm")
    workflow.add_edge("llm", END)
    
    # Initialize memory saver
    memory_saver = MemorySaver()
    
    # compile graph with memory saver as checkpointer
    app = workflow.compile(checkpointer=memory_saver)

    return app