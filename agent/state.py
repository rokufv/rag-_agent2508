"""
State management for LangGraph Agent
"""
from typing import List, Dict, Any, Optional, TypedDict, Annotated
from langchain_core.documents import Document
import operator

class AgentState(TypedDict):
    """State for the RAG Agent"""
    
    # Input and output
    question: str
    answer: str
    
    # Retrieval results
    retrieved_documents: List[Document]
    context: str
    sources: List[Dict[str, Any]]
    
    # Agent reasoning
    plan: str
    reasoning: str
    confidence: float
    needs_more_info: bool
    
    # Search and analysis
    search_queries: List[str]
    external_search_results: List[Dict[str, Any]]
    
    # Tool usage
    tool_calls: List[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]
    
    # Loop control
    loop_count: int
    max_loops: int
    
    # Metadata
    execution_trace: List[Dict[str, Any]]
    timestamp: str
    model_used: str
    
    # Error handling
    errors: List[str]
    warnings: List[str]

def update_state_with_trace(state: AgentState, node_name: str, data: Dict[str, Any]) -> AgentState:
    """Add execution trace to state"""
    from datetime import datetime
    
    trace_entry = {
        "node": node_name,
        "timestamp": datetime.now().isoformat(),
        "data": data
    }
    
    state["execution_trace"].append(trace_entry)
    return state

def initialize_agent_state(question: str, max_loops: int = 3) -> AgentState:
    """Initialize agent state with default values"""
    from datetime import datetime
    
    return AgentState(
        question=question,
        answer="",
        retrieved_documents=[],
        context="",
        sources=[],
        plan="",
        reasoning="",
        confidence=0.0,
        needs_more_info=True,
        search_queries=[],
        external_search_results=[],
        tool_calls=[],
        tool_results=[],
        loop_count=0,
        max_loops=max_loops,
        execution_trace=[],
        timestamp=datetime.now().isoformat(),
        model_used="",
        errors=[],
        warnings=[]
    )
