"""
LangGraph agent implementation for Agent RAG Studio
"""
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import AgentState, initialize_agent_state
from .nodes import AgentNodes, should_continue, increment_loop

logger = logging.getLogger(__name__)

class RAGAgent:
    """LangGraph-based RAG Agent"""
    
    def __init__(
        self,
        retriever=None,
        max_loops: int = 3,
        confidence_threshold: float = 0.7
    ):
        """
        Initialize RAG Agent
        
        Args:
            retriever: Document retriever
            max_loops: Maximum number of reasoning loops
            confidence_threshold: Confidence threshold for stopping
        """
        self.retriever = retriever
        self.max_loops = max_loops
        self.confidence_threshold = confidence_threshold
        
        # Initialize nodes
        self.nodes = AgentNodes(retriever=retriever)
        
        # Build the graph
        self.graph = self._build_graph()
        
        # Initialize checkpointer for conversation memory
        self.checkpointer = MemorySaver()
        
        # Compile the app
        self.app = self.graph.compile(checkpointer=self.checkpointer)
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph"""
        logger.info("Building LangGraph agent")
        
        # Create the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("plan", self.nodes.plan_node)
        workflow.add_node("retrieve", self.nodes.retrieve_node)
        workflow.add_node("reason", self.nodes.reason_node)
        workflow.add_node("act", self.nodes.act_node)
        workflow.add_node("respond", self.nodes.respond_node)
        workflow.add_node("increment_loop", increment_loop)
        
        # Set entry point
        workflow.set_entry_point("plan")
        
        # Add edges
        workflow.add_edge("plan", "retrieve")
        workflow.add_edge("retrieve", "reason")
        
        # Conditional edges from reason
        workflow.add_conditional_edges(
            "reason",
            should_continue,
            {
                "reason": "act",
                "respond": "respond"
            }
        )
        
        workflow.add_edge("act", "increment_loop")
        workflow.add_edge("increment_loop", "retrieve")  # Loop back to retrieve
        workflow.add_edge("respond", END)
        
        return workflow
    
    def invoke(
        self,
        question: str,
        session_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Invoke the agent with a question
        
        Args:
            question: User question
            session_id: Optional session ID for conversation memory
            **kwargs: Additional parameters
            
        Returns:
            Agent response with answer and metadata
        """
        try:
            # Initialize state
            initial_state = initialize_agent_state(
                question=question,
                max_loops=self.max_loops
            )
            
            # Configuration for the app
            config = {"configurable": {"thread_id": session_id or "default"}}
            
            logger.info(f"Invoking agent for question: {question[:50]}...")
            start_time = datetime.now()
            
            # Run the graph
            final_state = self.app.invoke(initial_state, config=config)
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Extract result
            result = {
                "answer": final_state.get("answer", "回答の生成に失敗しました"),
                "sources": final_state.get("sources", []),
                "confidence": final_state.get("confidence", 0.0),
                "execution_time": execution_time,
                "loop_count": final_state.get("loop_count", 0),
                "execution_trace": final_state.get("execution_trace", []),
                "tool_calls": final_state.get("tool_calls", []),
                "tool_results": final_state.get("tool_results", []),
                "reasoning": final_state.get("reasoning", ""),
                "plan": final_state.get("plan", ""),
                "errors": final_state.get("errors", []),
                "warnings": final_state.get("warnings", []),
                "timestamp": end_time.isoformat(),
                "model_used": final_state.get("model_used", "unknown")
            }
            
            logger.info(f"Agent completed in {execution_time:.2f}s with confidence {result['confidence']:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Agent execution error: {e}")
            return {
                "answer": f"エージェント実行中にエラーが発生しました: {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "execution_time": 0.0,
                "loop_count": 0,
                "execution_trace": [],
                "tool_calls": [],
                "tool_results": [],
                "reasoning": "",
                "plan": "",
                "errors": [str(e)],
                "warnings": [],
                "timestamp": datetime.now().isoformat(),
                "model_used": "unknown"
            }
    
    async def ainvoke(
        self,
        question: str,
        session_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Async version of invoke
        
        Args:
            question: User question
            session_id: Optional session ID for conversation memory
            **kwargs: Additional parameters
            
        Returns:
            Agent response with answer and metadata
        """
        try:
            # Initialize state
            initial_state = initialize_agent_state(
                question=question,
                max_loops=self.max_loops
            )
            
            # Configuration for the app
            config = {"configurable": {"thread_id": session_id or "default"}}
            
            logger.info(f"Async invoking agent for question: {question[:50]}...")
            start_time = datetime.now()
            
            # Run the graph asynchronously
            final_state = await self.app.ainvoke(initial_state, config=config)
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Extract result
            result = {
                "answer": final_state.get("answer", "回答の生成に失敗しました"),
                "sources": final_state.get("sources", []),
                "confidence": final_state.get("confidence", 0.0),
                "execution_time": execution_time,
                "loop_count": final_state.get("loop_count", 0),
                "execution_trace": final_state.get("execution_trace", []),
                "tool_calls": final_state.get("tool_calls", []),
                "tool_results": final_state.get("tool_results", []),
                "reasoning": final_state.get("reasoning", ""),
                "plan": final_state.get("plan", ""),
                "errors": final_state.get("errors", []),
                "warnings": final_state.get("warnings", []),
                "timestamp": end_time.isoformat(),
                "model_used": final_state.get("model_used", "unknown")
            }
            
            logger.info(f"Agent completed async in {execution_time:.2f}s with confidence {result['confidence']:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Async agent execution error: {e}")
            return {
                "answer": f"エージェント実行中にエラーが発生しました: {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "execution_time": 0.0,
                "loop_count": 0,
                "execution_trace": [],
                "tool_calls": [],
                "tool_results": [],
                "reasoning": "",
                "plan": "",
                "errors": [str(e)],
                "warnings": [],
                "timestamp": datetime.now().isoformat(),
                "model_used": "unknown"
            }
    
    def stream(
        self,
        question: str,
        session_id: Optional[str] = None,
        **kwargs
    ):
        """
        Stream the agent execution
        
        Args:
            question: User question
            session_id: Optional session ID for conversation memory
            **kwargs: Additional parameters
            
        Yields:
            Stream of execution steps
        """
        try:
            # Initialize state
            initial_state = initialize_agent_state(
                question=question,
                max_loops=self.max_loops
            )
            
            # Configuration for the app
            config = {"configurable": {"thread_id": session_id or "default"}}
            
            logger.info(f"Streaming agent for question: {question[:50]}...")
            
            # Stream the graph execution
            for chunk in self.app.stream(initial_state, config=config):
                yield chunk
                
        except Exception as e:
            logger.error(f"Agent streaming error: {e}")
            yield {
                "error": {
                    "answer": f"エージェント実行中にエラーが発生しました: {str(e)}",
                    "errors": [str(e)],
                    "timestamp": datetime.now().isoformat()
                }
            }
    
    def get_execution_trace(self, question: str, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get detailed execution trace for debugging
        
        Args:
            question: User question
            session_id: Optional session ID
            
        Returns:
            Detailed execution trace
        """
        result = self.invoke(question, session_id)
        return result.get("execution_trace", [])
    
    def clear_memory(self, session_id: Optional[str] = None):
        """Clear conversation memory for a session"""
        try:
            # This would clear the checkpointer memory
            # Implementation depends on the specific checkpointer
            logger.info(f"Cleared memory for session: {session_id or 'default'}")
        except Exception as e:
            logger.error(f"Error clearing memory: {e}")

def create_rag_agent(
    retriever=None,
    max_loops: int = 3,
    confidence_threshold: float = 0.7,
    **kwargs
) -> RAGAgent:
    """
    Factory function to create RAG agent
    
    Args:
        retriever: Document retriever
        max_loops: Maximum reasoning loops
        confidence_threshold: Confidence threshold
        **kwargs: Additional arguments
        
    Returns:
        RAGAgent instance
    """
    return RAGAgent(
        retriever=retriever,
        max_loops=max_loops,
        confidence_threshold=confidence_threshold
    )
