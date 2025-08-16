"""
LangGraph nodes for Agent RAG Studio
"""
import logging
from typing import Dict, Any, List
from datetime import datetime
import json

from langchain_openai import ChatOpenAI

try:
    from langchain_core.prompts import ChatPromptTemplate
    CORE_PROMPTS_AVAILABLE = True
except ImportError:
    CORE_PROMPTS_AVAILABLE = False
    # Fallback class
    class ChatPromptTemplate:
        def __init__(self, *args, **kwargs):
            raise ImportError("ChatPromptTemplate not available")

try:
    from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
    CORE_PARSERS_AVAILABLE = True
except ImportError:
    CORE_PARSERS_AVAILABLE = False
    # Fallback classes
    class StrOutputParser:
        def __init__(self, *args, **kwargs):
            raise ImportError("StrOutputParser not available")
    class JsonOutputParser:
        def __init__(self, *args, **kwargs):
            raise ImportError("JsonOutputParser not available")

from .state import AgentState, update_state_with_trace
from .tools import tool_registry, get_tool_descriptions
from rag.config import get_config

logger = logging.getLogger(__name__)

class AgentNodes:
    """Collection of LangGraph nodes for the RAG agent"""
    
    def __init__(self, retriever=None):
        self.config = get_config()
        self.retriever = retriever
        self.llm = ChatOpenAI(
            model=self.config.default_llm,
            temperature=0.1,  # Lower temperature for planning
            openai_api_key=self.config.openai_api_key,
        )
    
    def plan_node(self, state: AgentState) -> AgentState:
        """Planning node - analyze question and create plan"""
        logger.info("Executing plan node")
        
        try:
            question = state["question"]
            
            # Create planning prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", """あなたは高度なRAGエージェントのプランナーです。
ユーザーの質問を分析し、以下の情報を含む計画を立ててください：

1. 質問の種類と複雑さ
2. 必要な情報源（文書検索、ウェブ検索、計算等）
3. 実行すべきステップ
4. 期待される回答の形式

利用可能なツール:
{tools}

JSON形式で回答してください：
{{
    "question_type": "質問の種類",
    "complexity": "low/medium/high",
    "required_sources": ["source1", "source2"],
    "steps": ["step1", "step2", "step3"],
    "expected_format": "回答の期待形式",
    "confidence": 0.0-1.0の初期信頼度
}}"""),
                ("human", "質問: {question}")
            ])
            
            # Get tool descriptions
            tools_desc = get_tool_descriptions()
            tools_text = json.dumps(tools_desc, ensure_ascii=False, indent=2)
            
            # Execute planning
            chain = prompt | self.llm | JsonOutputParser()
            result = chain.invoke({
                "question": question,
                "tools": tools_text
            })
            
            # Update state
            state["plan"] = json.dumps(result, ensure_ascii=False, indent=2)
            state["confidence"] = result.get("confidence", 0.5)
            state["model_used"] = self.config.default_llm
            
            # Add to trace
            state = update_state_with_trace(state, "plan", {
                "plan_result": result,
                "question_type": result.get("question_type"),
                "complexity": result.get("complexity")
            })
            
            logger.info(f"Plan created for question type: {result.get('question_type')}")
            return state
            
        except Exception as e:
            logger.error(f"Error in plan node: {e}")
            state["errors"].append(f"Planning error: {str(e)}")
            state["plan"] = "エラー: 計画の作成に失敗しました"
            return state
    
    def retrieve_node(self, state: AgentState) -> AgentState:
        """Retrieval node - search for relevant documents"""
        logger.info("Executing retrieve node")
        
        try:
            question = state["question"]
            
            if not self.retriever:
                state["warnings"].append("Retriever not available")
                state["context"] = "検索機能が利用できません"
                return state
            
            # Perform retrieval
            if hasattr(self.retriever, '_get_relevant_documents'):
                documents = self.retriever._get_relevant_documents(question)
            else:
                documents = self.retriever.get_relevant_documents(question)
            
            # Format context
            context_parts = []
            sources = []
            
            for i, doc in enumerate(documents, 1):
                metadata = doc.metadata
                source_info = f"[{i}] {metadata.get('file_name', 'Unknown')}"
                if metadata.get('page'):
                    source_info += f" (ページ {metadata['page']})"
                
                context_parts.append(f"{source_info}:\n{doc.page_content}")
                
                # Collect source information
                source = {
                    "id": i,
                    "title": metadata.get('title', f"ドキュメント {i}"),
                    "file_name": metadata.get('file_name', 'Unknown'),
                    "source": metadata.get('source', 'Unknown'),
                    "page": metadata.get('page'),
                    "score": metadata.get('rerank_score', metadata.get('score', 0.0)),
                    "snippet": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                }
                sources.append(source)
            
            # Update state
            state["retrieved_documents"] = documents
            state["context"] = "\n\n".join(context_parts) if context_parts else "関連する情報が見つかりませんでした"
            state["sources"] = sources
            
            # Add to trace
            state = update_state_with_trace(state, "retrieve", {
                "num_documents": len(documents),
                "sources_found": len(sources),
                "has_context": bool(context_parts)
            })
            
            logger.info(f"Retrieved {len(documents)} documents")
            return state
            
        except Exception as e:
            logger.error(f"Error in retrieve node: {e}")
            state["errors"].append(f"Retrieval error: {str(e)}")
            state["context"] = "検索中にエラーが発生しました"
            return state
    
    def reason_node(self, state: AgentState) -> AgentState:
        """Reasoning node - analyze retrieved information and determine next steps"""
        logger.info("Executing reason node")
        
        try:
            question = state["question"]
            context = state["context"]
            plan = state["plan"]
            
            # Create reasoning prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", """あなたは質問応答システムの推論エンジンです。
提供された文脈を分析し、質問に対する回答の可能性と次のアクションを決定してください。

以下の観点で分析してください：
1. 提供された文脈の質問との関連性
2. 回答に必要な情報の充足度
3. 追加の検索や外部ツールの必要性
4. 現在の情報での回答の信頼度

JSON形式で回答してください：
{{
    "context_relevance": 0.0-1.0の関連性スコア,
    "information_sufficiency": 0.0-1.0の情報充足度,
    "confidence": 0.0-1.0の回答信頼度,
    "needs_more_info": true/false,
    "reasoning": "推論の詳細説明",
    "suggested_actions": ["action1", "action2"],
    "can_answer": true/false
}}"""),
                ("human", """計画: {plan}

質問: {question}

文脈: {context}

この情報を分析してください。""")
            ])
            
            # Execute reasoning
            chain = prompt | self.llm | JsonOutputParser()
            result = chain.invoke({
                "plan": plan,
                "question": question,
                "context": context
            })
            
            # Update state
            state["reasoning"] = result.get("reasoning", "")
            state["confidence"] = result.get("confidence", 0.5)
            state["needs_more_info"] = result.get("needs_more_info", True)
            
            # Add to trace
            state = update_state_with_trace(state, "reason", {
                "reasoning_result": result,
                "context_relevance": result.get("context_relevance"),
                "can_answer": result.get("can_answer")
            })
            
            logger.info(f"Reasoning completed. Confidence: {result.get('confidence')}, Needs more info: {result.get('needs_more_info')}")
            return state
            
        except Exception as e:
            logger.error(f"Error in reason node: {e}")
            state["errors"].append(f"Reasoning error: {str(e)}")
            state["reasoning"] = "推論中にエラーが発生しました"
            return state
    
    def act_node(self, state: AgentState) -> AgentState:
        """Action node - execute external tools if needed"""
        logger.info("Executing act node")
        
        try:
            question = state["question"]
            reasoning = state["reasoning"]
            
            # Determine if we need to use tools
            prompt = ChatPromptTemplate.from_messages([
                ("system", """あなたはツール実行の判断者です。
質問と推論結果を基に、外部ツールの使用が必要かどうかを判断してください。

利用可能なツール:
{tools}

JSON形式で回答してください：
{{
    "should_use_tools": true/false,
    "tool_calls": [
        {{
            "tool_name": "tool_name",
            "method": "method_name",
            "parameters": {{"param": "value"}},
            "reason": "使用理由"
        }}
    ],
    "reasoning": "判断の理由"
}}"""),
                ("human", "質問: {question}\n\n推論結果: {reasoning}")
            ])
            
            # Get tool descriptions
            tools_desc = get_tool_descriptions()
            tools_text = json.dumps(tools_desc, ensure_ascii=False, indent=2)
            
            # Execute tool selection
            chain = prompt | self.llm | JsonOutputParser()
            result = chain.invoke({
                "question": question,
                "reasoning": reasoning,
                "tools": tools_text
            })
            
            tool_results = []
            
            # Execute tools if needed
            if result.get("should_use_tools") and result.get("tool_calls"):
                for tool_call in result["tool_calls"]:
                    tool_name = tool_call.get("tool_name")
                    method = tool_call.get("method")
                    parameters = tool_call.get("parameters", {})
                    
                    # Execute tool
                    tool_result = tool_registry.execute_tool(tool_name, method, **parameters)
                    tool_results.append({
                        "tool_call": tool_call,
                        "result": tool_result
                    })
                    
                    logger.info(f"Executed tool: {tool_name}.{method}")
            
            # Update state
            state["tool_calls"] = result.get("tool_calls", [])
            state["tool_results"] = tool_results
            
            # Add to trace
            state = update_state_with_trace(state, "act", {
                "should_use_tools": result.get("should_use_tools"),
                "num_tool_calls": len(result.get("tool_calls", [])),
                "tool_results_count": len(tool_results)
            })
            
            logger.info(f"Action completed. Tools used: {len(tool_results)}")
            return state
            
        except Exception as e:
            logger.error(f"Error in act node: {e}")
            state["errors"].append(f"Action error: {str(e)}")
            return state
    
    def respond_node(self, state: AgentState) -> AgentState:
        """Response node - generate final answer"""
        logger.info("Executing respond node")
        
        try:
            question = state["question"]
            context = state["context"]
            reasoning = state["reasoning"]
            tool_results = state.get("tool_results", [])
            sources = state.get("sources", [])
            
            # Prepare tool results text
            tool_results_text = ""
            if tool_results:
                tool_results_text = "追加情報（ツール実行結果）:\n"
                for tr in tool_results:
                    tool_call = tr["tool_call"]
                    result = tr["result"]
                    tool_results_text += f"- {tool_call['tool_name']}.{tool_call['method']}: {json.dumps(result, ensure_ascii=False)}\n"
            
            # Create response prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", """あなたは親切で正確なAIアシスタントです。
提供された情報を基に、ユーザーの質問に回答してください。

重要な制約:
1. 提供されたコンテキストと追加情報のみを使用してください
2. 推測や憶測は避け、不明な場合は「提供された情報では不明です」と述べてください
3. 数値や事実は必ずソースと一致させてください
4. 回答の最後に必ず引用を付けてください

出力形式:
- Markdownを使用してください
- 見出し、箇条書き、表を適切に使用してください
- 引用は以下の形式で付けてください: [1] タイトル（ファイル名#ページ番号）

コンテキスト:
{context}

{tool_results}

推論結果:
{reasoning}"""),
                ("human", "質問: {question}")
            ])
            
            # Execute response generation
            chain = prompt | self.llm | StrOutputParser()
            answer = chain.invoke({
                "question": question,
                "context": context,
                "reasoning": reasoning,
                "tool_results": tool_results_text
            })
            
            # Update state
            state["answer"] = answer
            
            # Calculate final confidence
            base_confidence = state.get("confidence", 0.5)
            has_context = bool(context and context != "関連する情報が見つかりませんでした")
            has_sources = len(sources) > 0
            has_tools = len(tool_results) > 0
            
            # Adjust confidence based on available information
            final_confidence = base_confidence
            if has_context:
                final_confidence += 0.2
            if has_sources:
                final_confidence += 0.1
            if has_tools:
                final_confidence += 0.1
            
            final_confidence = min(final_confidence, 1.0)
            state["confidence"] = final_confidence
            
            # Add to trace
            state = update_state_with_trace(state, "respond", {
                "answer_generated": True,
                "answer_length": len(answer),
                "final_confidence": final_confidence,
                "has_sources": has_sources,
                "has_tool_results": has_tools
            })
            
            logger.info(f"Response generated. Length: {len(answer)}, Confidence: {final_confidence}")
            return state
            
        except Exception as e:
            logger.error(f"Error in respond node: {e}")
            state["errors"].append(f"Response error: {str(e)}")
            state["answer"] = f"申し訳ありません。回答の生成中にエラーが発生しました: {str(e)}"
            return state

def should_continue(state: AgentState) -> str:
    """Decide whether to continue the loop or end"""
    loop_count = state.get("loop_count", 0)
    max_loops = state.get("max_loops", 3)
    confidence = state.get("confidence", 0.0)
    needs_more_info = state.get("needs_more_info", True)
    
    # Check if we've reached max loops
    if loop_count >= max_loops:
        logger.info("Max loops reached, ending")
        return "respond"
    
    # Check if we have high confidence and don't need more info
    if confidence >= 0.7 and not needs_more_info:
        logger.info("High confidence reached, ending")
        return "respond"
    
    # Check if we have errors
    if state.get("errors"):
        logger.info("Errors detected, ending")
        return "respond"
    
    # Continue with reasoning
    logger.info(f"Continuing loop {loop_count + 1}/{max_loops}")
    return "reason"

def increment_loop(state: AgentState) -> AgentState:
    """Increment loop counter"""
    state["loop_count"] = state.get("loop_count", 0) + 1
    return state
