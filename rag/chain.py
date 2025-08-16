"""
LCEL chain implementation for Agent RAG Studio
"""
from typing import List, Dict, Any, Optional, Tuple
import logging
import json
from datetime import datetime
import inspect

try:
    from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
    CORE_PROMPTS_AVAILABLE = True
except ImportError:
    CORE_PROMPTS_AVAILABLE = False
    # Fallback classes
    class ChatPromptTemplate:
        def __init__(self, *args, **kwargs):
            raise ImportError("ChatPromptTemplate not available")
    class PromptTemplate:
        def __init__(self, *args, **kwargs):
            raise ImportError("PromptTemplate not available")

try:
    from langchain_core.output_parsers import JsonOutputParser
    CORE_PARSERS_AVAILABLE = True
except ImportError:
    CORE_PARSERS_AVAILABLE = False
    # Fallback class
    class JsonOutputParser:
        def __init__(self, *args, **kwargs):
            raise ImportError("JsonOutputParser not available")

try:
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
    CORE_RUNNABLES_AVAILABLE = True
except ImportError:
    CORE_RUNNABLES_AVAILABLE = False
    # Fallback classes
    class RunnablePassthrough:
        def __init__(self, *args, **kwargs):
            raise ImportError("RunnablePassthrough not available")
    class RunnableLambda:
        def __init__(self, *args, **kwargs):
            raise ImportError("RunnableLambda not available")
    class RunnableParallel:
        def __init__(self, *args, **kwargs):
            raise ImportError("RunnableParallel not available")

from langchain_openai import ChatOpenAI

try:
    from langchain_core.documents import Document
    CORE_DOCUMENTS_AVAILABLE = True
except ImportError:
    CORE_DOCUMENTS_AVAILABLE = False
    # Fallback Document class
    class Document:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

try:
    from langchain_core.messages import AIMessage
    CORE_MESSAGES_AVAILABLE = True
except ImportError:
    CORE_MESSAGES_AVAILABLE = False
    # Fallback AIMessage class
    class AIMessage:
        def __init__(self, content="", **kwargs):
            self.content = content
            self.additional_kwargs = kwargs

from .config import get_config
from .retriever import BaseRetriever, RerankingRetriever

logger = logging.getLogger(__name__)

class RAGChain:
    """RAG chain with LCEL implementation"""
    
    def __init__(
        self,
        retriever: BaseRetriever,
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_tokens: int = 1000,
        use_reranking: bool = True,
        rerank_top_k: int = 3,
        output_format: str = "markdown",  # "markdown" or "json"
    ):
        """
        Initialize RAG chain
        
        Args:
            retriever: Document retriever
            llm_model: LLM model name
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            use_reranking: Whether to use reranking
            rerank_top_k: Number of documents after reranking
            output_format: Output format ("markdown" or "json")
        """
        self.config = get_config()
        self.retriever = retriever
        self.llm_model = llm_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_reranking = use_reranking
        self.rerank_top_k = rerank_top_k
        self.output_format = output_format
        
        # Initialize LLM (map alias to real provider model for API)
        if llm_model.startswith("gpt-5"):
            # gpt-5 系は Responses API を利用
            self.llm = ChatOpenAI(
                model=llm_model,
                output_version="responses/v1",
                max_tokens=max_tokens,
                openai_api_key=self.config.openai_api_key,
            )
            # 内部保持の温度はUI表示用に1.0へ寄せる
            self.temperature = 1.0
        else:
            self.llm = ChatOpenAI(
                model=llm_model,
                temperature=temperature,
                max_tokens=max_tokens,
                openai_api_key=self.config.openai_api_key,
            )
        
        # Initialize reranking retriever if needed
        # Cohereキーが無い場合は自動でリランキングを無効化
        if use_reranking and self.config.cohere_api_key and self.config.cohere_api_key.strip():
            self.reranking_retriever = RerankingRetriever(
                base_retriever=retriever,
                rerank_top_k=rerank_top_k,
            )
        else:
            self.reranking_retriever = None
            if use_reranking:
                logger.info("Cohere reranking disabled: no valid API key")
        
        # Build the chain
        self.chain = self._build_chain()
    
    def _build_chain(self):
        """Build the LCEL chain"""
        # Create prompt based on output format
        if self.output_format == "json":
            prompt = self._create_json_prompt()
            json_parser = JsonOutputParser()
        else:
            prompt = self._create_markdown_prompt()
            json_parser = None
        
        # Create retrieval function
        def retrieve_documents(inputs: Dict[str, Any]) -> Dict[str, Any]:
            query = inputs["question"]
            
            try:
                if self.use_reranking and self.reranking_retriever:
                    documents = self.reranking_retriever.retrieve(query)
                else:
                    # Generic safe call to retriever (supports LangChain run_manager kw)
                    documents = []
                    if hasattr(self.retriever, 'invoke'):
                        documents = self.retriever.invoke(query)
                    else:
                        for meth_name in ['get_relevant_documents', '_get_relevant_documents']:
                            if hasattr(self.retriever, meth_name):
                                meth = getattr(self.retriever, meth_name)
                                try:
                                    if 'run_manager' in inspect.signature(meth).parameters:
                                        documents = meth(query, run_manager=None)
                                    else:
                                        documents = meth(query)
                                except Exception:
                                    continue
                                break
                
                # Format context
                context = self._format_context(documents)
                sources = self._extract_sources(documents)
                
                return {
                    "context": context,
                    "question": query,
                    "sources": sources,
                    "retrieved_docs": documents,
                }
                
            except Exception as e:
                logger.error(f"Error in document retrieval: {e}")
                return {
                    "context": "検索中にエラーが発生しました。",
                    "question": query,
                    "sources": [],
                    "retrieved_docs": [],
                }
        
        # --- 出力正規化 ---
        def _normalize_llm_output(model_output: Any) -> Any:
            # 文字列そのまま
            if isinstance(model_output, str):
                return model_output
            # LangChain AIMessage → content or additional_kwargsから復元
            try:
                if isinstance(model_output, AIMessage):
                    content = getattr(model_output, "content", None)
                    # contentが文字列
                    if isinstance(content, str) and content.strip():
                        return content
                    # contentがパーツ配列（Responses APIのcontent構造など）
                    if isinstance(content, list):
                        parts: List[str] = []
                        for part in content:
                            if isinstance(part, dict):
                                text_val = part.get("text")
                                if isinstance(text_val, str) and text_val:
                                    parts.append(text_val)
                                elif isinstance(text_val, dict):
                                    val = text_val.get("value")
                                    if isinstance(val, str):
                                        parts.append(val)
                            else:
                                t = getattr(part, "text", None)
                                if isinstance(t, str) and t:
                                    parts.append(t)
                        if parts:
                            return "".join(parts)
                    # additional_kwargsにResponses APIの出力がある場合
                    extra = getattr(model_output, "additional_kwargs", {}) or {}
                    if isinstance(extra, dict):
                        # トップレベルのoutput配列をチェック（新Responses API形式）
                        top_output = extra.get("output")
                        if isinstance(top_output, list):
                            for item in top_output:
                                if isinstance(item, dict):
                                    contents = item.get("content", [])
                                    if isinstance(contents, list):
                                        for c in contents:
                                            if isinstance(c, dict):
                                                c_type = c.get("type")
                                                if c_type == "output_text":
                                                    text_val = c.get("text")
                                                    if isinstance(text_val, str) and text_val.strip():
                                                        return text_val
                        # 直接の output_text
                        if isinstance(extra.get("output_text"), str):
                            return extra["output_text"]
                        # response オブジェクト（Responses API）
                        if isinstance(extra.get("response"), dict):
                            resp = extra["response"]
                            # まず output_text があれば最優先
                            ot = resp.get("output_text")
                            if isinstance(ot, str) and ot.strip():
                                return ot
                            # output の配列から output_text / reasoning.summary を抽出
                            out_text_parts: List[str] = []
                            output_items = resp.get("output")
                            if isinstance(output_items, list):
                                for item in output_items:
                                    if not isinstance(item, dict):
                                        continue
                                    contents = item.get("content")
                                    if isinstance(contents, list):
                                        for c in contents:
                                            if not isinstance(c, dict):
                                                continue
                                            c_type = c.get("type")
                                            # 出力テキスト
                                            if c_type == "output_text":
                                                t = c.get("text")
                                                if isinstance(t, str) and t:
                                                    out_text_parts.append(t)
                                                elif isinstance(t, dict):
                                                    val = t.get("value")
                                                    if isinstance(val, str) and val:
                                                        out_text_parts.append(val)
                                            # reasoningのみの場合はsummaryを利用
                                            elif c_type == "reasoning":
                                                summary = c.get("summary")
                                                if isinstance(summary, list):
                                                    # summaryが文字列の配列の場合に結合
                                                    out_text_parts.append("\n".join([s for s in summary if isinstance(s, str)]))
                            if out_text_parts:
                                return "".join(out_text_parts).strip()
                        # Chat Completionsのchoices形式へのフォールバック
                        choices = extra.get("choices")
                        if isinstance(choices, list) and choices:
                            first = choices[0] or {}
                            msg = first.get("message", {})
                            if isinstance(msg, dict):
                                cc = msg.get("content")
                                if isinstance(cc, str):
                                    return cc
                # ここまでで抽出できなければ、そのまま文字列化
                return str(getattr(model_output, "content", model_output))
            except Exception:
                try:
                    return str(model_output)
                except Exception:
                    return ""

        # --- ここからチェーン構築 ---
        base_chain = (
            RunnableLambda(retrieve_documents)
            | prompt
            | self.llm
            | RunnableLambda(_normalize_llm_output)
        )

        # JSON形式の場合は正規化後にJSONパースを適用
        chain = base_chain if json_parser is None else (base_chain | json_parser)
        return chain
    
    def _create_markdown_prompt(self) -> ChatPromptTemplate:
        """Create prompt for markdown output"""
        system_prompt = """あなたは親切で正確なAIアシスタントです。提供されたコンテキストのみを使用して、ユーザーの質問に回答してください。

重要な制約:
1. コンテキストに含まれていない情報は使用しないでください
2. 推測や憶測は避け、不明な場合は「提供された情報では不明です」と述べてください
3. 数値や事実は必ずソースと一致させてください
4. 回答の最後に必ず引用を付けてください

出力形式:
- Markdownを使用してください
- 見出し、箇条書き、表を適切に使用してください
- 引用は以下の形式で付けてください: [1] タイトル（ファイル名#ページ番号）

コンテキスト:
{context}

質問: {question}

回答:"""
        
        human_prompt = "{question}"
        
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt),
        ])
    
    def _create_json_prompt(self) -> ChatPromptTemplate:
        """Create prompt for JSON output"""
        system_prompt = """あなたは親切で正確なAIアシスタントです。提供されたコンテキストのみを使用して、ユーザーの質問に構造化されたJSON形式で回答してください。

重要な制約:
1. コンテキストに含まれていない情報は使用しないでください
2. 推測や憶測は避け、不明な場合は適切に示してください
3. 数値や事実は必ずソースと一致させてください

出力は以下のJSON形式で返してください:
{{
    "answer": "回答内容（Markdown形式）",
    "confidence": 0.0-1.0の信頼度,
    "key_points": ["要点1", "要点2", "要点3"],
    "tables": [
        {{
            "title": "表のタイトル",
            "headers": ["列1", "列2"],
            "rows": [["値1", "値2"]]
        }}
    ],
    "sources": [
        {{
            "title": "ソースタイトル",
            "file": "ファイル名",
            "page": "ページ番号",
            "relevance": 0.0-1.0の関連度
        }}
    ]
}}

コンテキスト:
{context}"""
        
        human_prompt = "{question}"
        
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt),
        ])
    
    def _format_context(self, documents: List[Document]) -> str:
        """Format documents as context"""
        if not documents:
            return "関連する情報が見つかりませんでした。"
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            metadata = doc.metadata
            source_info = f"[{i}] {metadata.get('file_name', 'Unknown')}"
            if metadata.get('page'):
                source_info += f" (ページ {metadata['page']})"
            
            context_part = f"{source_info}:\n{doc.page_content}\n"
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _extract_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Extract source information from documents"""
        sources = []
        for i, doc in enumerate(documents, 1):
            metadata = doc.metadata
            source = {
                "id": i,
                "title": metadata.get('title', f"ドキュメント {i}"),
                "file_name": metadata.get('file_name', 'Unknown'),
                "source": metadata.get('source', 'Unknown'),
                "page": metadata.get('page'),
                "chunk_id": metadata.get('chunk_id'),
                "retrieval_source": metadata.get('retrieval_source', 'unknown'),
                "score": metadata.get('rerank_score', metadata.get('score', 0.0)),
                "snippet": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            }
            sources.append(source)
        
        return sources
    
    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke the RAG chain
        
        Args:
            inputs: Input dictionary with 'question' key
            
        Returns:
            Dictionary with answer and metadata
        """
        start_time = datetime.now()
        
        try:
            # Run the chain
            result = self.chain.invoke(inputs)
            
            # Calculate timing
            end_time = datetime.now()
            latency = (end_time - start_time).total_seconds()
            
            # Process result based on output format
            if self.output_format == "json":
                if isinstance(result, str):
                    try:
                        result = json.loads(result)
                    except json.JSONDecodeError:
                        logger.error("Failed to parse JSON response")
                        result = {
                            "answer": result,
                            "confidence": 0.5,
                            "key_points": [],
                            "tables": [],
                            "sources": [],
                        }
                
                final_answer = (result.get("answer", "") or "").strip()
                if not final_answer:
                    final_answer = "提供された情報から回答を生成できませんでした。質問を具体化するか、別の観点でお試しください。"
                return {
                    "answer": final_answer,
                    "sources": result.get("sources", []),
                    "confidence": result.get("confidence", 0.5),
                    "key_points": result.get("key_points", []),
                    "tables": result.get("tables", []),
                    "latency": latency,
                    "model": self.llm_model,
                    "timestamp": end_time.isoformat(),
                }
            else:
                # For markdown format, we need to get sources from the retrieval step
                # This is a simplified implementation
                answer_text = (result or "").strip()
                if not answer_text:
                    answer_text = "提供された情報から回答を生成できませんでした。質問を具体化するか、別の観点でお試しください。"
                return {
                    "answer": answer_text,
                    "sources": [],  # Would need to be populated from retrieval step
                    "latency": latency,
                    "model": self.llm_model,
                    "timestamp": end_time.isoformat(),
                }
            
        except Exception as e:
            logger.error(f"Error in RAG chain execution: {e}")
            return {
                "answer": f"申し訳ありません。回答の生成中にエラーが発生しました: {str(e)}",
                "sources": [],
                "latency": (datetime.now() - start_time).total_seconds(),
                "model": self.llm_model,
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
            }
    
    async def ainvoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Async version of invoke"""
        # For now, just call the sync version
        # In production, you'd implement proper async support
        return self.invoke(inputs)
    
    def update_config(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        use_reranking: Optional[bool] = None,
        rerank_top_k: Optional[int] = None,
    ):
        """Update chain configuration"""
        if temperature is not None:
            self.temperature = temperature
            self.llm.temperature = temperature
        
        if max_tokens is not None:
            self.max_tokens = max_tokens
            self.llm.max_tokens = max_tokens
        
        if use_reranking is not None:
            self.use_reranking = use_reranking
            if use_reranking and self.reranking_retriever is None:
                self.reranking_retriever = RerankingRetriever(
                    base_retriever=self.retriever,
                    rerank_top_k=self.rerank_top_k,
                )
        
        if rerank_top_k is not None:
            self.rerank_top_k = rerank_top_k
            if self.reranking_retriever:
                self.reranking_retriever.rerank_top_k = rerank_top_k

class ConversationalRAGChain:
    """Conversational RAG chain with chat history"""
    
    def __init__(self, base_chain: RAGChain):
        """
        Initialize conversational RAG chain
        
        Args:
            base_chain: Base RAG chain
        """
        self.base_chain = base_chain
        self.chat_history = []
    
    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke with chat history context
        
        Args:
            inputs: Input dictionary with 'question' key
            
        Returns:
            Dictionary with answer and metadata
        """
        question = inputs["question"]
        
        # Add chat history context if available
        if self.chat_history:
            context_question = self._create_contextualized_question(question)
        else:
            context_question = question
        
        # Invoke base chain
        result = self.base_chain.invoke({"question": context_question})
        
        # Update chat history
        self.chat_history.append({
            "question": question,
            "answer": result["answer"],
            "timestamp": result["timestamp"],
        })
        
        # Keep only last N exchanges
        if len(self.chat_history) > 10:
            self.chat_history = self.chat_history[-10:]
        
        return result
    
    def _create_contextualized_question(self, question: str) -> str:
        """Create question with chat history context"""
        if not self.chat_history:
            return question
        
        # Simple implementation - append recent context
        recent_exchanges = self.chat_history[-3:]  # Last 3 exchanges
        
        context_parts = []
        for exchange in recent_exchanges:
            context_parts.append(f"Q: {exchange['question']}")
            context_parts.append(f"A: {exchange['answer'][:200]}...")
        
        context = "\n".join(context_parts)
        
        contextualized = f"""過去の会話の文脈:
{context}

現在の質問: {question}

上記の文脈を考慮して、現在の質問に回答してください。"""
        
        return contextualized
    
    def clear_history(self):
        """Clear chat history"""
        self.chat_history = []
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get chat history"""
        return self.chat_history.copy()

def create_rag_chain(
    retriever: BaseRetriever,
    llm_model: str = "gpt-4o-mini",
    **kwargs
) -> RAGChain:
    """
    Factory function to create RAG chain
    
    Args:
        retriever: Document retriever
        llm_model: LLM model name
        **kwargs: Additional arguments
        
    Returns:
        RAGChain instance
    """
    return RAGChain(
        retriever=retriever,
        llm_model=llm_model,
        **kwargs
    )

def create_conversational_rag_chain(
    retriever: BaseRetriever,
    **kwargs
) -> ConversationalRAGChain:
    """
    Factory function to create conversational RAG chain
    
    Args:
        retriever: Document retriever
        **kwargs: Additional arguments for RAGChain
        
    Returns:
        ConversationalRAGChain instance
    """
    base_chain = create_rag_chain(retriever, **kwargs)
    return ConversationalRAGChain(base_chain)
