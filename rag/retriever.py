"""
Retrieval functionality for Agent RAG Studio
"""
from typing import List, Optional, Dict, Any, Tuple
# from pydantic import Field
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.documents import Document
# from langchain_community.retrievers import EnsembleRetriever  # Not needed for our implementation
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever
from rank_bm25 import BM25Okapi
import cohere

from .store import VectorStoreManager
from .embedder import EmbeddingManager
from .config import get_config

logger = logging.getLogger(__name__)

class HybridRetriever:
    """Hybrid retriever combining BM25 and vector similarity search"""
    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        documents: Optional[List[Document]] = None,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        top_k: int = 6,
    ):
        self.vector_store_manager = vector_store_manager
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.top_k = top_k
        self.bm25_retriever = None
        if documents:
            self._initialize_bm25(documents)
        self.vector_retriever = self.vector_store_manager.vector_store.as_retriever(
            search_kwargs={"k": self.top_k}
        ) if self.vector_store_manager.vector_store else None

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents using hybrid search"""
        return self._get_relevant_documents(query)
    
    def _initialize_bm25(self, documents: List[Document]):
        """Initialize BM25 retriever"""
        try:
            # Prepare texts for BM25
            texts = [doc.page_content for doc in documents]
            
            # Create BM25 retriever
            self.bm25_retriever = BM25Retriever.from_texts(texts, metadatas=[doc.metadata for doc in documents])
            self.bm25_retriever.k = self.top_k
            
            logger.info(f"Initialized BM25 retriever with {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Error initializing BM25: {e}")
            self.bm25_retriever = None
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents using hybrid search"""
        try:
            all_results = []
            
            # Vector search
            if self.vector_retriever:
                try:
                    if hasattr(self.vector_retriever, 'invoke'):
                        vector_docs = self.vector_retriever.invoke(query)
                    else:
                        vector_docs = self.vector_retriever.get_relevant_documents(query)
                    # Add source and weight to metadata
                    for doc in vector_docs:
                        doc.metadata['retrieval_source'] = 'vector'
                        doc.metadata['retrieval_weight'] = self.vector_weight
                    all_results.extend(vector_docs)
                    logger.info(f"Vector search returned {len(vector_docs)} documents")
                except Exception as e:
                    logger.error(f"Vector search failed: {e}")
            
            # BM25 search
            if self.bm25_retriever:
                try:
                    if hasattr(self.bm25_retriever, 'invoke'):
                        bm25_docs = self.bm25_retriever.invoke(query)
                    else:
                        bm25_docs = self.bm25_retriever.get_relevant_documents(query)
                    # Add source and weight to metadata
                    for doc in bm25_docs:
                        doc.metadata['retrieval_source'] = 'bm25'
                        doc.metadata['retrieval_weight'] = self.bm25_weight
                    all_results.extend(bm25_docs)
                    logger.info(f"BM25 search returned {len(bm25_docs)} documents")
                except Exception as e:
                    logger.error(f"BM25 search failed: {e}")
            
            # Combine and deduplicate results
            combined_results = self._combine_results(all_results, query)
            
            # Limit to top_k
            final_results = combined_results[:self.top_k]
            
            logger.info(f"Hybrid retrieval returned {len(final_results)} documents")
            return final_results
            
        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {e}")
            return []
    
    def _combine_results(self, results: List[Document], query: str) -> List[Document]:
        """Combine and deduplicate results from different retrievers"""
        # Simple deduplication based on content similarity
        unique_results = []
        seen_content = set()
        
        for doc in results:
            content_hash = hash(doc.page_content.strip())
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(doc)
        
        # Sort by relevance (this is a simple implementation)
        # In practice, you might want to implement more sophisticated ranking
        return unique_results

class RerankingRetriever:
    """Retriever with reranking capability"""
    
    def __init__(
        self,
        base_retriever: BaseRetriever,
        reranker_model: str = "rerank-multilingual-v3.0",
        rerank_top_k: int = 3,
    ):
        """
        Initialize reranking retriever
        
        Args:
            base_retriever: Base retriever to use
            reranker_model: Cohere reranking model name
            rerank_top_k: Number of documents to return after reranking
        """
        self.base_retriever = base_retriever
        self.reranker_model = reranker_model
        self.rerank_top_k = rerank_top_k
        self.config = get_config()
        self.rerank_disabled = False
        
        # Initialize Cohere client
        self.cohere_client = None
        if self.config.cohere_api_key and self.config.cohere_api_key.strip():
            try:
                self.cohere_client = cohere.Client(self.config.cohere_api_key)
                logger.info("Initialized Cohere reranker")
            except Exception as e:
                logger.error(f"Failed to initialize Cohere client: {e}")
                self.cohere_client = None
        else:
            logger.info("Cohere API key not provided, reranking disabled")
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        """
        Retrieve and rerank documents
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve initially (before reranking)
            
        Returns:
            Reranked list of documents
        """
        try:
            # Get initial results
            if hasattr(self.base_retriever, 'invoke'):
                initial_docs = self.base_retriever.invoke(query)
            elif hasattr(self.base_retriever, 'get_relevant_documents'):
                initial_docs = self.base_retriever.get_relevant_documents(query)
            else:
                initial_docs = self.base_retriever._get_relevant_documents(query)
            
            if not initial_docs:
                return []
            
            # If no reranker available, API key invalid, or disabled due to prior error → return base results
            if (not self.cohere_client) or (not self.config.cohere_api_key) or self.rerank_disabled:
                logger.warning("Cohere reranker unavailable/disabled, returning base results")
                return initial_docs[:self.rerank_top_k]
            
            # Rerank with Cohere
            reranked_docs = self._rerank_with_cohere(query, initial_docs)
            
            logger.info(f"Reranked {len(initial_docs)} documents to {len(reranked_docs)}")
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Error in reranking retrieval: {e}")
            # Fallback to initial results
            try:
                if hasattr(self.base_retriever, 'invoke'):
                    fallback_docs = self.base_retriever.invoke(query)
                elif hasattr(self.base_retriever, 'get_relevant_documents'):
                    fallback_docs = self.base_retriever.get_relevant_documents(query)
                else:
                    fallback_docs = self.base_retriever._get_relevant_documents(query)
                return fallback_docs[:self.rerank_top_k]
            except:
                return []
    
    def _rerank_with_cohere(self, query: str, documents: List[Document]) -> List[Document]:
        """Rerank documents using Cohere"""
        try:
            # Prepare documents for reranking
            doc_texts = [doc.page_content for doc in documents]
            
            # Call Cohere rerank API
            response = self.cohere_client.rerank(
                model=self.reranker_model,
                query=query,
                documents=doc_texts,
                top_n=min(self.rerank_top_k, len(documents)),
            )
            
            # Reorder documents based on reranking scores
            reranked_docs = []
            for result in response.results:
                original_doc = documents[result.index]
                # Add reranking score to metadata
                original_doc.metadata['rerank_score'] = result.relevance_score
                original_doc.metadata['rerank_model'] = self.reranker_model
                reranked_docs.append(original_doc)
            
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Error in Cohere reranking: {e}")
            # Disable reranker for subsequent calls to avoid repeated 401 spam
            self.rerank_disabled = True
            logger.warning("Disabling Cohere reranker for this session due to error. Using base results.")
            # Return original documents if reranking fails
            return documents[:self.rerank_top_k]

class MultiQueryRetriever:
    """Retriever that generates multiple query variations"""
    
    def __init__(
        self,
        base_retriever: BaseRetriever,
        llm,
        num_queries: int = 3,
    ):
        """
        Initialize multi-query retriever
        
        Args:
            base_retriever: Base retriever to use
            llm: Language model for query generation
            num_queries: Number of query variations to generate
        """
        self.base_retriever = base_retriever
        self.llm = llm
        self.num_queries = num_queries
    
    def generate_queries(self, original_query: str) -> List[str]:
        """Generate multiple query variations"""
        prompt = f"""
        元の質問に基づいて、同じ情報を検索するための{self.num_queries}つの異なる質問を生成してください。
        各質問は元の質問の意図を保ちながら、異なる表現や観点から述べてください。

        元の質問: {original_query}

        生成した質問:
        1. 
        2. 
        3. 
        """
        
        try:
            response = self.llm.invoke(prompt)
            
            # Parse generated queries
            lines = response.content.split('\n') if hasattr(response, 'content') else str(response).split('\n')
            queries = [original_query]  # Include original query
            
            for line in lines:
                line = line.strip()
                if line and (line.startswith('1.') or line.startswith('2.') or line.startswith('3.')):
                    query = line.split('.', 1)[1].strip()
                    if query:
                        queries.append(query)
            
            return queries[:self.num_queries + 1]  # Include original + generated
            
        except Exception as e:
            logger.error(f"Error generating queries: {e}")
            return [original_query]  # Fallback to original query
    
    def retrieve(self, query: str) -> List[Document]:
        """Retrieve using multiple query variations"""
        try:
            # Generate query variations
            queries = self.generate_queries(query)
            logger.info(f"Generated {len(queries)} query variations")
            
            # Retrieve for each query
            all_docs = []
            for q in queries:
                if hasattr(self.base_retriever, '_get_relevant_documents'):
                    docs = self.base_retriever._get_relevant_documents(q)
                else:
                    docs = self.base_retriever.get_relevant_documents(q)
                all_docs.extend(docs)
            
            # Deduplicate and score
            unique_docs = self._deduplicate_and_score(all_docs, query)
            
            logger.info(f"Multi-query retrieval returned {len(unique_docs)} unique documents")
            return unique_docs
            
        except Exception as e:
            logger.error(f"Error in multi-query retrieval: {e}")
            # Fallback to single query
            if hasattr(self.base_retriever, 'invoke'):
                return self.base_retriever.invoke(query)
            elif hasattr(self.base_retriever, 'get_relevant_documents'):
                return self.base_retriever.get_relevant_documents(query)
            else:
                return self.base_retriever._get_relevant_documents(query)
    
    def _deduplicate_and_score(self, documents: List[Document], original_query: str) -> List[Document]:
        """Deduplicate documents and score by frequency"""
        doc_counts = {}
        doc_objects = {}
        
        for doc in documents:
            content_hash = hash(doc.page_content.strip())
            if content_hash in doc_counts:
                doc_counts[content_hash] += 1
            else:
                doc_counts[content_hash] = 1
                doc_objects[content_hash] = doc
        
        # Sort by frequency (documents appearing in multiple query results are likely more relevant)
        sorted_docs = sorted(
            doc_objects.items(),
            key=lambda x: doc_counts[x[0]],
            reverse=True
        )
        
        # Add frequency score to metadata
        result_docs = []
        for content_hash, doc in sorted_docs:
            doc.metadata['multi_query_frequency'] = doc_counts[content_hash]
            result_docs.append(doc)
        
        return result_docs

def create_retriever(
    vector_store_manager: VectorStoreManager,
    documents: Optional[List[Document]] = None,
    retriever_type: str = "hybrid",
    **kwargs
):
    """
    Factory function to create retrievers
    
    Args:
        vector_store_manager: Vector store manager
        documents: Documents for BM25 initialization
        retriever_type: Type of retriever ('vector', 'bm25', 'hybrid')
        **kwargs: Additional arguments
        
    Returns:
        Configured retriever
    """
    if retriever_type == "vector":
        if vector_store_manager.vector_store:
            return vector_store_manager.vector_store.as_retriever(
                search_kwargs={"k": kwargs.get("top_k", 6)}
            )
        else:
            raise ValueError("Vector store not initialized")
    
    elif retriever_type == "bm25":
        if not documents:
            raise ValueError("Documents required for BM25 retriever")
        texts = [doc.page_content for doc in documents]
        bm25_retriever = BM25Retriever.from_texts(
            texts, 
            metadatas=[doc.metadata for doc in documents]
        )
        bm25_retriever.k = kwargs.get("top_k", 6)
        return bm25_retriever
    
    elif retriever_type == "hybrid":
        return HybridRetriever(
            vector_store_manager=vector_store_manager,
            documents=documents,
            **kwargs
        )
    
    else:
        raise ValueError(f"Unsupported retriever type: {retriever_type}")

def create_reranking_retriever(
    base_retriever: BaseRetriever,
    rerank_top_k: int = 3,
    **kwargs
) -> RerankingRetriever:
    """
    Create reranking retriever
    
    Args:
        base_retriever: Base retriever
        rerank_top_k: Number of documents after reranking
        **kwargs: Additional arguments
        
    Returns:
        RerankingRetriever instance
    """
    return RerankingRetriever(
        base_retriever=base_retriever,
        rerank_top_k=rerank_top_k,
        **kwargs
    )
