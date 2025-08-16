"""
Embedding functionality for Agent RAG Studio
"""
from typing import List, Optional, Dict, Any
from langchain_openai import OpenAIEmbeddings
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    HuggingFaceEmbeddings = None

try:
    from langchain_core.documents import Document
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False
    # Fallback Document class
    class Document:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

import logging
import numpy as np
from .config import get_config, RAGConfig

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Manages different embedding models and strategies"""
    
    SUPPORTED_MODELS = {
        # OpenAI models
        'text-embedding-3-small': {'provider': 'openai', 'dimensions': 1536},
        'text-embedding-3-large': {'provider': 'openai', 'dimensions': 3072},
        'text-embedding-ada-002': {'provider': 'openai', 'dimensions': 1536},
        
        # HuggingFace models
        'sentence-transformers/all-MiniLM-L6-v2': {'provider': 'huggingface', 'dimensions': 384},
        'sentence-transformers/all-mpnet-base-v2': {'provider': 'huggingface', 'dimensions': 768},
        'intfloat/multilingual-e5-large': {'provider': 'huggingface', 'dimensions': 1024},
        'intfloat/e5-large-v2': {'provider': 'huggingface', 'dimensions': 1024},
    }
    
    def __init__(self, config: Optional[RAGConfig] = None, model_name: Optional[str] = None):
        """
        Initialize EmbeddingManager
        
        Args:
            config: RAGConfig instance or None to use default
            model_name: Name of the embedding model to use
        """
        # 設定オブジェクトの一貫性を確保
        if config is None:
            from .config import get_config
            config = get_config()
            logger.info(f"Using global config in EmbeddingManager (ID: {id(config)})")
        else:
            logger.info(f"Using provided config in EmbeddingManager (ID: {id(config)})")
        
        self.config = config
        self.model_name = model_name or self.config.embedding_model
        self.model_info = self.SUPPORTED_MODELS.get(self.model_name, {})
        self.embeddings = None
        
        # 設定の状態をログ出力
        logger.info(f"EmbeddingManager initialized with model: {self.model_name}")
        logger.info(f"Config OpenAI key available: {bool(self.config.openai_api_key)}")
        logger.info(f"Config OpenAI key length: {len(self.config.openai_api_key) if self.config.openai_api_key else 0}")
        
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize the embedding model"""
        try:
            provider = self.model_info.get('provider', 'openai')
            
            if provider == 'openai':
                # Check if OpenAI API key is available and valid
                if not self.config.openai_api_key or self.config.openai_api_key.startswith('sk-test-'):
                    logger.warning("OpenAI API key not available or invalid, falling back to local model")
                    self._initialize_fallback()
                    return
                
                self.embeddings = OpenAIEmbeddings(
                    model=self.model_name,
                    openai_api_key=self.config.openai_api_key,
                )
                
            elif provider == 'huggingface':
                if not HUGGINGFACE_AVAILABLE:
                    logger.warning("HuggingFace embeddings not available, falling back to OpenAI")
                    self._initialize_fallback()
                    return
                
                # Configure device and model kwargs for better performance
                model_kwargs = {'device': 'cpu'}  # Can be changed to 'cuda' if GPU available
                encode_kwargs = {'normalize_embeddings': True}
                
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=self.model_name,
                    model_kwargs=model_kwargs,
                    encode_kwargs=encode_kwargs,
                )
            
            else:
                raise ValueError(f"Unsupported embedding provider: {provider}")
            
            logger.info(f"Initialized {provider} embeddings with model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            # Fallback to a simple model
            self._initialize_fallback()
    
    def _initialize_fallback(self):
        """Initialize fallback embedding model"""
        try:
            if not HUGGINGFACE_AVAILABLE:
                logger.error("HuggingFace embeddings not available and no fallback possible")
                raise ImportError("HuggingFace embeddings not available")
            
            logger.warning("Falling back to all-MiniLM-L6-v2 model")
            self.model_name = 'sentence-transformers/all-MiniLM-L6-v2'
            self.model_info = self.SUPPORTED_MODELS[self.model_name]
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True},
            )
            
        except Exception as e:
            logger.error(f"Fallback embedding initialization failed: {e}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        try:
            embeddings = self.embeddings.embed_documents(texts)
            logger.info(f"Embedded {len(texts)} documents")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text
        
        Args:
            text: Query text
            
        Returns:
            Embedding vector
        """
        try:
            embedding = self.embeddings.embed_query(text)
            return embedding
            
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            raise
    
    def get_dimensions(self) -> int:
        """Get the dimensions of the embedding model"""
        return self.model_info.get('dimensions', 1536)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            'model_name': self.model_name,
            'provider': self.model_info.get('provider', 'unknown'),
            'dimensions': self.get_dimensions(),
            'is_available': self.embeddings is not None,
        }
    
    @classmethod
    def get_available_models(cls) -> Dict[str, Dict[str, Any]]:
        """Get information about all available models"""
        return cls.SUPPORTED_MODELS.copy()
    
    @classmethod
    def get_model_recommendations(cls) -> Dict[str, str]:
        """Get model recommendations for different use cases"""
        return {
            'fast_and_efficient': 'sentence-transformers/all-MiniLM-L6-v2',
            'balanced_performance': 'text-embedding-3-small',
            'highest_quality': 'text-embedding-3-large',
            'multilingual': 'intfloat/multilingual-e5-large',
            'japanese_optimized': 'intfloat/multilingual-e5-large',
        }

class BatchEmbedder:
    """Handles batch embedding with progress tracking"""
    
    def __init__(self, embedding_manager: EmbeddingManager, batch_size: int = 100):
        """
        Initialize batch embedder
        
        Args:
            embedding_manager: EmbeddingManager instance
            batch_size: Number of texts to embed in each batch
        """
        self.embedding_manager = embedding_manager
        self.batch_size = batch_size
    
    def embed_documents_with_progress(
        self, 
        documents: List[Document],
        progress_callback: Optional[callable] = None
    ) -> List[Document]:
        """
        Embed documents with progress tracking
        
        Args:
            documents: List of Document objects
            progress_callback: Optional callback function for progress updates
            
        Returns:
            List of documents with embeddings in metadata
        """
        total_docs = len(documents)
        embedded_docs = []
        
        for i in range(0, total_docs, self.batch_size):
            batch = documents[i:i + self.batch_size]
            batch_texts = [doc.page_content for doc in batch]
            
            try:
                # Embed batch
                embeddings = self.embedding_manager.embed_documents(batch_texts)
                
                # Add embeddings to document metadata
                for doc, embedding in zip(batch, embeddings):
                    doc_copy = Document(
                        page_content=doc.page_content,
                        metadata={
                            **doc.metadata,
                            'embedding': embedding,
                            'embedding_model': self.embedding_manager.model_name,
                            'embedding_dimensions': len(embedding),
                        }
                    )
                    embedded_docs.append(doc_copy)
                
                # Progress callback
                if progress_callback:
                    progress = min((i + self.batch_size) / total_docs, 1.0)
                    progress_callback(progress, f"Embedded {min(i + self.batch_size, total_docs)}/{total_docs} documents")
                
                logger.info(f"Embedded batch {i//self.batch_size + 1}/{(total_docs + self.batch_size - 1)//self.batch_size}")
                
            except Exception as e:
                logger.error(f"Error embedding batch {i}-{i+self.batch_size}: {e}")
                # Skip this batch or handle error as needed
                continue
        
        logger.info(f"Successfully embedded {len(embedded_docs)}/{total_docs} documents")
        return embedded_docs

def calculate_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """
    Calculate cosine similarity between two embeddings
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        Cosine similarity score
    """
    try:
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
        
    except Exception as e:
        logger.error(f"Error calculating similarity: {e}")
        return 0.0

def get_embedding_manager(model_name: Optional[str] = None) -> EmbeddingManager:
    """
    Factory function to create embedding manager
    
    Args:
        model_name: Optional model name
        
    Returns:
        EmbeddingManager instance
    """
    return EmbeddingManager(model_name=model_name)
