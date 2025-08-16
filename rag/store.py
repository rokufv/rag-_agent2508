"""
Vector store functionality for Agent RAG Studio
"""
import os
import pickle
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import logging
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

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

try:
    from langchain_community.vectorstores import Chroma, FAISS
    VECTORSTORES_AVAILABLE = True
except ImportError:
    VECTORSTORES_AVAILABLE = False
    # Fallback classes
    class Chroma:
        def __init__(self, *args, **kwargs):
            raise ImportError("Chroma not available")
    class FAISS:
        def __init__(self, *args, **kwargs):
            raise ImportError("FAISS not available")
        @classmethod
        def load_local(cls, *args, **kwargs):
            raise ImportError("FAISS not available")

try:
    from langchain_core.vectorstores import VectorStore
    CORE_VECTORSTORES_AVAILABLE = True
except ImportError:
    CORE_VECTORSTORES_AVAILABLE = False
    # Fallback VectorStore class
    class VectorStore:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("VectorStore not available")

try:
    from langchain_core.embeddings import Embeddings
    CORE_EMBEDDINGS_AVAILABLE = True
except ImportError:
    CORE_EMBEDDINGS_AVAILABLE = False
    # Fallback Embeddings class
    class Embeddings:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("Embeddings not available")

from .config import RAGConfig
from .embedder import EmbeddingManager
# HybridRetriever is imported later to avoid circular import

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Manages different vector store implementations"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.persist_directory = Path(config.data_dir) / "vector_store"
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding manager first
        self.embedding_manager = EmbeddingManager(config)
        
        # Set store type from config
        self.store_type = config.vector_store.lower()
        
        # Initialize vector store based on config
        if self.store_type == "chroma":
            self._load_or_create_chroma()
        elif self.store_type == "faiss":
            self._load_or_create_faiss()
        else:
            logger.warning(f"Unknown vector store type: {self.store_type}. Falling back to FAISS.")
            self.store_type = "faiss"
            self._load_or_create_faiss()
        
        # Initialize hybrid retriever
        self.hybrid_retriever = None # Initialize to None, will be set later
    
    def _ensure_hybrid_retriever(self):
        """Ensure hybrid retriever is initialized"""
        if self.hybrid_retriever is None:
            from .retriever import HybridRetriever
            self.hybrid_retriever = HybridRetriever(
                vector_retriever=self._create_vector_retriever(),
                bm25_retriever=self._create_bm25_retriever(),
                config=self.config
            )
        return self.hybrid_retriever
    
    def _load_or_create_chroma(self):
        """Load or create Chroma vector store"""
        chroma_db_path = self.persist_directory / "chroma_db"
        
        try:
            # Check SQLite version compatibility first
            import sqlite3
            sqlite_version = tuple(map(int, sqlite3.sqlite_version.split('.')))
            if sqlite_version < (3, 35, 0):
                logger.warning(f"SQLite version {sqlite3.sqlite_version} is too old for ChromaDB. Switching to FAISS.")
                self.store_type = "faiss"
                self._load_or_create_faiss()
                return
            
            # Check if ChromaDB is available
            if not VECTORSTORES_AVAILABLE:
                logger.warning("ChromaDB not available. Switching to FAISS.")
                self.store_type = "faiss"
                self._load_or_create_faiss()
                return
            
            # Try to load existing store
            if chroma_db_path.exists() and any(chroma_db_path.iterdir()):
                self.vector_store = Chroma(
                    persist_directory=str(chroma_db_path),
                    embedding_function=self.embedding_manager.embeddings,
                )
                logger.info(f"Loaded existing Chroma store from {chroma_db_path}")
            else:
                # Create new store
                self.vector_store = Chroma(
                    persist_directory=str(chroma_db_path),
                    embedding_function=self.embedding_manager.embeddings,
                )
                logger.info(f"Created new Chroma store at {chroma_db_path}")
                
        except Exception as e:
            logger.error(f"Error with Chroma store: {e}")
            # Any error with ChromaDB should trigger fallback to FAISS
            if "sqlite" in str(e).lower() or "unsupported version" in str(e).lower() or "chroma" in str(e).lower():
                logger.warning("ChromaDB error detected. Switching to FAISS.")
                self.store_type = "faiss"
                self._load_or_create_faiss()
            else:
                # For any other error, also fallback to FAISS
                logger.warning("Unexpected error with ChromaDB. Switching to FAISS.")
                self.store_type = "faiss"
                self._load_or_create_faiss()
    
    def _load_or_create_faiss(self):
        """Load or create FAISS vector store"""
        faiss_index_path = self.persist_directory / "faiss_index"
        
        try:
            # Try to load existing store
            if faiss_index_path.exists():
                self.vector_store = FAISS.load_local(
                    str(faiss_index_path),
                    self.embedding_manager.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"Loaded existing FAISS store from {faiss_index_path}")
            else:
                # FAISS requires at least one document to initialize
                # We'll create it when first documents are added
                self.vector_store = None
                logger.info("FAISS store will be created when first documents are added")
                
        except Exception as e:
            logger.error(f"Error with FAISS store: {e}")
            # Reset to None if loading failed
            self.vector_store = None
            # Try to create a minimal FAISS store
            try:
                from langchain_community.vectorstores import FAISS
                # Create a dummy document to initialize FAISS
                dummy_doc = Document(page_content="dummy", metadata={})
                self.vector_store = FAISS.from_documents(
                    [dummy_doc], 
                    self.embedding_manager.embeddings
                )
                logger.info("Created minimal FAISS store for compatibility")
            except Exception as fallback_error:
                logger.error(f"FAISS fallback also failed: {fallback_error}")
                self.vector_store = None
    
    def add_documents(
        self, 
        documents: List[Document],
        batch_size: int = 100,
        progress_callback: Optional[callable] = None
    ) -> List[str]:
        """
        Add documents to the vector store
        
        Args:
            documents: List of Document objects
            batch_size: Number of documents to process in each batch
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of document IDs
        """
        if not documents:
            return []
        
        try:
            all_ids = []
            total_docs = len(documents)
            
            # Process in batches
            for i in range(0, total_docs, batch_size):
                batch = documents[i:i + batch_size]
                
                if self.store_type == "chroma":
                    ids = self._add_to_chroma(batch)
                elif self.store_type == "faiss":
                    ids = self._add_to_faiss(batch)
                else:
                    raise ValueError(f"Unsupported store type: {self.store_type}")
                
                all_ids.extend(ids)
                
                # Progress callback
                if progress_callback:
                    progress = min((i + batch_size) / total_docs, 1.0)
                    progress_callback(
                        progress, 
                        f"Added {min(i + batch_size, total_docs)}/{total_docs} documents to vector store"
                    )
                
                logger.info(f"Added batch {i//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size} to vector store")
            
            # Persist the store
            self.persist()
            
            # Save metadata
            self._save_metadata({
                'total_documents': len(all_ids),
                'store_type': self.store_type,
                'embedding_model': self.embedding_manager.model_name,
                'last_updated': str(Path().resolve()),
            })
            
            logger.info(f"Successfully added {len(all_ids)} documents to {self.store_type} store")
            return all_ids
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise
    
    def _add_to_chroma(self, documents: List[Document]) -> List[str]:
        """Add documents to Chroma store"""
        return self.vector_store.add_documents(documents)
    
    def _add_to_faiss(self, documents: List[Document]) -> List[str]:
        """Add documents to FAISS store"""
        if self.vector_store is None:
            # Create FAISS store with first batch
            self.vector_store = FAISS.from_documents(
                documents,
                self.embedding_manager.embeddings
            )
            ids = [f"doc_{i}" for i in range(len(documents))]
        else:
            # Add to existing store
            ids = self.vector_store.add_documents(documents)
        
        return ids
    
    def similarity_search(
        self,
        query: str,
        k: int = 6,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Document]:
        """
        Perform similarity search
        
        Args:
            query: Query string
            k: Number of documents to return
            filter: Optional metadata filter
            **kwargs: Additional search parameters
            
        Returns:
            List of similar documents
        """
        if not self.vector_store:
            logger.warning("Vector store not initialized")
            return []
        
        try:
            if filter and self.store_type == "chroma":
                # Chroma supports metadata filtering
                results = self.vector_store.similarity_search(
                    query=query,
                    k=k,
                    filter=filter,
                    **kwargs
                )
            else:
                # Basic similarity search
                results = self.vector_store.similarity_search(
                    query=query,
                    k=k,
                    **kwargs
                )
            
            logger.info(f"Found {len(results)} similar documents for query")
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 6,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search with scores
        
        Args:
            query: Query string
            k: Number of documents to return
            filter: Optional metadata filter
            **kwargs: Additional search parameters
            
        Returns:
            List of (document, score) tuples
        """
        if not self.vector_store:
            logger.warning("Vector store not initialized")
            return []
        
        try:
            if filter and self.store_type == "chroma":
                # Chroma supports metadata filtering
                results = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=k,
                    filter=filter,
                    **kwargs
                )
            else:
                # Basic similarity search with scores
                results = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=k,
                    **kwargs
                )
            
            logger.info(f"Found {len(results)} similar documents with scores")
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search with scores: {e}")
            return []
    
    def persist(self):
        """Persist the vector store to disk"""
        try:
            if self.store_type == "chroma" and self.vector_store:
                self.vector_store.persist()
                
            elif self.store_type == "faiss" and self.vector_store:
                faiss_index_path = self.persist_directory / "faiss_index"
                self.vector_store.save_local(str(faiss_index_path))
            
            logger.info(f"Persisted {self.store_type} vector store")
            
        except Exception as e:
            logger.error(f"Error persisting vector store: {e}")
            raise
    
    def delete(self, ids: List[str]) -> bool:
        """
        Delete documents by IDs
        
        Args:
            ids: List of document IDs to delete
            
        Returns:
            Success status
        """
        try:
            if self.store_type == "chroma" and self.vector_store:
                self.vector_store.delete(ids)
                self.persist()
                logger.info(f"Deleted {len(ids)} documents from Chroma store")
                return True
                
            elif self.store_type == "faiss":
                logger.warning("FAISS does not support document deletion")
                return False
            
            return False
            
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False
    
    def clear(self):
        """Clear all documents from the vector store"""
        try:
            if self.store_type == "chroma":
                # Recreate Chroma store
                chroma_db_path = self.persist_directory / "chroma_db"
                if chroma_db_path.exists():
                    import shutil
                    shutil.rmtree(chroma_db_path)
                
                self.vector_store = Chroma(
                    persist_directory=str(chroma_db_path),
                    embedding_function=self.embedding_manager.embeddings,
                )
                
            elif self.store_type == "faiss":
                # Remove FAISS files
                faiss_index_path = self.persist_directory / "faiss_index"
                if faiss_index_path.exists():
                    import shutil
                    shutil.rmtree(faiss_index_path)
                
                self.vector_store = None
            
            # Clear metadata
            if self._metadata_file.exists():
                self._metadata_file.unlink()
            
            logger.info(f"Cleared {self.store_type} vector store")
            
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        try:
            stats = {
                'store_type': self.store_type,
                'embedding_model': self.embedding_manager.model_name,
                'embedding_dimensions': self.embedding_manager.get_dimensions(),
                'persist_directory': str(self.persist_directory),
                'is_initialized': self.vector_store is not None,
            }
            
            # Try to get document count
            if self.vector_store:
                if self.store_type == "chroma":
                    try:
                        collection = self.vector_store._collection
                        stats['document_count'] = collection.count()
                    except:
                        stats['document_count'] = 'unknown'
                        
                elif self.store_type == "faiss":
                    try:
                        stats['document_count'] = self.vector_store.index.ntotal
                    except:
                        stats['document_count'] = 'unknown'
            else:
                stats['document_count'] = 0
            
            # Load metadata if available
            metadata = self._load_metadata()
            if metadata:
                stats.update(metadata)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting vector store stats: {e}")
            return {'error': str(e)}
    
    def _save_metadata(self, metadata: Dict[str, Any]):
        """Save metadata to file"""
        try:
            with open(self._metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def _load_metadata(self) -> Optional[Dict[str, Any]]:
        """Load metadata from file"""
        try:
            if self._metadata_file.exists():
                with open(self._metadata_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
        return None

    def get_hybrid_retriever(self):
        """Get the hybrid retriever"""
        return self._ensure_hybrid_retriever()
    
    def search(self, query: str, k: int = 4, filter: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Search documents using hybrid retrieval"""
        return self._ensure_hybrid_retriever().get_relevant_documents(query)

def get_vector_store(
    store_type: str = "chroma",
    embedding_manager: Optional[EmbeddingManager] = None,
    persist_directory: Optional[str] = None,
) -> VectorStoreManager:
    """
    Factory function to create vector store manager
    
    Args:
        store_type: Type of vector store
        embedding_manager: Optional embedding manager
        persist_directory: Optional persistence directory
        
    Returns:
        VectorStoreManager instance
    """
    return VectorStoreManager(
        store_type=store_type,
        embedding_manager=embedding_manager,
        persist_directory=persist_directory,
    )
