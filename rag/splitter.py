"""
Text splitting functionality for Agent RAG Studio
"""
from typing import List, Dict, Any, Optional
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
    HTMLHeaderTextSplitter,
)
from langchain_core.documents import Document
import tiktoken
import logging

logger = logging.getLogger(__name__)

class DocumentSplitter:
    """Unified document splitting with multiple strategies"""
    
    def __init__(
        self,
        strategy: str = "recursive",
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        **kwargs
    ):
        """
        Initialize document splitter
        
        Args:
            strategy: Splitting strategy ('recursive', 'token', 'markdown', 'html', 'semantic')
            chunk_size: Maximum chunk size
            chunk_overlap: Overlap between chunks
            **kwargs: Additional arguments for specific splitters
        """
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.kwargs = kwargs
        
        self.splitter = self._create_splitter()
    
    def _create_splitter(self):
        """Create appropriate text splitter based on strategy"""
        if self.strategy == "recursive":
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", "。", ".", " ", ""],
                **self.kwargs
            )
        
        elif self.strategy == "token":
            encoding_name = self.kwargs.get("encoding_name", "cl100k_base")
            return TokenTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                encoding_name=encoding_name,
                **{k: v for k, v in self.kwargs.items() if k != "encoding_name"}
            )
        
        elif self.strategy == "markdown":
            headers_to_split_on = self.kwargs.get("headers_to_split_on", [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
                ("####", "Header 4"),
            ])
            return MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on,
                **{k: v for k, v in self.kwargs.items() if k != "headers_to_split_on"}
            )
        
        elif self.strategy == "html":
            headers_to_split_on = self.kwargs.get("headers_to_split_on", [
                ("h1", "Header 1"),
                ("h2", "Header 2"),
                ("h3", "Header 3"),
                ("h4", "Header 4"),
            ])
            return HTMLHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on,
                **{k: v for k, v in self.kwargs.items() if k != "headers_to_split_on"}
            )
        
        else:
            # Default to recursive
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", "。", ".", " ", ""],
            )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of Document chunks
        """
        try:
            if self.strategy in ["markdown", "html"]:
                # Header-based splitters work differently
                chunks = []
                for doc in documents:
                    doc_chunks = self.splitter.split_text(doc.page_content)
                    for i, chunk in enumerate(doc_chunks):
                        chunk_doc = Document(
                            page_content=chunk.page_content if hasattr(chunk, 'page_content') else chunk,
                            metadata={
                                **doc.metadata,
                                'chunk_id': f"{doc.metadata.get('source', 'unknown')}_{i}",
                                'chunk_index': i,
                                'total_chunks': len(doc_chunks),
                                'splitter_strategy': self.strategy,
                                **(chunk.metadata if hasattr(chunk, 'metadata') else {})
                            }
                        )
                        chunks.append(chunk_doc)
            else:
                # Character and token-based splitters
                chunks = self.splitter.split_documents(documents)
                
                # Add chunk metadata
                for i, chunk in enumerate(chunks):
                    chunk.metadata.update({
                        'chunk_id': f"{chunk.metadata.get('source', 'unknown')}_{i}",
                        'chunk_index': i,
                        'splitter_strategy': self.strategy,
                    })
            
            # Add token count estimates
            for chunk in chunks:
                chunk.metadata['estimated_tokens'] = self._estimate_tokens(chunk.page_content)
            
            logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks using {self.strategy} strategy")
            return chunks
            
        except Exception as e:
            logger.error(f"Error splitting documents: {e}")
            raise
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            # Fallback to character-based estimate
            return len(text) // 4

class SemanticSplitter:
    """Semantic-aware document splitting (experimental)"""
    
    def __init__(
        self,
        embedding_model,
        similarity_threshold: float = 0.8,
        max_chunk_size: int = 1000,
        min_chunk_size: int = 100,
    ):
        """
        Initialize semantic splitter
        
        Args:
            embedding_model: Embedding model for similarity calculation
            similarity_threshold: Threshold for semantic similarity
            max_chunk_size: Maximum chunk size
            min_chunk_size: Minimum chunk size
        """
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        
        # Use recursive splitter as base
        self.base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size // 4,  # Smaller initial chunks
            chunk_overlap=50,
            length_function=len,
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents using semantic similarity
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of semantically coherent chunks
        """
        all_chunks = []
        
        for doc in documents:
            try:
                # First, split into small segments
                segments = self.base_splitter.split_documents([doc])
                
                if len(segments) <= 1:
                    all_chunks.extend(segments)
                    continue
                
                # Group segments by semantic similarity
                semantic_chunks = self._group_by_similarity(segments)
                
                # Add metadata
                for i, chunk in enumerate(semantic_chunks):
                    chunk.metadata.update({
                        'chunk_id': f"{doc.metadata.get('source', 'unknown')}_semantic_{i}",
                        'chunk_index': i,
                        'total_chunks': len(semantic_chunks),
                        'splitter_strategy': 'semantic',
                        'estimated_tokens': len(chunk.page_content) // 4,
                    })
                
                all_chunks.extend(semantic_chunks)
                
            except Exception as e:
                logger.warning(f"Semantic splitting failed for document, falling back to recursive: {e}")
                # Fallback to recursive splitting
                fallback_splitter = DocumentSplitter(strategy="recursive")
                fallback_chunks = fallback_splitter.split_documents([doc])
                all_chunks.extend(fallback_chunks)
        
        logger.info(f"Semantic splitting created {len(all_chunks)} chunks")
        return all_chunks
    
    def _group_by_similarity(self, segments: List[Document]) -> List[Document]:
        """Group segments by semantic similarity"""
        # This is a simplified implementation
        # In practice, you'd use embeddings to calculate similarity
        grouped_chunks = []
        current_chunk_content = []
        current_chunk_metadata = segments[0].metadata.copy()
        
        for i, segment in enumerate(segments):
            current_chunk_content.append(segment.page_content)
            
            # Check if we should start a new chunk
            combined_content = "\n\n".join(current_chunk_content)
            
            if (len(combined_content) >= self.max_chunk_size or 
                i == len(segments) - 1):
                
                # Create chunk
                if len(combined_content) >= self.min_chunk_size:
                    chunk = Document(
                        page_content=combined_content,
                        metadata=current_chunk_metadata.copy()
                    )
                    grouped_chunks.append(chunk)
                
                # Reset for next chunk
                current_chunk_content = []
                if i < len(segments) - 1:
                    current_chunk_metadata = segments[i + 1].metadata.copy()
        
        return grouped_chunks

def get_splitter(
    strategy: str = "recursive",
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    **kwargs
) -> DocumentSplitter:
    """
    Factory function to create document splitter
    
    Args:
        strategy: Splitting strategy
        chunk_size: Maximum chunk size
        chunk_overlap: Overlap between chunks
        **kwargs: Additional arguments
        
    Returns:
        DocumentSplitter instance
    """
    return DocumentSplitter(
        strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **kwargs
    )

def analyze_chunks(chunks: List[Document]) -> Dict[str, Any]:
    """
    Analyze chunk statistics
    
    Args:
        chunks: List of Document chunks
        
    Returns:
        Dictionary with chunk statistics
    """
    if not chunks:
        return {
            'total_chunks': 0,
            'total_characters': 0,
            'total_tokens_estimate': 0,
            'average_chunk_size': 0,
            'strategies_used': [],
        }
    
    stats = {
        'total_chunks': len(chunks),
        'total_characters': sum(len(chunk.page_content) for chunk in chunks),
        'chunk_sizes': [len(chunk.page_content) for chunk in chunks],
        'strategies_used': [],
    }
    
    # Strategy analysis
    strategies = set()
    for chunk in chunks:
        strategy = chunk.metadata.get('splitter_strategy', 'unknown')
        strategies.add(strategy)
    stats['strategies_used'] = list(strategies)
    
    # Size statistics
    if stats['chunk_sizes']:
        stats['average_chunk_size'] = sum(stats['chunk_sizes']) / len(stats['chunk_sizes'])
        stats['min_chunk_size'] = min(stats['chunk_sizes'])
        stats['max_chunk_size'] = max(stats['chunk_sizes'])
        stats['median_chunk_size'] = sorted(stats['chunk_sizes'])[len(stats['chunk_sizes']) // 2]
    
    # Token estimate
    stats['total_tokens_estimate'] = sum(
        chunk.metadata.get('estimated_tokens', len(chunk.page_content) // 4)
        for chunk in chunks
    )
    
    return stats
