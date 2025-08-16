"""
Document loading functionality for Agent RAG Studio
"""
import os
import tempfile
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
import streamlit as st

try:
    from langchain_community.document_loaders import (
        PyPDFLoader,
        UnstructuredHTMLLoader,
        UnstructuredMarkdownLoader,
        TextLoader,
        UnstructuredWordDocumentLoader,
    )
    LOADERS_AVAILABLE = True
except ImportError:
    LOADERS_AVAILABLE = False
    # Fallback to basic text loader only
    class PyPDFLoader:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyPDFLoader not available")
    class UnstructuredHTMLLoader:
        def __init__(self, *args, **kwargs):
            raise ImportError("UnstructuredHTMLLoader not available")
    class UnstructuredMarkdownLoader:
        def __init__(self, *args, **kwargs):
            raise ImportError("UnstructuredMarkdownLoader not available")
    class TextLoader:
        def __init__(self, *args, **kwargs):
            raise ImportError("TextLoader not available")
    class UnstructuredWordDocumentLoader:
        def __init__(self, *args, **kwargs):
            raise ImportError("UnstructuredWordDocumentLoader not available")

from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)

class DocumentLoader:
    """Unified document loader for multiple file types"""
    
    SUPPORTED_EXTENSIONS = {
        '.pdf': PyPDFLoader,
        '.html': UnstructuredHTMLLoader,
        '.htm': UnstructuredHTMLLoader,
        '.md': UnstructuredMarkdownLoader,
        '.markdown': UnstructuredMarkdownLoader,
        '.txt': TextLoader,
        '.docx': UnstructuredWordDocumentLoader,
        '.doc': UnstructuredWordDocumentLoader,
    }
    
    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """Get list of supported file extensions"""
        return list(cls.SUPPORTED_EXTENSIONS.keys())
    
    @classmethod
    def is_supported(cls, file_path: Union[str, Path]) -> bool:
        """Check if file type is supported"""
        suffix = Path(file_path).suffix.lower()
        return suffix in cls.SUPPORTED_EXTENSIONS
    
    @classmethod
    def load_document(cls, file_path: Union[str, Path], **kwargs) -> List[Document]:
        """
        Load a single document
        
        Args:
            file_path: Path to the document
            **kwargs: Additional arguments for the loader
            
        Returns:
            List of Document objects
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        
        if suffix not in cls.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {suffix}")
        
        loader_class = cls.SUPPORTED_EXTENSIONS[suffix]
        
        try:
            loader = loader_class(str(file_path), **kwargs)
            documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    'source': str(file_path),
                    'file_name': file_path.name,
                    'file_type': suffix,
                    'file_size': file_path.stat().st_size if file_path.exists() else 0,
                })
            
            logger.info(f"Loaded {len(documents)} documents from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            raise
    
    @classmethod
    def load_uploaded_file(cls, uploaded_file, save_dir: Optional[str] = None) -> List[Document]:
        """
        Load document from Streamlit uploaded file
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            save_dir: Directory to save the uploaded file
            
        Returns:
            List of Document objects
        """
        if save_dir:
            # Save file permanently
            os.makedirs(save_dir, exist_ok=True)
            file_path = Path(save_dir) / uploaded_file.name
            
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            documents = cls.load_document(file_path)
            
        else:
            # Use temporary file
            with tempfile.NamedTemporaryFile(
                delete=False, 
                suffix=Path(uploaded_file.name).suffix
            ) as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_file.flush()
                
                documents = cls.load_document(tmp_file.name)
                
                # Clean up
                os.unlink(tmp_file.name)
        
        return documents
    
    @classmethod
    def load_directory(cls, directory: Union[str, Path], recursive: bool = True) -> List[Document]:
        """
        Load all supported documents from a directory
        
        Args:
            directory: Directory path
            recursive: Whether to search recursively
            
        Returns:
            List of Document objects
        """
        directory = Path(directory)
        documents = []
        
        if recursive:
            pattern = "**/*"
        else:
            pattern = "*"
        
        for file_path in directory.glob(pattern):
            if file_path.is_file() and cls.is_supported(file_path):
                try:
                    docs = cls.load_document(file_path)
                    documents.extend(docs)
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")
                    continue
        
        logger.info(f"Loaded {len(documents)} documents from {directory}")
        return documents

class DocumentStats:
    """Document statistics and metadata"""
    
    @staticmethod
    def analyze_documents(documents: List[Document]) -> Dict[str, Any]:
        """
        Analyze document collection and return statistics
        
        Args:
            documents: List of Document objects
            
        Returns:
            Dictionary with statistics
        """
        if not documents:
            return {
                'total_documents': 0,
                'total_characters': 0,
                'total_tokens_estimate': 0,
                'file_types': {},
                'sources': [],
                'average_doc_length': 0,
            }
        
        stats = {
            'total_documents': len(documents),
            'total_characters': sum(len(doc.page_content) for doc in documents),
            'file_types': {},
            'sources': [],
            'source_files': set(),
        }
        
        for doc in documents:
            # File type stats
            file_type = doc.metadata.get('file_type', 'unknown')
            stats['file_types'][file_type] = stats['file_types'].get(file_type, 0) + 1
            
            # Source tracking
            source = doc.metadata.get('source', 'unknown')
            if source not in stats['sources']:
                stats['sources'].append(source)
            
            file_name = doc.metadata.get('file_name', 'unknown')
            stats['source_files'].add(file_name)
        
        # Calculate estimates
        stats['total_tokens_estimate'] = stats['total_characters'] // 4  # Rough estimate
        stats['average_doc_length'] = stats['total_characters'] / len(documents)
        stats['unique_sources'] = len(stats['sources'])
        stats['unique_files'] = len(stats['source_files'])
        
        return stats
    
    @staticmethod
    def format_stats_for_display(stats: Dict[str, Any]) -> str:
        """Format statistics for display in Streamlit"""
        if stats['total_documents'] == 0:
            return "📄 ドキュメントがありません"
        
        formatted = f"""
        📊 **ドキュメント統計**
        - 📄 総ドキュメント数: {stats['total_documents']:,}
        - 📁 ユニークファイル数: {stats['unique_files']:,}
        - 📝 総文字数: {stats['total_characters']:,}
        - 🔤 推定トークン数: {stats['total_tokens_estimate']:,}
        - 📏 平均文書長: {stats['average_doc_length']:.0f} 文字
        
        **ファイルタイプ別:**
        """
        
        for file_type, count in stats['file_types'].items():
            formatted += f"\n        - {file_type}: {count:,} ドキュメント"
        
        return formatted.strip()
