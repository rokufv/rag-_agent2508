"""
Configuration management for Agent RAG Studio
"""
import os
from typing import Optional, Dict, Any
from dataclasses import dataclass
import streamlit as st

# Load environment variables safely
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, skip loading
    pass

@dataclass
class RAGConfig:
    """RAG configuration settings (Pydantic非依存)"""
    # API Keys
    openai_api_key: Optional[str] = None
    cohere_api_key: Optional[str] = None
    langsmith_api_key: Optional[str] = None
    # LangSmith tracing toggle
    langsmith_enabled: bool = False
    serpapi_api_key: Optional[str] = None

    # Models
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    default_llm: str = "gpt-4o-mini"

    # Storage
    data_dir: str = "./data"
    logs_dir: str = "./logs"
    vector_store: str = "faiss"  # chroma or faiss (FAISS is more compatible)
    demo_mode: bool = False

    # RAG Parameters
    chunk_size: int = 1000
    chunk_overlap: int = 100
    top_k: int = 6
    rerank_top_r: int = 3
    use_reranking: bool = True

    # Generation Parameters
    temperature: float = 0.2
    max_tokens: int = 1000

    # Agent Parameters
    max_loops: int = 3
    confidence_threshold: float = 0.6
    
    @classmethod
    def from_env(cls) -> "RAGConfig":
        """Create config from environment variables and Streamlit secrets"""
        config_data = {}
        
        # Try to get from Streamlit secrets first
        if hasattr(st, 'secrets'):
            try:
                config_data.update({
                    'openai_api_key': st.secrets.get('api_keys', {}).get('openai_api_key'),
                    'cohere_api_key': st.secrets.get('api_keys', {}).get('cohere_api_key'),
                    'langsmith_api_key': st.secrets.get('api_keys', {}).get('langsmith_api_key'),
                    'serpapi_api_key': st.secrets.get('api_keys', {}).get('serpapi_api_key'),
                    'embedding_model': st.secrets.get('app_config', {}).get('embedding_model', 'text-embedding-3-small'),
                    'default_llm': st.secrets.get('app_config', {}).get('default_llm', 'gpt-4o-mini'),
                    'data_dir': st.secrets.get('app_config', {}).get('data_dir', './data'),
                    'logs_dir': st.secrets.get('app_config', {}).get('logs_dir', './logs'),
                    'vector_store': st.secrets.get('app_config', {}).get('vector_store', 'chroma'),
                    'demo_mode': st.secrets.get('app_config', {}).get('demo_mode', False),
                    'use_reranking': st.secrets.get('app_config', {}).get('use_reranking'),
                    'langsmith_enabled': st.secrets.get('app_config', {}).get('langsmith_enabled'),
                })
            except Exception:
                pass
        
        # Fallback to environment variables
        config_data.update({
            'openai_api_key': config_data.get('openai_api_key') or os.getenv('OPENAI_API_KEY'),
            'cohere_api_key': config_data.get('cohere_api_key') or os.getenv('COHERE_API_KEY'),
            'langsmith_api_key': config_data.get('langsmith_api_key') or os.getenv('LANGCHAIN_API_KEY'),
            'serpapi_api_key': config_data.get('serpapi_api_key') or os.getenv('SERPAPI_API_KEY'),
            'embedding_model': config_data.get('embedding_model') or os.getenv('RAG_EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'),
            'default_llm': config_data.get('default_llm') or os.getenv('RAG_DEFAULT_LLM', 'gpt-4o-mini'),
            'data_dir': config_data.get('data_dir') or os.getenv('RAG_DATA_DIR', './data'),
            'logs_dir': config_data.get('logs_dir') or os.getenv('RAG_LOGS_DIR', './logs'),
            'vector_store': config_data.get('vector_store') or os.getenv('RAG_VECTOR_STORE', 'chroma'),
            'demo_mode': config_data.get('demo_mode') or os.getenv('RAG_DEMO_MODE', 'false').lower() == 'true',
        })
        
        # 強制的に無効化（設定ファイルの値が優先）
        if config_data.get('use_reranking') is None:
            config_data['use_reranking'] = False
        if config_data.get('langsmith_enabled') is None:
            config_data['langsmith_enabled'] = False
        
        # Filter out None values
        config_data = {k: v for k, v in config_data.items() if v is not None}
        
        return cls(**config_data)
    
    def setup_environment(self):
        """Setup environment variables for LangChain and other services"""
        if self.openai_api_key:
            os.environ['OPENAI_API_KEY'] = self.openai_api_key
        
        if self.cohere_api_key:
            os.environ['COHERE_API_KEY'] = self.cohere_api_key
            
        # LangSmith tracing only if explicitly enabled
        if self.langsmith_api_key and self.langsmith_enabled:
            os.environ['LANGCHAIN_API_KEY'] = self.langsmith_api_key
            os.environ['LANGCHAIN_TRACING_V2'] = 'true'
            os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
            os.environ['LANGCHAIN_PROJECT'] = 'agent-rag-studio'
        else:
            # Ensure tracing is disabled to avoid 403 noise
            os.environ['LANGCHAIN_TRACING_V2'] = 'false'
        
        if self.serpapi_api_key:
            os.environ['SERPAPI_API_KEY'] = self.serpapi_api_key
    
    def validate_keys(self) -> Dict[str, bool]:
        """Validate that required API keys are present"""
        return {
            'openai': bool(self.openai_api_key),
            'cohere': bool(self.cohere_api_key),
            'langsmith': bool(self.langsmith_api_key),
            'serpapi': bool(self.serpapi_api_key),
        }

# Global config instance
_config = None

def get_config() -> RAGConfig:
    """Get the global configuration instance"""
    global _config
    if _config is None:
        _config = RAGConfig.from_env()
        _config.setup_environment()
    return _config

def update_config(**kwargs) -> RAGConfig:
    """Update the global configuration"""
    global _config
    if _config is None:
        _config = RAGConfig.from_env()
    
    for key, value in kwargs.items():
        if hasattr(_config, key):
            setattr(_config, key, value)
    
    _config.setup_environment()
    return _config
