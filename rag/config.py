"""
Configuration management for Agent RAG Studio
"""
import os
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

# Try to import streamlit for secrets
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    # Mock st for non-streamlit environments
    class MockStreamlit:
        @staticmethod
        def secrets():
            return {}
    st = MockStreamlit()

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
    openai_api_key: str = field(default="")
    cohere_api_key: str = field(default="")
    langsmith_api_key: str = field(default="")
    serpapi_api_key: str = field(default="")
    bing_search_api_key: str = field(default="")
    
    # App Configuration
    data_dir: str = field(default="./data")
    logs_dir: str = field(default="./logs")
    vector_store: str = field(default="faiss")  # chroma or faiss (FAISS is more compatible)
    embedding_model: str = field(default="text-embedding-3-small")
    default_llm: str = field(default="gpt-4o-mini")
    demo_mode: bool = field(default=False)
    app_password: str = field(default="")
    use_reranking: bool = field(default=False)
    langsmith_enabled: bool = field(default=False)
    
    # Generation Parameters
    temperature: float = field(default=0.7)
    max_tokens: int = field(default=1000)
    
    # Retrieval Parameters
    top_k: int = field(default=4)
    rerank_top_r: int = field(default=3)

    # Agent Parameters
    max_loops: int = 3
    confidence_threshold: float = 0.6
    
    @classmethod
    def from_env(cls) -> "RAGConfig":
        """Create config from environment variables and Streamlit secrets"""
        config_data = {}
        
        # Try to get from Streamlit secrets first
        if STREAMLIT_AVAILABLE and hasattr(st, 'secrets'):
            try:
                secrets_data = st.secrets.get('api_keys', {})
                app_config = st.secrets.get('app_config', {})
                langsmith_config = st.secrets.get('langsmith', {})
                
                # Load API keys from secrets
                config_data.update({
                    'openai_api_key': secrets_data.get('openai_api_key', ''),
                    'cohere_api_key': secrets_data.get('cohere_api_key', ''),
                    'langsmith_api_key': secrets_data.get('langsmith_api_key', ''),
                    'serpapi_api_key': secrets_data.get('serpapi_api_key', ''),
                    'bing_search_api_key': secrets_data.get('bing_search_api_key', ''),
                })
                
                # Load app config from secrets
                config_data.update({
                    'data_dir': app_config.get('data_dir', './data'),
                    'logs_dir': app_config.get('logs_dir', './logs'),
                    'vector_store': app_config.get('vector_store', 'faiss'),
                    'embedding_model': app_config.get('embedding_model', 'text-embedding-3-small'),
                    'default_llm': app_config.get('default_llm', 'gpt-4o-mini'),
                    'demo_mode': app_config.get('demo_mode', False),
                    'use_reranking': app_config.get('use_reranking', False),
                    'langsmith_enabled': app_config.get('langsmith_enabled', False),
                })
                
                # Load LangSmith config from secrets
                if langsmith_config.get('tracing_v2') is not None:
                    config_data['langsmith_enabled'] = langsmith_config['tracing_v2']
                
                logger.info(f"Loaded config from Streamlit secrets: {list(config_data.keys())}")
                    
            except Exception as e:
                logger.warning(f"Error loading Streamlit secrets: {e}")
        
        # Fallback to environment variables if secrets not available
        if not config_data:
            logger.info("Falling back to environment variables")
            config_data.update({
                'openai_api_key': os.getenv('OPENAI_API_KEY', ''),
                'cohere_api_key': os.getenv('COHERE_API_KEY', ''),
                'langsmith_api_key': os.getenv('LANGSMITH_API_KEY', ''),
                'serpapi_api_key': os.getenv('SERPAPI_API_KEY', ''),
                'bing_search_api_key': os.getenv('BING_SEARCH_API_KEY', ''),
                'data_dir': os.getenv('RAG_DATA_DIR', './data'),
                'logs_dir': os.getenv('RAG_LOGS_DIR', './logs'),
                'vector_store': os.getenv('RAG_VECTOR_STORE', 'faiss'),
                'embedding_model': os.getenv('RAG_EMBEDDING_MODEL', 'text-embedding-3-small'),
                'default_llm': os.getenv('RAG_DEFAULT_LLM', 'gpt-4o-mini'),
                'demo_mode': os.getenv('RAG_DEMO_MODE', 'false').lower() == 'true',
                'app_password': os.getenv('RAG_APP_PASSWORD', ''),
                'use_reranking': os.getenv('RAG_USE_RERANKING', 'false').lower() == 'true',
                'langsmith_enabled': os.getenv('LANGSMITH_ENABLED', 'false').lower() == 'true',
            })
        
        # Filter out None values (but keep empty strings for API keys)
        config_data = {k: v for k, v in config_data.items() if v is not None}
        
        logger.info(f"Final config keys: {list(config_data.keys())}")
        logger.info(f"API keys status: OpenAI={bool(config_data.get('openai_api_key'))}, COHERE={bool(config_data.get('cohere_api_key'))}, LANGSMITH={bool(config_data.get('langsmith_api_key'))}, SERPAPI={bool(config_data.get('serpapi_api_key'))}")
        
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
