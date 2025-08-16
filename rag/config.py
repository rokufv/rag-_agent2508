"""
Configuration management for Agent RAG Studio
"""
import os
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

# Setup logging
logger = logging.getLogger(__name__)

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
        
        logger.info("Starting configuration loading process...")
        
        # Try to get from Streamlit secrets first
        if STREAMLIT_AVAILABLE and hasattr(st, 'secrets'):
            try:
                logger.info("Streamlit secrets available, attempting to load...")
                secrets_data = st.secrets.get('api_keys', {})
                app_config = st.secrets.get('app_config', {})
                langsmith_config = st.secrets.get('langsmith', {})
                
                logger.info(f"Secrets data keys: {list(secrets_data.keys()) if secrets_data else 'None'}")
                logger.info(f"App config keys: {list(app_config.keys()) if app_config else 'None'}")
                logger.info(f"LangSmith config keys: {list(langsmith_config.keys()) if langsmith_config else 'None'}")
                
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
                logger.info(f"API keys from secrets: OpenAI={bool(config_data.get('openai_api_key'))}, COHERE={bool(config_data.get('cohere_api_key'))}, LANGSMITH={bool(config_data.get('langsmith_api_key'))}, SERPAPI={bool(config_data.get('serpapi_api_key'))}")
                    
            except Exception as e:
                logger.error(f"Error loading Streamlit secrets: {e}")
        else:
            logger.warning("Streamlit secrets not available")
        
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
        logger.info(f"Final API keys status: OpenAI={bool(config_data.get('openai_api_key'))}, COHERE={bool(config_data.get('cohere_api_key'))}, LANGSMITH={bool(config_data.get('langsmith_api_key'))}, SERPAPI={bool(config_data.get('serpapi_api_key'))}")
        
        # 設定オブジェクトを作成
        config_instance = cls(**config_data)
        logger.info(f"Configuration object created: {type(config_instance)}")
        
        return config_instance
    
    def setup_environment(self):
        """Setup environment variables for LangChain and other services"""
        # 確実に環境変数を設定
        if self.openai_api_key:
            os.environ['OPENAI_API_KEY'] = self.openai_api_key
            logger.info("OpenAI API key set in environment")
        else:
            logger.warning("OpenAI API key not available")
        
        if self.cohere_api_key:
            os.environ['COHERE_API_KEY'] = self.cohere_api_key
            logger.info("Cohere API key set in environment")
        else:
            logger.warning("Cohere API key not available")
            
        # LangSmith tracing only if explicitly enabled
        if self.langsmith_api_key and self.langsmith_enabled:
            os.environ['LANGCHAIN_API_KEY'] = self.langsmith_api_key
            os.environ['LANGCHAIN_TRACING_V2'] = 'true'
            os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
            os.environ['LANGCHAIN_PROJECT'] = 'agent-rag-studio'
            logger.info("LangSmith tracing enabled")
        else:
            # Ensure tracing is disabled to avoid 403 noise
            os.environ['LANGCHAIN_TRACING_V2'] = 'false'
            logger.info("LangSmith tracing disabled")
        
        if self.serpapi_api_key:
            os.environ['SERPAPI_API_KEY'] = self.serpapi_api_key
            logger.info("SerpAPI key set in environment")
        else:
            logger.warning("SerpAPI key not available")
        
        # 設定の検証をログ出力
        logger.info(f"Environment setup complete. API keys: OpenAI={bool(os.getenv('OPENAI_API_KEY'))}, Cohere={bool(os.getenv('COHERE_API_KEY'))}, LangSmith={bool(os.getenv('LANGCHAIN_API_KEY'))}, SerpAPI={bool(os.getenv('SERPAPI_API_KEY'))}")
    
    def validate_keys(self) -> Dict[str, bool]:
        """Validate that required API keys are present"""
        validation_results = {
            'openai': bool(self.openai_api_key),
            'cohere': bool(self.cohere_api_key),
            'langsmith': bool(self.langsmith_api_key),
            'serpapi': bool(self.serpapi_api_key),
        }
        
        # 詳細な検証情報をログ出力
        logger.info(f"Key validation results: {validation_results}")
        logger.info(f"OpenAI key length: {len(self.openai_api_key) if self.openai_api_key else 0}")
        logger.info(f"Cohere key length: {len(self.cohere_api_key) if self.cohere_api_key else 0}")
        logger.info(f"LangSmith key length: {len(self.langsmith_api_key) if self.langsmith_api_key else 0}")
        logger.info(f"SerpAPI key length: {len(self.serpapi_api_key) if self.serpapi_api_key else 0}")
        
        return validation_results

# Global config instance
_config = None
_config_initialized = False

def get_config() -> RAGConfig:
    """Get the global configuration instance"""
    global _config, _config_initialized
    
    if _config is None or not _config_initialized:
        logger.info("Initializing global configuration...")
        _config = RAGConfig.from_env()
        _config.setup_environment()
        _config_initialized = True
        logger.info("Global configuration initialized and environment variables set")
        
        # 設定の状態を詳細にログ出力
        logger.info(f"Config object ID: {id(_config)}")
        logger.info(f"Config object type: {type(_config)}")
        logger.info(f"OpenAI API key available: {bool(_config.openai_api_key)}")
        logger.info(f"Environment OPENAI_API_KEY: {bool(os.getenv('OPENAI_API_KEY'))}")
        logger.info(f"Environment COHERE_API_KEY: {bool(os.getenv('COHERE_API_KEY'))}")
        logger.info(f"Environment LANGCHAIN_API_KEY: {bool(os.getenv('LANGCHAIN_API_KEY'))}")
        logger.info(f"Environment SERPAPI_API_KEY: {bool(os.getenv('SERPAPI_API_KEY'))}")
    
    return _config

def update_config(**kwargs) -> RAGConfig:
    """Update the global configuration"""
    global _config, _config_initialized
    
    if _config is None:
        _config = get_config()
    
    for key, value in kwargs.items():
        if hasattr(_config, key):
            setattr(_config, key, value)
            logger.info(f"Updated config.{key} = {value}")
    
    # 環境変数を再設定
    _config.setup_environment()
    return _config

def reset_config():
    """Reset the global configuration (for testing/debugging)"""
    global _config, _config_initialized
    _config = None
    _config_initialized = False
    logger.info("Global configuration reset")
