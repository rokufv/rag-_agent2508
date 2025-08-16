"""
Agent RAG Studio - Main Streamlit Application
"""
import streamlit as st
import os
import logging
from pathlib import Path
import traceback

# 環境変数を確実に無効化（警告停止のため）
os.environ['LANGCHAIN_TRACING_V2'] = 'false'
os.environ['LANGCHAIN_ENDPOINT'] = ''
os.environ['LANGCHAIN_API_KEY'] = ''
os.environ['LANGSMITH_ENABLED'] = 'false'
os.environ['COHERE_API_KEY'] = ''

# Resolve a writable log directory (Streamlit Cloud mounts /mount/src as read-only)
def _resolve_log_dir() -> Path:
    # 1) Explicit env var
    env_dir = os.getenv('STREAMLIT_LOG_DIR')
    if env_dir:
        p = Path(env_dir)
        try:
            p.mkdir(parents=True, exist_ok=True)
            return p
        except Exception:
            pass
    # 2) Streamlit Cloud writable mount
    mount_data = Path('/mount/data')
    if mount_data.exists():
        p = mount_data / 'logs'
        try:
            p.mkdir(parents=True, exist_ok=True)
            return p
        except Exception:
            pass
    # 3) Fallback to local relative path
    p = Path('logs')
    p.mkdir(parents=True, exist_ok=True)
    return p

LOG_DIR = _resolve_log_dir()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(LOG_DIR / 'app.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Agent RAG Studio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import our modules
try:
    from rag.config import get_config, update_config
    from rag.loader import DocumentLoader, DocumentStats
    from rag.splitter import get_splitter, analyze_chunks
    from rag.embedder import get_embedding_manager
    from rag.store import get_vector_store
    from rag.retriever import create_retriever, create_reranking_retriever
    from rag.chain import create_rag_chain, create_conversational_rag_chain
except ImportError as e:
    st.error(f"モジュールのインポートエラー: {e}")
    st.stop()

def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.config = get_config()
        st.session_state.messages = []
        st.session_state.documents = []
        st.session_state.chunks = []
        st.session_state.vector_store = None
        st.session_state.retriever = None
        st.session_state.chain = None
        st.session_state.index_ready = False
        st.session_state.stats = {}
        
        # Create necessary directories
        os.makedirs("data", exist_ok=True)
        os.makedirs("logs", exist_ok=True)

def render_sidebar():
    """Render sidebar with configuration options"""
    with st.sidebar:
        st.title("🤖 Agent RAG Studio")
        st.markdown("---")
        
        # API Key Status
        st.subheader("🔑 API設定状況")
        config = st.session_state.config
        key_status = config.validate_keys()
        
        for service, status in key_status.items():
            icon = "✅" if status else "❌"
            st.write(f"{icon} {service.upper()}: {'設定済み' if status else '未設定'}")
        
        if not any(key_status.values()):
            demo_mode = config.__dict__.get('demo_mode', False)
            if demo_mode:
                st.info("🔧 デモモード：ローカル埋め込みモデルを使用（OpenAI APIキー不要）")
            else:
                st.warning("APIキーが設定されていません。.streamlit/secrets.toml または .env ファイルを確認してください。")
        
        st.markdown("---")
        
        # RAG Configuration
        st.subheader("⚙️ RAG設定")
        
        # Model selection
        embedding_models = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "text-embedding-3-small",
            "text-embedding-3-large"
        ]
        
        embedding_model = st.selectbox(
            "埋め込みモデル",
            embedding_models,
            index=embedding_models.index(config.embedding_model) if config.embedding_model in embedding_models else 0
        )
        
        llm_model = "gpt-4o-mini"  # 固定モデル
        
        # Vector store selection
        vector_store_type = st.selectbox(
            "ベクタストア",
            ["chroma", "faiss"],
            index=0 if config.vector_store == "chroma" else 1
        )
        
        # Generation parameters
        st.subheader("📝 生成パラメータ")
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=config.temperature,
            step=0.05,
            help="生成の創造性を制御（0=決定的、1=創造的）"
        )
        
        max_tokens = st.slider(
            "最大トークン数",
            min_value=100,
            max_value=2000,
            value=config.max_tokens,
            step=100,
            help="生成する最大トークン数"
        )
        
        # Retrieval parameters
        st.subheader("🔍 検索パラメータ")
        
        top_k = st.slider(
            "Top-K",
            min_value=1,
            max_value=20,
            value=config.top_k,
            help="検索する文書数"
        )
        
        # Disable reranking toggle if no Cohere key
        use_reranking = st.toggle(
            "Cohere Rerank使用",
            value=config.use_reranking,
            help="Cohere Rerankで検索結果を再順位付け",
            disabled=not bool(config.cohere_api_key)
        )
        
        rerank_top_r = st.slider(
            "Rerank Top-R",
            min_value=1,
            max_value=10,
            value=config.rerank_top_r,
            help="再順位付け後の文書数",
            disabled=not use_reranking
        )
        
        # LangSmith tracing
        st.subheader("🪪 ログ/トレース")
        langsmith_enabled = st.toggle(
            "LangSmith トレースを有効化",
            value=config.__dict__.get('langsmith_enabled', False),
            help="LangSmithで実行トレースを収集（APIキーが必要）"
        )

        # Update configuration
        if st.button("設定を更新"):
            update_config(
                embedding_model=embedding_model,
                default_llm=llm_model,
                vector_store=vector_store_type,
                temperature=temperature,
                max_tokens=max_tokens,
                top_k=top_k,
                rerank_top_r=rerank_top_r,
                use_reranking=use_reranking,
                langsmith_enabled=langsmith_enabled,
            )
            st.session_state.config = get_config()
            st.rerun()
        
        st.markdown("---")
        
        # Index Status
        st.subheader("📊 インデックス状況")
        if st.session_state.index_ready:
            st.success("✅ インデックス準備完了")
            
            if st.session_state.stats:
                stats = st.session_state.stats
                st.write(f"📄 ドキュメント数: {stats.get('total_documents', 0)}")
                st.write(f"📝 チャンク数: {stats.get('total_chunks', 0)}")
                st.write(f"💾 ベクタストア: {stats.get('vector_store_type', 'unknown')}")
        else:
            st.warning("⚠️ インデックス未作成")
            st.write("ドキュメント管理ページでファイルをアップロードしてください")

def main():
    """Main application function"""
    initialize_session_state()
    render_sidebar()
    
    # Main content area
    st.title("🤖 Agent RAG Studio")
    st.markdown("**高精度 RAG + エージェント型 QA アプリケーション**")
    
    # Quick Start Guide
    if not st.session_state.index_ready:
        st.info("**🚀 クイックスタートガイド**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **1️⃣ ドキュメント準備**
            - サイドバーで「ドキュメント管理」ページに移動
            - PDF、Markdown、HTMLファイルをアップロード
            - インデックスを作成
            """)
        
        with col2:
            st.markdown("""
            **2️⃣ チャット開始**
            - 「チャット」ページで質問を入力
            - AIが関連文書を検索して回答
            - 引用付きで正確な情報を提供
            """)
        
        with col3:
            st.markdown("""
            **3️⃣ 評価・改善**
            - 「評価」ページでシステム性能を確認
            - Ragasによる自動評価
            - 回答品質の継続的改善
            """)
    
    else:
        # Quick chat interface for the main page
        st.subheader("💬 クイックチャット")
        st.markdown("ここで簡単な質問ができます。詳細なチャットは「チャット」ページをご利用ください。")
        
        # Chat input
        user_input = st.text_input(
            "質問を入力してください:",
            placeholder="例: LangChainとは何ですか？",
            key="quick_chat_input"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("送信", type="primary", disabled=not user_input):
                if st.session_state.chain:
                    with st.spinner("回答を生成中..."):
                        try:
                            result = st.session_state.chain.invoke({"question": user_input})
                            
                            st.markdown("### 回答")
                            st.markdown(result["answer"])
                            
                            if result.get("sources"):
                                with st.expander("📚 参照ソース"):
                                    for source in result["sources"]:
                                        st.markdown(f"""
                                        **{source.get('title', 'Untitled')}**  
                                        📁 {source.get('file_name', 'Unknown')}  
                                        📄 ページ: {source.get('page', 'N/A')}  
                                        📊 スコア: {source.get('score', 0.0):.3f}  
                                        
                                        {source.get('snippet', '')}
                                        """)
                            
                            st.success(f"⏱️ 応答時間: {result.get('latency', 0):.2f}秒")
                            
                        except Exception as e:
                            st.error(f"エラーが発生しました: {str(e)}")
                            logger.error(f"Quick chat error: {e}")
                            logger.error(traceback.format_exc())
                else:
                    st.error("チェーンが初期化されていません。まずインデックスを作成してください。")
    
    # Status and Statistics
    if st.session_state.index_ready:
        st.markdown("---")
        st.subheader("📈 システム状況")
        
        col1, col2, col3, col4 = st.columns(4)
        
        stats = st.session_state.stats
        
        with col1:
            st.metric(
                "📄 ドキュメント数",
                stats.get('total_documents', 0)
            )
        
        with col2:
            st.metric(
                "📝 チャンク数", 
                stats.get('total_chunks', 0)
            )
        
        with col3:
            st.metric(
                "🤖 モデル",
                st.session_state.config.default_llm
            )
        
        with col4:
            st.metric(
                "💾 ベクタストア",
                st.session_state.config.vector_store.upper()
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Agent RAG Studio v1.0 | 
        <a href='https://github.com/your-repo' target='_blank'>GitHub</a> | 
        <a href='https://docs.your-site.com' target='_blank'>ドキュメント</a>
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"アプリケーションエラー: {e}")
        logger.error(f"Application error: {e}")
        logger.error(traceback.format_exc())
        
        # Show debug information in development
        if os.getenv("STREAMLIT_ENV") == "development":
            st.exception(e)
