"""
Document Management Page for Agent RAG Studio
"""
import streamlit as st
import os
import logging
from pathlib import Path
import traceback
from typing import List
import time

# Configure logging
logger = logging.getLogger(__name__)

# Import our modules
try:
    from rag.config import get_config
    from rag.loader import DocumentLoader, DocumentStats
    from rag.splitter import get_splitter, analyze_chunks
    from rag.embedder import get_embedding_manager
    from rag.store import get_vector_store
    from rag.retriever import create_retriever
    from rag.chain import create_rag_chain
except ImportError as e:
    st.error(f"モジュールのインポートエラー: {e}")
    st.stop()

def initialize_session_state():
    """Initialize session state if not already done"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.config = get_config()
        st.session_state.documents = []
        st.session_state.chunks = []
        st.session_state.vector_store = None
        st.session_state.retriever = None
        st.session_state.chain = None
        st.session_state.index_ready = False
        st.session_state.stats = {}

def render_upload_section():
    """Render file upload section"""
    st.subheader("📁 ファイルアップロード")
    
    # Supported file types
    supported_types = DocumentLoader.get_supported_extensions()
    st.info(f"**サポートファイル形式:** {', '.join(supported_types)}")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "ドキュメントを選択してください",
        type=[ext.replace('.', '') for ext in supported_types],
        accept_multiple_files=True,
        help="複数ファイルを同時にアップロードできます"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} ファイルが選択されました")
        
        # Show file details
        with st.expander("📋 ファイル詳細"):
            for file in uploaded_files:
                file_size = len(file.getbuffer()) / 1024 / 1024  # MB
                st.write(f"📄 **{file.name}** ({file_size:.2f} MB)")
        
        # Load documents button
        if st.button("📖 ドキュメントを読み込み", type="primary"):
            with st.spinner("ドキュメントを読み込み中..."):
                try:
                    # Create data directory
                    data_dir = Path("data")
                    data_dir.mkdir(exist_ok=True)
                    
                    # Load documents
                    all_documents = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, uploaded_file in enumerate(uploaded_files):
                        status_text.text(f"読み込み中: {uploaded_file.name}")
                        
                        # Load document
                        docs = DocumentLoader.load_uploaded_file(
                            uploaded_file, 
                            save_dir=str(data_dir)
                        )
                        all_documents.extend(docs)
                        
                        # Update progress
                        progress = (i + 1) / len(uploaded_files)
                        progress_bar.progress(progress)
                    
                    # Store in session state
                    st.session_state.documents = all_documents
                    
                    # Calculate statistics
                    stats = DocumentStats.analyze_documents(all_documents)
                    st.session_state.stats.update(stats)
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.success(f"✅ {len(all_documents)} ドキュメントを読み込みました")
                    
                    # Display statistics
                    st.markdown(DocumentStats.format_stats_for_display(stats))
                    
                except Exception as e:
                    st.error(f"ドキュメント読み込みエラー: {str(e)}")
                    logger.error(f"Document loading error: {e}")
                    logger.error(traceback.format_exc())

def render_chunking_section():
    """Render document chunking configuration"""
    if not st.session_state.documents:
        st.info("まずドキュメントをアップロードしてください")
        return
    
    st.subheader("✂️ テキスト分割設定")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Splitting strategy
        strategy = st.selectbox(
            "分割戦略",
            ["recursive", "token", "markdown", "html"],
            help="テキスト分割の方法を選択"
        )
        
        # Chunk size
        chunk_size = st.slider(
            "チャンクサイズ",
            min_value=200,
            max_value=2000,
            value=1000,
            step=100,
            help="各チャンクの最大文字数"
        )
    
    with col2:
        # Chunk overlap
        chunk_overlap = st.slider(
            "チャンクオーバーラップ",
            min_value=0,
            max_value=500,
            value=100,
            step=50,
            help="隣接チャンク間の重複文字数"
        )
    
    # Split documents button
    if st.button("✂️ ドキュメントを分割", type="primary"):
        with st.spinner("ドキュメントを分割中..."):
            try:
                # Create splitter
                splitter = get_splitter(
                    strategy=strategy,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                
                # Split documents
                chunks = splitter.split_documents(st.session_state.documents)
                st.session_state.chunks = chunks
                
                # Analyze chunks
                chunk_stats = analyze_chunks(chunks)
                st.session_state.stats.update(chunk_stats)
                
                st.success(f"✅ {len(chunks)} チャンクを作成しました")
                
                # Display chunk statistics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("総チャンク数", chunk_stats['total_chunks'])
                
                with col2:
                    st.metric("平均チャンクサイズ", f"{chunk_stats.get('average_chunk_size', 0):.0f}")
                
                with col3:
                    st.metric("最小/最大サイズ", f"{chunk_stats.get('min_chunk_size', 0)}/{chunk_stats.get('max_chunk_size', 0)}")
                
                with col4:
                    st.metric("推定トークン数", f"{chunk_stats.get('total_tokens_estimate', 0):,}")
                
                # Show sample chunks
                with st.expander("📝 チャンクサンプル"):
                    for i, chunk in enumerate(chunks[:3]):
                        st.markdown(f"**チャンク {i+1}:**")
                        st.text(chunk.page_content[:200] + "..." if len(chunk.page_content) > 200 else chunk.page_content)
                        st.json(chunk.metadata, expanded=False)
                        st.markdown("---")
                
            except Exception as e:
                st.error(f"ドキュメント分割エラー: {str(e)}")
                logger.error(f"Document splitting error: {e}")
                logger.error(traceback.format_exc())

def render_indexing_section():
    """Render vector indexing section"""
    if not st.session_state.chunks:
        st.info("まずドキュメントを分割してください")
        return
    
    st.subheader("🚀 ベクタインデックス作成")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Embedding model selection
        embedding_models = [
            "text-embedding-3-small",
            "text-embedding-3-large",
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2"
        ]
        
        embedding_model = st.selectbox(
            "埋め込みモデル",
            embedding_models,
            index=0
        )
    
    with col2:
        # Vector Store Selection
        vector_store = st.selectbox(
            "🗄️ ベクトルストア",
            ["faiss", "chroma"],  # FAISS first
            index=0 if st.session_state.config.vector_store == "faiss" else 1
        )
    
    # Create index button
    if st.button("🚀 インデックスを作成", type="primary"):
        with st.spinner("ベクタインデックスを作成中..."):
            try:
                # Initialize embedding manager
                embedding_manager = get_embedding_manager(embedding_model)
                
                # Initialize vector store
                vector_store_manager = get_vector_store(
                    store_type=vector_store,
                    embedding_manager=embedding_manager
                )
                
                # Clear existing store if any
                if st.checkbox("既存インデックスをクリア", value=False):
                    vector_store_manager.clear()
                
                # Create progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def progress_callback(progress, message):
                    progress_bar.progress(progress)
                    status_text.text(message)
                
                # Add documents to vector store
                doc_ids = vector_store_manager.add_documents(
                    st.session_state.chunks,
                    batch_size=50,
                    progress_callback=progress_callback
                )
                
                # Create retriever
                retriever = create_retriever(
                    vector_store_manager=vector_store_manager,
                    documents=st.session_state.chunks,
                    retriever_type="hybrid"
                )
                
                # Create RAG chain
                chain = create_rag_chain(
                    retriever=retriever,
                    llm_model=st.session_state.config.default_llm,
                    temperature=st.session_state.config.temperature,
                    max_tokens=st.session_state.config.max_tokens,
                    use_reranking=st.session_state.config.use_reranking,
                    rerank_top_k=st.session_state.config.rerank_top_r,
                    output_format=st.session_state.get('output_format', 'markdown')
                )
                
                # Store in session state
                st.session_state.vector_store = vector_store_manager
                st.session_state.retriever = retriever
                st.session_state.chain = chain
                st.session_state.index_ready = True
                
                # Update stats
                vector_stats = vector_store_manager.get_stats()
                st.session_state.stats.update({
                    'vector_store_type': vector_store,
                    'embedding_model': embedding_model,
                    'vector_store_stats': vector_stats
                })
                
                progress_bar.empty()
                status_text.empty()
                
                st.success(f"✅ インデックスが正常に作成されました！")
                st.balloons()
                
                # Display final statistics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("インデックス済み文書", len(doc_ids))
                
                with col2:
                    st.metric("ベクタストア", vector_store.upper())
                
                with col3:
                    st.metric("埋め込みモデル", embedding_model.split('/')[-1])
                
                st.info("🎉 これで「チャット」ページで質問を開始できます！")
                
            except Exception as e:
                st.error(f"インデックス作成エラー: {str(e)}")
                logger.error(f"Indexing error: {e}")
                logger.error(traceback.format_exc())

def render_status_section():
    """Render current status"""
    st.subheader("📊 現在の状況")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        docs_count = len(st.session_state.documents)
        st.metric("📄 ドキュメント", docs_count, delta=docs_count if docs_count > 0 else None)
    
    with col2:
        chunks_count = len(st.session_state.chunks)
        st.metric("✂️ チャンク", chunks_count, delta=chunks_count if chunks_count > 0 else None)
    
    with col3:
        index_status = "✅ 完了" if st.session_state.index_ready else "❌ 未作成"
        st.metric("🚀 インデックス", index_status)
    
    with col4:
        chain_status = "✅ 準備完了" if st.session_state.chain else "❌ 未準備"
        st.metric("🤖 チェーン", chain_status)
    
    # Detailed statistics
    if st.session_state.stats:
        with st.expander("📈 詳細統計"):
            st.json(st.session_state.stats, expanded=False)

def main():
    """Main function for the Documents page"""
    st.set_page_config(
        page_title="Documents - Agent RAG Studio",
        page_icon="📄",
        layout="wide"
    )
    
    initialize_session_state()
    
    st.title("📄 ドキュメント管理")
    st.markdown("ドキュメントのアップロード、分割、インデックス作成を行います")
    
    # Main workflow
    render_upload_section()
    st.markdown("---")
    render_chunking_section()
    st.markdown("---")
    render_indexing_section()
    st.markdown("---")
    render_status_section()
    
    # Clear all button
    if st.button("🗑️ 全データをクリア", type="secondary"):
        if st.checkbox("本当にクリアしますか？"):
            st.session_state.documents = []
            st.session_state.chunks = []
            st.session_state.vector_store = None
            st.session_state.retriever = None
            st.session_state.chain = None
            st.session_state.index_ready = False
            st.session_state.stats = {}
            st.success("全データがクリアされました")
            st.rerun()

if __name__ == "__main__":
    main()
