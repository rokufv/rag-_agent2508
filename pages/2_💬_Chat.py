"""
Chat Interface Page for Agent RAG Studio
"""
import streamlit as st
import logging
from datetime import datetime
import json
import traceback
from typing import Dict, Any, List

# Configure logging
logger = logging.getLogger(__name__)

# Import our modules
try:
    from rag.config import get_config
    from rag.chain import create_conversational_rag_chain
except ImportError as e:
    st.error(f"モジュールのインポートエラー: {e}")
    st.stop()

def initialize_session_state():
    """Initialize session state for chat"""
    if 'chat_initialized' not in st.session_state:
        st.session_state.chat_initialized = True
        st.session_state.messages = []
        st.session_state.conversation_chain = None
        # 出力形式の既定
        if 'output_format' not in st.session_state:
            st.session_state.output_format = 'markdown'
        st.session_state.chat_stats = {
            'total_queries': 0,
            'total_tokens': 0,
            'average_response_time': 0.0,
            'session_start': datetime.now().isoformat()
        }

def create_conversation_chain():
    """Create or recreate the conversation chain"""
    if st.session_state.get('chain') and st.session_state.get('retriever'):
        try:
            conv_chain = create_conversational_rag_chain(
                retriever=st.session_state.retriever,
                llm_model=st.session_state.config.default_llm,
                temperature=st.session_state.config.temperature,
                max_tokens=st.session_state.config.max_tokens,
                use_reranking=st.session_state.config.use_reranking,
                rerank_top_k=st.session_state.config.rerank_top_r,
                output_format=st.session_state.get('output_format', 'markdown')
            )
            st.session_state.conversation_chain = conv_chain
            return True
        except Exception as e:
            logger.error(f"Error creating conversation chain: {e}")
            return False
    return False

def render_chat_interface():
    """Render the main chat interface"""
    if not st.session_state.get('index_ready', False):
        st.warning("⚠️ インデックスが作成されていません")
        st.info("「ドキュメント管理」ページでドキュメントをアップロードし、インデックスを作成してください。")
        return
    
    # Create conversation chain if not exists
    if not st.session_state.conversation_chain:
        if not create_conversation_chain():
            st.error("会話チェーンの初期化に失敗しました")
            return
    
    # Chat configuration sidebar
    with st.sidebar:
        st.subheader("💬 チャット設定")
        
        # 出力形式（チェーンに反映）
        prev_format = st.session_state.get('output_format', 'markdown')
        output_format = st.selectbox(
            "回答形式",
            ["markdown", "json"],
            index=(0 if prev_format == 'markdown' else 1),
            help="回答の出力形式を選択"
        )
        if output_format != prev_format:
            st.session_state.output_format = output_format
            # 形式変更時は会話チェーンを作り直す
            st.session_state.conversation_chain = None
            create_conversation_chain()
        
        # Conversation mode
        conversation_mode = st.toggle(
            "会話履歴を考慮",
            value=True,
            help="前の質問・回答を文脈として考慮"
        )
        
        # Advanced settings
        with st.expander("⚙️ 詳細設定"):
            show_sources = st.toggle("ソース表示", value=True)
            show_confidence = st.toggle("信頼度表示", value=True)
            show_metadata = st.toggle("メタデータ表示", value=False)
            auto_scroll = st.toggle("自動スクロール", value=True)
        
        # Session statistics
        st.subheader("📊 セッション統計")
        stats = st.session_state.chat_stats
        st.metric("質問数", stats['total_queries'])
        st.metric("平均応答時間", f"{stats['average_response_time']:.2f}s")
        
        # Clear chat button
        if st.button("🗑️ チャット履歴をクリア"):
            st.session_state.messages = []
            if st.session_state.conversation_chain:
                st.session_state.conversation_chain.clear_history()
            st.session_state.chat_stats['total_queries'] = 0
            st.rerun()
    
    # Display chat messages
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.write(message["content"])
                else:
                    # Display assistant response
                    st.markdown(message["content"])
                    
                    # Display sources if available
                    if show_sources and message.get("sources"):
                        with st.expander("📚 参照ソース"):
                            for i, source in enumerate(message["sources"], 1):
                                col1, col2 = st.columns([3, 1])
                                
                                with col1:
                                    st.markdown(f"""
                                    **[{i}] {source.get('title', 'Untitled')}**  
                                    📁 {source.get('file_name', 'Unknown')}  
                                    📄 ページ: {source.get('page', 'N/A')}
                                    """)
                                    
                                    if source.get('snippet'):
                                        st.text(source['snippet'])
                                
                                with col2:
                                    if show_confidence:
                                        score = source.get('score', 0.0)
                                        st.metric("スコア", f"{score:.3f}")
                                    
                                    if show_metadata and source.get('metadata'):
                                        with st.expander("詳細"):
                                            st.json(source['metadata'], expanded=False)
                    
                    # Display metadata if enabled
                    if show_metadata and message.get("metadata"):
                        with st.expander("🔍 メタデータ"):
                            st.json(message["metadata"], expanded=False)

def process_user_input(user_input: str):
    """Process user input and generate response"""
    if not user_input.strip():
        return
    
    # Add user message to chat
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "timestamp": datetime.now().isoformat()
    })
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("回答を生成中..."):
            try:
                # Use conversation chain if available
                if st.session_state.conversation_chain:
                    result = st.session_state.conversation_chain.invoke({"question": user_input})
                else:
                    result = st.session_state.chain.invoke({"question": user_input})

                if not result or not isinstance(result, dict) or "answer" not in result:
                    st.error("AIから有効な回答が返りませんでした。システム管理者にご連絡ください。")
                    return

                # Display response
                st.markdown(result["answer"])
                
                # Display sources
                if result.get("sources"):
                    with st.expander("📚 参照ソース"):
                        for i, source in enumerate(result["sources"], 1):
                            st.markdown(f"""
                            **[{i}] {source.get('title', 'Untitled')}**  
                            📁 {source.get('file_name', 'Unknown')}  
                            📄 ページ: {source.get('page', 'N/A')}  
                            📊 スコア: {source.get('score', 0.0):.3f}
                            
                            {source.get('snippet', '')}
                            """)
                
                # Display response metadata
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.success(f"⏱️ {result.get('latency', 0):.2f}秒")
                with col2:
                    st.info(f"🤖 {st.session_state.config.default_llm}")
                with col3:
                    if result.get('confidence'):
                        st.metric("信頼度", f"{result['confidence']:.2f}")
                
                # Add assistant message to chat
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result.get("sources", []),
                    "metadata": {
                        "latency": result.get("latency", 0),
                        "model": st.session_state.config.default_llm,
                        "confidence": result.get("confidence"),
                        "timestamp": result.get("timestamp"),
                    },
                    "timestamp": datetime.now().isoformat()
                })
                
                # Update statistics
                stats = st.session_state.chat_stats
                stats['total_queries'] += 1
                
                # Update average response time
                if stats['total_queries'] == 1:
                    stats['average_response_time'] = result.get('latency', 0)
                else:
                    current_avg = stats['average_response_time']
                    new_time = result.get('latency', 0)
                    stats['average_response_time'] = (current_avg * (stats['total_queries'] - 1) + new_time) / stats['total_queries']
                
            except Exception as e:
                error_message = f"申し訳ありません。エラーが発生しました: {str(e)}"
                st.error(error_message)
                
                # Add error message to chat
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message,
                    "sources": [],
                    "metadata": {"error": str(e)},
                    "timestamp": datetime.now().isoformat()
                })
                
                logger.error(f"Chat error: {e}")
                logger.error(traceback.format_exc())

def render_suggested_questions():
    """Render suggested questions section"""
    st.subheader("💡 質問例")
    
    suggested_questions = [
        "LangChainとは何ですか？",
        "RAGシステムの仕組みを教えてください",
        "ベクタデータベースの利点は？",
        "エージェントの機能について説明してください",
        "LangGraphの特徴は？",
        "プロンプトエンジニアリングのベストプラクティスは？"
    ]
    
    cols = st.columns(2)
    for i, question in enumerate(suggested_questions):
        col = cols[i % 2]
        with col:
            if st.button(question, key=f"suggested_{i}"):
                # Set the question in the input (this is a workaround since we can't directly modify chat input)
                st.session_state.suggested_question = question
                st.rerun()

def render_export_section():
    """Render chat export section"""
    if not st.session_state.messages:
        return
    
    with st.expander("📤 チャット履歴エクスポート"):
        # Export as JSON
        chat_data = {
            "session_info": st.session_state.chat_stats,
            "messages": st.session_state.messages,
            "exported_at": datetime.now().isoformat()
        }
        
        json_str = json.dumps(chat_data, ensure_ascii=False, indent=2)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📄 JSONでダウンロード",
                data=json_str,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col2:
            # Export as markdown
            markdown_content = "# チャット履歴\n\n"
            for message in st.session_state.messages:
                role = "🧑 ユーザー" if message["role"] == "user" else "🤖 アシスタント"
                markdown_content += f"## {role}\n\n{message['content']}\n\n"
                
                if message.get("sources"):
                    markdown_content += "### 参照ソース\n\n"
                    for i, source in enumerate(message["sources"], 1):
                        markdown_content += f"{i}. {source.get('title', 'Untitled')} ({source.get('file_name', 'Unknown')})\n"
                    markdown_content += "\n"
            
            st.download_button(
                label="📝 Markdownでダウンロード",
                data=markdown_content,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )

def main():
    """Main function for the Chat page"""
    st.set_page_config(
        page_title="Chat - Agent RAG Studio",
        page_icon="💬",
        layout="wide"
    )
    
    initialize_session_state()
    
    st.title("💬 チャット")
    st.markdown("AIアシスタントと対話して、ドキュメントの内容について質問できます")
    
    # Check if system is ready
    if not st.session_state.get('index_ready', False):
        st.warning("⚠️ システムが準備できていません")
        st.info("「ドキュメント管理」ページでドキュメントをアップロードし、インデックスを作成してください。")
        
        # Show suggested questions anyway
        render_suggested_questions()
        return
    
    # Main chat interface
    render_chat_interface()
    
    # Chat input
    user_input = st.chat_input("質問を入力してください...")
    
    # Handle suggested question
    if st.session_state.get('suggested_question'):
        user_input = st.session_state.suggested_question
        del st.session_state.suggested_question
    
    # Process user input
    if user_input:
        process_user_input(user_input)
        st.rerun()
    
    # Show suggested questions if no messages yet
    if not st.session_state.messages:
        render_suggested_questions()
    else:
        # Show export options
        render_export_section()

if __name__ == "__main__":
    main()
