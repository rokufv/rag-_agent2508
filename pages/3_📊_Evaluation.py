"""
Evaluation Page for Agent RAG Studio
"""
import streamlit as st
import logging
import pandas as pd
from datetime import datetime
import json
import traceback
from typing import Dict, Any, List
import os

# Configure logging
logger = logging.getLogger(__name__)

# Import our modules
try:
    from rag.config import get_config
except ImportError as e:
    st.error(f"モジュールのインポートエラー: {e}")
    st.stop()

def initialize_session_state():
    """Initialize session state for evaluation"""
    if 'eval_initialized' not in st.session_state:
        st.session_state.eval_initialized = True
        st.session_state.eval_datasets = []
        st.session_state.eval_results = []
        st.session_state.eval_history = []


def extract_chat_history_for_evaluation():
    """Extract questions from chat history for evaluation"""
    if not hasattr(st.session_state, 'messages') or not st.session_state.messages:
        return []
    
    evaluation_data = []
    
    # Extract user questions and assistant answers
    for i in range(len(st.session_state.messages) - 1):
        current_msg = st.session_state.messages[i]
        next_msg = st.session_state.messages[i + 1]
        
        # Check if current message is user question and next is assistant answer
        if (current_msg.get('role') == 'user' and 
            next_msg.get('role') == 'assistant'):
            
            question = current_msg.get('content', '').strip()
            answer = next_msg.get('content', '').strip()
            
            if question and answer:
                evaluation_data.append({
                    "question": question,
                    "expected_answer": answer,  # Use actual answer as reference
                    "category": "チャット履歴",
                    "timestamp": current_msg.get('timestamp', ''),
                    "sources": next_msg.get('sources', []),
                    "metadata": next_msg.get('metadata', {})
                })
    
    return evaluation_data



def create_sample_dataset():
    """Create sample evaluation dataset"""
    sample_questions = [
        {
            "question": "LangChainとは何ですか？",
            "expected_answer": "LangChainは大規模言語モデル（LLM）を活用したアプリケーション開発のためのフレームワークです。",
            "category": "基本概念"
        },
        {
            "question": "RAGシステムの利点を教えてください",
            "expected_answer": "RAGは最新情報への対応、情報の正確性向上、透明性の確保などの利点があります。",
            "category": "RAG"
        },
        {
            "question": "ベクタデータベースはなぜ必要ですか？",
            "expected_answer": "セマンティック検索を可能にし、類似する文書を効率的に見つけることができるためです。",
            "category": "技術"
        },
        {
            "question": "プロンプトエンジニアリングのコツは？",
            "expected_answer": "明確で具体的な指示、例の提供、段階的な説明などが重要です。",
            "category": "プロンプト"
        },
        {
            "question": "エージェントシステムの特徴は？",
            "expected_answer": "自律的な意思決定、ツール使用、複数ステップでの問題解決が可能です。",
            "category": "エージェント"
        }
    ]
    
    return sample_questions

def run_basic_evaluation(selected_dataset=None):
    """Run basic evaluation without Ragas"""
    if not st.session_state.get('chain'):
        st.error("RAGチェーンが初期化されていません")
        return None
    
    # Get test questions based on selected dataset
    if selected_dataset and selected_dataset.startswith("チャット履歴"):
        chat_history = extract_chat_history_for_evaluation()
        dataset = chat_history
        st.info(f"チャット履歴を使用 ({len(dataset)} 質問)")
    else:
        dataset = create_sample_dataset()
        st.info("サンプルデータセットを使用")
    
    if not dataset:
        st.warning("評価用のデータセットが見つかりません")
        return None
    
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, item in enumerate(dataset):
        status_text.text(f"評価中: {item['question'][:50]}...")
        
        try:
            # Get answer from RAG system
            start_time = datetime.now()
            result = st.session_state.chain.invoke({"question": item["question"]})
            end_time = datetime.now()
            
            response_time = (end_time - start_time).total_seconds()
            
            # Simple evaluation metrics
            answer = result.get("answer", "")
            sources = result.get("sources", [])
            
            # Basic metrics
            has_sources = len(sources) > 0
            answer_length = len(answer)
            source_count = len(sources)
            
            # Simple relevance check (keyword overlap)
            question_words = set(item["question"].lower().split())
            answer_words = set(answer.lower().split())
            keyword_overlap = len(question_words.intersection(answer_words)) / len(question_words) if question_words else 0
            
            # For chat history evaluation, compare with previous answer
            consistency_score = 0.0
            if selected_dataset and selected_dataset.startswith("チャット履歴"):
                # Calculate consistency with previous answer
                expected_words = set(item["expected_answer"].lower().split())
                consistency_score = len(expected_words.intersection(answer_words)) / len(expected_words) if expected_words else 0
            
            eval_result = {
                "question": item["question"],
                "expected_answer": item["expected_answer"],
                "actual_answer": answer,
                "category": item["category"],
                "response_time": response_time,
                "has_sources": has_sources,
                "source_count": source_count,
                "answer_length": answer_length,
                "keyword_overlap": keyword_overlap,
                "consistency_score": consistency_score,  # New metric for chat history
                "sources": sources,
                "timestamp": datetime.now().isoformat(),
                "evaluation_type": "chat_history" if selected_dataset and selected_dataset.startswith("チャット履歴") else "standard"
            }
            
            results.append(eval_result)
            
        except Exception as e:
            logger.error(f"Evaluation error for question: {item['question']}: {e}")
            continue
        
        progress_bar.progress((i + 1) / len(dataset))
    
    progress_bar.empty()
    status_text.empty()
    
    return results

def run_ragas_evaluation(selected_dataset=None):
    """Run evaluation using Ragas (if available)"""
    try:
        # Try to import Ragas
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        )
        from datasets import Dataset
        
        if not st.session_state.get('chain'):
            st.error("RAGチェーンが初期化されていません")
            return None
        
        st.info("Ragas評価を実行中...")
        
        # Get test questions based on selected dataset
        if selected_dataset and selected_dataset.startswith("チャット履歴"):
            chat_history = extract_chat_history_for_evaluation()
            dataset = chat_history
            st.info(f"チャット履歴を使用 ({len(dataset)} 質問)")
        else:
            dataset = create_sample_dataset()
            st.info("サンプルデータセットを使用")
        
        if not dataset:
            st.warning("評価用のデータセットが見つかりません")
            return None
        
        # Prepare data for Ragas
        questions = []
        answers = []
        contexts = []
        ground_truths = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, item in enumerate(dataset):
            status_text.text(f"データ準備中: {item['question'][:50]}...")
            
            try:
                # Get answer and context from RAG system
                result = st.session_state.chain.invoke({"question": item["question"]})
                
                questions.append(item["question"])
                answers.append(result.get("answer", ""))
                
                # Get contexts from sources
                sources = result.get("sources", [])
                context_list = [source.get("snippet", "") for source in sources if source.get("snippet")]
                contexts.append(context_list)
                
                ground_truths.append(item["expected_answer"])
                
            except Exception as e:
                logger.error(f"Error preparing data for question: {item['question']}: {e}")
                continue
            
            progress_bar.progress((i + 1) / len(dataset))
        
        # Create Ragas dataset
        ragas_dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        })
        
        status_text.text("Ragas評価を実行中...")
        
        # Run evaluation
        ragas_result = evaluate(
            ragas_dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            ]
        )
        
        progress_bar.empty()
        status_text.empty()
        
        return ragas_result
        
    except ImportError:
        st.warning("Ragasがインストールされていないため、基本評価のみ実行します")
        return None
    except Exception as e:
        st.error(f"Ragas評価エラー: {str(e)}")
        logger.error(f"Ragas evaluation error: {e}")
        return None

def render_evaluation_section():
    """Render evaluation configuration and execution"""
    st.subheader("🧪 評価実行")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 評価タイプ")
        eval_type = st.selectbox(
            "評価方法を選択",
            ["基本評価", "Ragas評価", "両方"],
            help="基本評価は簡単な指標、Ragas評価は高度な指標を使用"
        )
        
        st.markdown("### データセット選択")
        chat_history = extract_chat_history_for_evaluation()
        
        dataset_options = ["サンプルデータセット"]
        if chat_history:
            dataset_options.insert(0, f"チャット履歴 ({len(chat_history)} 質問)")
        
        selected_dataset = st.selectbox(
            "評価に使用するデータセット",
            dataset_options,
            help="チャット履歴またはサンプルデータセットから選択"
        )
        
        # Display dataset preview
        if selected_dataset.startswith("チャット履歴") and chat_history:
            with st.expander("チャット履歴プレビュー"):
                preview_df = pd.DataFrame(chat_history[:3])  # Show first 3 items
                st.dataframe(preview_df[['question', 'timestamp']], use_container_width=True)

    
    with col2:
        st.markdown("### 評価設定")
        
        include_sources = st.checkbox("ソース情報を含める", value=True)
        save_results = st.checkbox("結果を保存", value=True)
        show_details = st.checkbox("詳細結果を表示", value=True)
    
    # Run evaluation button
    if st.button("🚀 評価を実行", type="primary"):
        if not st.session_state.get('index_ready', False):
            st.error("インデックスが作成されていません")
            return
        
        with st.spinner("評価を実行中..."):
            results = []
            
            # Run basic evaluation
            if eval_type in ["基本評価", "両方"]:
                basic_results = run_basic_evaluation(selected_dataset)
                if basic_results:
                    results.extend(basic_results)
            
            # Run Ragas evaluation
            if eval_type in ["Ragas評価", "両方"]:
                ragas_results = run_ragas_evaluation(selected_dataset)
                if ragas_results:
                    st.session_state.ragas_results = ragas_results
            
            if results:
                st.session_state.eval_results = results
                st.session_state.eval_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "eval_type": eval_type,
                    "num_questions": len(results),
                    "results": results
                })
                
                if save_results:
                    # Save to file
                    os.makedirs("eval_results", exist_ok=True)
                    filename = f"eval_results/evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
                    st.success(f"結果を {filename} に保存しました")
                
                st.success("✅ 評価が完了しました！")

def render_results_section():
    """Render evaluation results"""
    if not st.session_state.eval_results:
        st.info("まず評価を実行してください")
        return
    
    st.subheader("📈 評価結果")
    
    results = st.session_state.eval_results
    
    # Summary metrics
    is_chat_history_eval = any(r.get("evaluation_type") == "chat_history" for r in results)
    
    if is_chat_history_eval:
        col1, col2, col3, col4, col5 = st.columns(5)
    else:
        col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_response_time = sum(r["response_time"] for r in results) / len(results)
        st.metric("平均応答時間", f"{avg_response_time:.2f}s")
    
    with col2:
        source_coverage = sum(1 for r in results if r["has_sources"]) / len(results)
        st.metric("ソース付与率", f"{source_coverage:.1%}")
    
    with col3:
        avg_sources = sum(r["source_count"] for r in results) / len(results)
        st.metric("平均ソース数", f"{avg_sources:.1f}")
    
    with col4:
        avg_overlap = sum(r["keyword_overlap"] for r in results) / len(results)
        st.metric("キーワード一致率", f"{avg_overlap:.1%}")
    
    if is_chat_history_eval:
        with col5:
            avg_consistency = sum(r.get("consistency_score", 0) for r in results) / len(results)
            st.metric("一貫性スコア", f"{avg_consistency:.1%}")
    
    # Detailed results table
    st.markdown("### 詳細結果")
    
    df_data = []
    for r in results:
        row_data = {
            "質問": r["question"][:50] + "..." if len(r["question"]) > 50 else r["question"],
            "カテゴリ": r["category"],
            "応答時間(s)": f"{r['response_time']:.2f}",
            "ソース数": r["source_count"],
            "回答長": r["answer_length"],
            "キーワード一致": f"{r['keyword_overlap']:.1%}",
        }
        
        # Add consistency score for chat history evaluation
        if is_chat_history_eval and "consistency_score" in r:
            row_data["一貫性"] = f"{r['consistency_score']:.1%}"
        
        df_data.append(row_data)
    
    df = pd.DataFrame(df_data)
    st.dataframe(df, use_container_width=True)
    
    # Category analysis
    st.markdown("### カテゴリ別分析")
    
    category_stats = {}
    for r in results:
        cat = r["category"]
        if cat not in category_stats:
            category_stats[cat] = {
                "count": 0,
                "avg_time": 0,
                "avg_sources": 0,
                "avg_overlap": 0
            }
        
        stats = category_stats[cat]
        stats["count"] += 1
        stats["avg_time"] += r["response_time"]
        stats["avg_sources"] += r["source_count"]
        stats["avg_overlap"] += r["keyword_overlap"]
    
    # Calculate averages
    for cat, stats in category_stats.items():
        count = stats["count"]
        stats["avg_time"] /= count
        stats["avg_sources"] /= count
        stats["avg_overlap"] /= count
    
    # Display category stats
    for cat, stats in category_stats.items():
        with st.expander(f"📊 {cat} ({stats['count']} 件)"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("平均応答時間", f"{stats['avg_time']:.2f}s")
            with col2:
                st.metric("平均ソース数", f"{stats['avg_sources']:.1f}")
            with col3:
                st.metric("平均一致率", f"{stats['avg_overlap']:.1%}")

def render_ragas_results():
    """Render Ragas evaluation results"""
    if not st.session_state.get('ragas_results'):
        return
    
    st.subheader("🎯 Ragas評価結果")
    
    ragas_results = st.session_state.ragas_results
    
    # Main metrics
    # --- ここからRagasスコア表示 ---
    faithfulness_score = getattr(ragas_results, "faithfulness", 0)
    relevancy_score = getattr(ragas_results, "answer_relevancy", 0)
    precision_score = getattr(ragas_results, "context_precision", 0)
    recall_score = getattr(ragas_results, "context_recall", 0)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Faithfulness", f"{faithfulness_score:.3f}")
        st.caption("回答の事実性")
    with col2:
        st.metric("Relevancy", f"{relevancy_score:.3f}")
        st.caption("回答の関連性")
    with col3:
        st.metric("Precision", f"{precision_score:.3f}")
        st.caption("根拠の精度")
    with col4:
        st.metric("Recall", f"{recall_score:.3f}")
        st.caption("根拠の網羅性")
    # --- ここまでRagasスコア表示 ---
    
    # Ragas recommendations
    st.markdown("### 📋 改善提案")
    
    if faithfulness_score < 0.7:
        st.warning("**Faithfulness** が低めです。プロンプトで事実のみに基づく回答を強調してください。")
    
    if relevancy_score < 0.7:
        st.warning("**Answer Relevancy** が低めです。質問により直接的に回答するようプロンプトを調整してください。")
    
    if precision_score < 0.7:
        st.warning("**Context Precision** が低めです。より関連性の高い文書を取得するよう検索パラメータを調整してください。")
    
    if recall_score < 0.7:
        st.warning("**Context Recall** が低めです。チャンクサイズや検索対象数を増やすことを検討してください。")

def render_history_section():
    """Render evaluation history"""
    if not st.session_state.eval_history:
        st.info("評価履歴がありません")
        return
    
    st.subheader("📚 評価履歴")
    
    for i, eval_run in enumerate(reversed(st.session_state.eval_history)):
        with st.expander(f"評価 {len(st.session_state.eval_history) - i}: {eval_run['timestamp'][:19]}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**評価タイプ:** {eval_run['eval_type']}")
            with col2:
                st.write(f"**質問数:** {eval_run['num_questions']}")
            with col3:
                # Calculate average metrics for this run
                results = eval_run['results']
                if results:
                    avg_time = sum(r['response_time'] for r in results) / len(results)
                    st.write(f"**平均応答時間:** {avg_time:.2f}s")
            
            # Download button for this run
            json_str = json.dumps(eval_run, ensure_ascii=False, indent=2)
            st.download_button(
                label="📄 ダウンロード",
                data=json_str,
                file_name=f"evaluation_{eval_run['timestamp'][:10]}.json",
                mime="application/json",
                key=f"download_{i}"
            )

def main():
    """Main function for the Evaluation page"""
    st.set_page_config(
        page_title="Evaluation - Agent RAG Studio",
        page_icon="📊",
        layout="wide"
    )
    
    initialize_session_state()
    
    st.title("📊 評価・分析")
    st.markdown("RAGシステムの性能を評価し、改善点を特定します")
    
    # Main sections
    render_evaluation_section()
    st.markdown("---")
    render_results_section()
    st.markdown("---")
    render_ragas_results()
    st.markdown("---")
    render_history_section()

if __name__ == "__main__":
    main()
