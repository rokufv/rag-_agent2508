"""
Setup script for Agent RAG Studio
"""
import os
import subprocess
import sys
from pathlib import Path

def create_directories():
    """Create necessary directories"""
    directories = [
        "data",
        "logs", 
        "eval_results",
        ".streamlit"
    ]
    
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✅ Created directory: {dir_name}")

def setup_config_files():
    """Setup configuration files if they don't exist"""
    # .env file
    if not Path(".env").exists():
        if Path(".env.template").exists():
            print("📝 Creating .env from template...")
            subprocess.run(["cp", ".env.template", ".env"])
            print("⚠️  Please edit .env file and add your API keys")
        else:
            print("❌ .env.template not found")
    
    # Streamlit secrets
    secrets_path = Path(".streamlit/secrets.toml")
    if not secrets_path.exists():
        if Path(".streamlit/secrets.toml.template").exists():
            print("📝 Creating secrets.toml from template...")
            subprocess.run(["cp", ".streamlit/secrets.toml.template", str(secrets_path)])
            print("⚠️  Please edit .streamlit/secrets.toml and add your API keys")
        else:
            print("❌ .streamlit/secrets.toml.template not found")

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        "streamlit",
        "langchain",
        "langchain_openai",
        "langchain_community", 
        "langchain_core",
        "langgraph", 
        "openai",
        "chromadb",
        "cohere",
        "pandas",
        "numpy",
        "sentence_transformers"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True

def create_sample_documents():
    """Create sample documents for testing"""
    sample_dir = Path("data/samples")
    sample_dir.mkdir(exist_ok=True)
    
    # Sample markdown document
    sample_md = """# LangChain と RAG について

## LangChain とは

LangChain は、大規模言語モデル（LLM）を活用したアプリケーションを構築するためのフレームワークです。

### 主な機能

1. **プロンプト管理**: テンプレートとして管理
2. **チェーン**: 複数のステップを組み合わせ
3. **エージェント**: 自律的な判断と行動
4. **メモリ**: 会話履歴の管理

## RAG (Retrieval-Augmented Generation)

RAGは情報検索と生成を組み合わせた手法で、以下の利点があります：

- **最新情報への対応**: 外部データベースから情報を取得
- **事実性の向上**: 参照元を明確にした回答
- **透明性**: ソースを示すことで信頼性を確保

### RAGの構成要素

1. **ドキュメントローダー**: 様々な形式のファイルを読み込み
2. **テキスト分割**: 適切なサイズのチャンクに分割
3. **埋め込み**: ベクトル化して類似度検索可能に
4. **ベクタストア**: 埋め込みを効率的に保存・検索
5. **リトリーバー**: 関連文書を取得
6. **生成**: LLMで回答を生成

## LangGraph によるエージェント

LangGraphは状態を持つエージェントを作成するためのライブラリです。

### エージェントの特徴

- **計画立案**: タスクを分析し実行計画を作成
- **ツール使用**: 外部APIやツールを活用
- **自己評価**: 結果の品質を判断
- **反復改善**: 必要に応じて再実行

これらの技術を組み合わせることで、高度なAIアシスタントを構築できます。
"""
    
    with open(sample_dir / "langchain_rag_guide.md", "w", encoding="utf-8") as f:
        f.write(sample_md)
    
    # Sample text document
    sample_txt = """エージェントRAGスタジオへようこそ

このアプリケーションは、以下の機能を提供します：

1. ドキュメント管理
   - ファイルアップロード（PDF, Markdown, HTML, Word）
   - テキスト分割とチャンク化
   - ベクタインデックス作成

2. インテリジェントチャット
   - ハイブリッド検索（BM25 + ベクタ検索）
   - Cohere Rerankによる精度向上
   - 引用付き回答生成

3. LangGraphエージェント
   - 自律的な問題解決
   - ツール使用（Web検索、計算、データ分析）
   - 反復改善による高精度回答

4. 評価・分析
   - Ragasによる自動評価
   - 性能メトリクス分析
   - 継続的改善サポート

使い方：
1. まず「ドキュメント管理」でファイルをアップロード
2. インデックスを作成
3. 「チャット」で質問開始
4. 「評価」で性能確認

サポートが必要でしたら、GitHubのIssuesまでお気軽にどうぞ。
"""
    
    with open(sample_dir / "user_guide.txt", "w", encoding="utf-8") as f:
        f.write(sample_txt)
    
    print(f"✅ Created sample documents in {sample_dir}")

def main():
    """Main setup function"""
    print("🚀 Agent RAG Studio セットアップ開始\n")
    
    # Create directories
    print("📁 ディレクトリ作成中...")
    create_directories()
    print()
    
    # Setup config files
    print("⚙️ 設定ファイル準備中...")
    setup_config_files()
    print()
    
    # Check dependencies
    print("📦 依存関係チェック中...")
    deps_ok = check_dependencies()
    print()
    
    # Create sample documents
    print("📄 サンプルドキュメント作成中...")
    create_sample_documents()
    print()
    
    # Final instructions
    print("✅ セットアップ完了！\n")
    
    if not deps_ok:
        print("⚠️  まず依存関係をインストールしてください：")
        print("   pip install -r requirements.txt\n")
    
    print("📝 次のステップ：")
    print("1. .env または .streamlit/secrets.toml にAPIキーを設定")
    print("2. streamlit run app.py でアプリを起動")
    print("3. ブラウザで http://localhost:8501 にアクセス")
    print("4. 「ドキュメント管理」でサンプルファイルをアップロード")
    print("\n🎉 Agent RAG Studio をお楽しみください！")

if __name__ == "__main__":
    main()
