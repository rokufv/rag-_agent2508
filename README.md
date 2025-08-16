# Agent RAG Studio

**LangChain と LangGraph による高精度 RAG・AI エージェント［実践］アプリケーション**

## 📋 概要

Agent RAG Studio は、書籍『LangChain と LangGraph による RAG・AI エージェント［実践］入門』の実装知見を踏まえて構築された、Streamlit で運用可能な高精度 RAG + エージェント型 QA アプリケーションです。

### 🎯 主な特徴

- **高精度 RAG システム**: ハイブリッド検索（BM25 + ベクタ）+ Cohere Rerank
- **LangGraph エージェント**: 自律的な推論・検索・回答生成ループ
- **多形式ドキュメント対応**: PDF、Markdown、HTML、Word など
- **評価機能**: Ragas による自動評価とメトリクス分析
- **直感的 UI**: Streamlit による美しく使いやすいインターフェース
- **監査・ログ機能**: 完全な操作履歴とトレーサビリティ

## 🚀 クイックスタート

### 1. 環境セットアップ

```bash
# リポジトリをクローン
git clone <repository-url>
cd "LangChain と LangGraph による RAG・AI エージェント［実践］入門"

# Python環境作成（推奨: Python 3.10+）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 依存関係インストール
pip install -r requirements.txt
```

### 2. API キー設定

`.env.template` を `.env` にコピーして API キーを設定：

```bash
cp .env.template .env
```

`.env` ファイルを編集：

```env
OPENAI_API_KEY=your_openai_api_key_here
COHERE_API_KEY=your_cohere_api_key_here
LANGCHAIN_API_KEY=your_langsmith_api_key_here
```

または、`.streamlit/secrets.toml.template` を `.streamlit/secrets.toml` にコピーして設定：

```bash
cp .streamlit/secrets.toml.template .streamlit/secrets.toml
```

### 3. アプリケーション起動

```bash
streamlit run app.py
```

ブラウザで `http://localhost:8501` にアクセス

### 4. 基本的な使い方

1. **ドキュメント管理**ページでファイルをアップロード
2. テキスト分割とベクタインデックスを作成
3. **チャット**ページで AI と対話
4. **評価**ページでシステム性能を確認

## 📁 プロジェクト構造

```
project/
├── app.py                      # メインアプリケーション
├── pages/                      # Streamlit ページ
│   ├── 1_📄_Documents.py      # ドキュメント管理
│   ├── 2_💬_Chat.py           # チャットインターフェース
│   └── 3_📊_Evaluation.py     # 評価・分析
├── rag/                        # RAG コアモジュール
│   ├── config.py              # 設定管理
│   ├── loader.py              # ドキュメントローダー
│   ├── splitter.py            # テキスト分割
│   ├── embedder.py            # 埋め込み生成
│   ├── store.py               # ベクタストア
│   ├── retriever.py           # ハイブリッド検索
│   └── chain.py               # LCEL チェーン
├── agent/                      # LangGraph エージェント
│   ├── state.py               # 状態管理
│   ├── nodes.py               # エージェントノード
│   ├── tools.py               # ツール実装
│   └── graph.py               # グラフ定義
├── utils/                      # ユーティリティ
│   └── logging.py             # ログ機能
├── data/                       # ドキュメント保存
├── logs/                       # ログファイル
├── requirements.txt            # 依存関係
└── README.md                   # このファイル
```

## 🛠️ 技術スタック

### コア技術
- **Python 3.10+**
- **Streamlit 1.36+** - UI フレームワーク
- **LangChain 0.2.x** - LLM アプリケーション開発
- **LangGraph 0.2.x** - エージェント実装
- **OpenAI GPT-4o-mini** - 言語モデル

### RAG システム
- **ChromaDB / FAISS** - ベクタストア
- **OpenAI Embeddings** - 埋め込みモデル
- **BM25 + Vector Search** - ハイブリッド検索
- **Cohere Rerank** - 再ランキング

### 評価・監視
- **Ragas** - RAG システム評価
- **LangSmith** - トレーシング・モニタリング
- **構造化ログ** - 分析・監査

## 📊 機能詳細

### RAG システム

#### ドキュメント処理
- **対応形式**: PDF, Markdown, HTML, Word, テキスト
- **分割戦略**: Recursive, Token-based, Semantic, Header-based
- **チャンクサイズ**: 200-2000 文字（調整可能）

#### 検索・生成
- **ハイブリッド検索**: BM25 + ベクタ検索の組み合わせ
- **再ランキング**: Cohere Rerank による精度向上
- **引用付き回答**: ソース情報を必ず含む回答生成

### LangGraph エージェント

#### ワークフロー
1. **Plan**: 質問分析と実行計画作成
2. **Retrieve**: ドキュメント検索
3. **Reason**: 情報分析と信頼度評価
4. **Act**: 外部ツール実行（必要に応じて）
5. **Respond**: 最終回答生成

#### 利用可能ツール
- **Web Search**: リアルタイム検索
- **Calculator**: 数値計算
- **Data Analysis**: データ分析（pandas）
- **DateTime**: 日付・時刻操作

### 評価システム

#### 基本評価
- **応答時間**: P95 レイテンシ測定
- **ソース付与率**: 引用の網羅性
- **キーワード一致率**: 質問との関連性

#### Ragas 評価
- **Faithfulness**: 回答の事実性
- **Answer Relevancy**: 回答の関連性
- **Context Precision**: 文脈の精度
- **Context Recall**: 文脈の再現率

## ⚙️ 設定・カスタマイズ

### 環境変数

| 変数名 | 説明 | デフォルト |
|--------|------|------------|
| `OPENAI_API_KEY` | OpenAI API キー | 必須 |
| `COHERE_API_KEY` | Cohere API キー | オプション |
| `LANGCHAIN_API_KEY` | LangSmith API キー | オプション |
| `RAG_DATA_DIR` | データディレクトリ | `./data` |
| `RAG_VECTOR_STORE` | ベクタストア | `chroma` |
| `RAG_EMBEDDING_MODEL` | 埋め込みモデル | `text-embedding-3-small` |

### パフォーマンス設定

```python
# rag/config.py で調整可能
chunk_size = 1000           # チャンクサイズ
chunk_overlap = 100         # オーバーラップ
top_k = 6                   # 検索文書数
rerank_top_r = 3           # 再ランク数
temperature = 0.2           # 生成温度
max_tokens = 1000          # 最大トークン数
```

## 📈 パフォーマンス指標

### 目標 KPI
- **初回応答時間**: ≤ 3.0s（キャッシュ命中時）
- **新規生成時間**: ≤ 8.0s
- **回答有用性**: ≥ 0.75（Ragas スコア）
- **出典リンク付与率**: 100%

### 実際の性能
- 同時接続 10 ユーザーで P95 応答 ≤ 12s
- ドキュメント 50MB×5 ファイルの索引化対応
- メモリ使用量: < 2GB（標準設定）

## 🔧 開発・運用

### 開発環境

```bash
# 開発モード起動
STREAMLIT_ENV=development streamlit run app.py

# ログレベル調整
export LOG_LEVEL=DEBUG

# デバッグ用 Jupyter
jupyter notebook eval/ragas.ipynb
```

### ログ・監視

- **アプリケーションログ**: `logs/app.log`
- **分析ログ**: `logs/analytics.jsonl`
- **監査ログ**: `logs/audit.log`
- **LangSmith**: トレーシング・比較分析

### Docker 対応

```bash
# Dockerfile 作成（今後対応予定）
docker build -t agent-rag-studio .
docker run -p 8501:8501 agent-rag-studio
```

## 🚨 トラブルシューティング

### よくある問題

#### API キーエラー
```
APIキーが設定されていません
```
**解決**: `.env` または `.streamlit/secrets.toml` でキーを設定

#### メモリ不足
```
ChromaDB initialization failed
```
**解決**: ベクタストアを FAISS に変更、またはチャンクサイズを削減

#### 検索結果なし
```
関連する情報が見つかりませんでした
```
**解決**: インデックス作成を確認、検索パラメータ（top_k）を増加

### ログ確認

```bash
# エラーログ確認
tail -f logs/app.log

# 分析データ確認
cat logs/analytics.jsonl | jq '.'

# 監査ログ確認
grep "ERROR" logs/audit.log
```

## 🔒 セキュリティ

### データ保護
- API キーは環境変数またはSecrets管理
- アップロードファイルはローカル保存
- 重要ログのマスキング
- 操作履歴の完全トレーサビリティ

### アクセス制御
- 簡易パスコード認証（初期実装）
- IPアドレス制限対応（設定可能）
- セッション管理

## 📚 参考資料

- [LangChain Documentation](https://docs.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Ragas Documentation](https://docs.ragas.io/)
- [Cohere Rerank API](https://docs.cohere.com/reference/rerank-1)

## 🤝 コントリビューション

1. フォークしてください
2. フィーチャーブランチを作成（`git checkout -b feature/amazing-feature`）
3. 変更をコミット（`git commit -m 'Add amazing feature'`）
4. ブランチをプッシュ（`git push origin feature/amazing-feature`）
5. プルリクエストを作成

## 📄 ライセンス

このプロジェクトは MIT ライセンスの下で公開されています。詳細は [LICENSE](LICENSE) ファイルを参照してください。

## 📞 サポート

- **Issues**: GitHub Issues でバグ報告・機能要望
- **Discussions**: GitHub Discussions で質問・議論
- **Email**: support@agent-rag-studio.com

---

**Agent RAG Studio** - Powered by LangChain & LangGraph 🚀
