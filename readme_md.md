# 東吳大學對話式智能問答系統 🎓

這是一個基於 LangChain 和 Streamlit 的智能問答系統，具備對話記憶功能。

## 功能特色

- 🤖 智能問答對話
- 🧠 對話記憶功能
- 🔍 向量相似度搜索
- 📚 知識庫管理
- 💬 自然語言處理

## 在線體驗

[點擊這裡使用系統](你的-streamlit-cloud-網址)

## 本地運行

1. 安裝依賴：
```bash
pip install -r requirements.txt
```

2. 建立向量資料庫：
```bash
python setup_database.py
```

3. 啟動應用：
```bash
streamlit run app.py
```

## 技術架構

- **前端**: Streamlit
- **LLM**: Ollama (gemma3n)
- **向量資料庫**: ChromaDB
- **嵌入模型**: text2vec-large-chinese
- **框架**: LangChain

## 作者

東吳大學智能問答系統開發團隊