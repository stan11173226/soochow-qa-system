# 修復版本 - test0811_fixed.py

import logging
from typing import List, Dict, Any, Tuple
from langchain_community.llms import Ollama
# 更新的匯入方式
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    print("✅ 使用新版 langchain_huggingface")
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    print("⚠️ 使用舊版 langchain_community，建議升級")

from langchain_chroma import Chroma
from langchain.docstore.document import Document
import streamlit as st
import asyncio
import httpx
import torch
import json
import os
import shutil
from datetime import datetime
from huggingface_hub import snapshot_download

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('qa_system.log')
    ]
)

def download_and_verify_model():
    """下載並驗證嵌入模型"""
    model_name = "GanymedeNil/text2vec-large-chinese"
    local_dir = "./text2vec-large-chinese"
    
    # 檢查模型目錄是否存在且包含必要文件
    required_files = ['pytorch_model.bin', 'config.json', 'tokenizer.json']
    model_complete = False
    
    if os.path.exists(local_dir):
        existing_files = os.listdir(local_dir)
        model_complete = any(
            file in existing_files for file in ['pytorch_model.bin', 'model.safetensors']
        ) and 'config.json' in existing_files
        
        if model_complete:
            logging.info(f"模型已存在且完整: {local_dir}")
            return True
        else:
            logging.warning(f"模型目錄存在但不完整，將重新下載")
            try:
                shutil.rmtree(local_dir)
            except Exception as e:
                logging.error(f"無法刪除不完整的模型目錄: {e}")
                return False
    
    # 下載模型
    try:
        logging.info(f"開始下載模型: {model_name}")
        print(f"📥 正在下載模型 '{model_name}'，請稍候...")
        
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            resume_download=True,  # 支援斷點續傳
            local_dir_use_symlinks=False  # 不使用符號連結
        )
        
        # 驗證下載是否成功
        if os.path.exists(local_dir):
            downloaded_files = os.listdir(local_dir)
            model_file_exists = any(
                file in downloaded_files for file in ['pytorch_model.bin', 'model.safetensors']
            )
            config_exists = 'config.json' in downloaded_files
            
            if model_file_exists and config_exists:
                logging.info(f"✅ 模型下載成功: {local_dir}")
                print(f"✅ 模型下載完成")
                return True
            else:
                logging.error(f"模型下載不完整，缺少必要文件")
                return False
        else:
            logging.error(f"下載後模型目錄不存在")
            return False
            
    except Exception as e:
        logging.error(f"下載模型時發生錯誤: {e}")
        print(f"❌ 模型下載失敗: {e}")
        return False

class ConversationMemory:
    """對話記憶管理類"""
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.conversation_history = []
        
    def add_exchange(self, user_input: str, system_response: str, context_used: str = ""):
        """添加一組對話交換"""
        exchange = {
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'system_response': system_response,
            'context_used': context_used
        }
        
        self.conversation_history.append(exchange)
        
        # 保持歷史記錄在最大限制內
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def get_conversation_context(self, current_query: str) -> str:
        """生成對話上下文字串"""
        if not self.conversation_history:
            return ""
        
        context_parts = ["=== 對話歷史 ==="]
        
        for i, exchange in enumerate(self.conversation_history[-5:], 1):  # 只取最近5輪對話
            context_parts.append(f"第{i}輪對話:")
            context_parts.append(f"使用者問: {exchange['user_input']}")
            context_parts.append(f"系統答: {exchange['system_response'][:200]}...")  # 截取前200字
            context_parts.append("---")
        
        context_parts.append(f"當前問題: {current_query}")
        context_parts.append("=== 對話歷史結束 ===\n")
        
        return "\n".join(context_parts)
    
    def clear_history(self):
        """清除對話歷史"""
        self.conversation_history = []
    
    def get_related_previous_qa(self, current_query: str) -> List[Dict]:
        """根據當前問題找出相關的歷史對話"""
        if not self.conversation_history:
            return []
        
        # 簡單的關鍵字匹配邏輯
        current_keywords = set(current_query.lower().split())
        related_exchanges = []
        
        for exchange in self.conversation_history:
            previous_keywords = set(exchange['user_input'].lower().split())
            # 如果有共同關鍵字，認為是相關的
            if current_keywords & previous_keywords:
                related_exchanges.append(exchange)
        
        return related_exchanges[-3:]  # 返回最近3個相關對話

@st.cache_resource
def get_embeddings() -> HuggingFaceEmbeddings:
    """初始化嵌入模型，包含完整的錯誤處理"""
    logging.info("正在初始化嵌入模型...")
    
    # 首先確保模型已下載
    if not download_and_verify_model():
        st.error("❌ 模型下載失敗，請檢查網路連接或嘗試重新啟動")
        st.stop()
    
    # 檢測硬體
    if torch.cuda.is_available():
        device = 'cuda'
        logging.info("檢測到 NVIDIA GPU，使用 CUDA 加速")
    elif torch.backends.mps.is_available():
        device = 'mps'
        logging.info("檢測到 Apple Silicon GPU，使用 MPS 加速")
    else:
        device = 'cpu'
        logging.info("未檢測到 GPU，使用 CPU")
    
    try:
        # 嘗試載入模型
        embeddings = HuggingFaceEmbeddings(
            model_name="./text2vec-large-chinese",
            model_kwargs={'device': device}
        )
        logging.info("✅ 嵌入模型初始化成功")
        return embeddings
        
    except Exception as e:
        logging.error(f"嵌入模型初始化失敗: {e}")
        
        # 嘗試備用方案：使用線上模型
        st.warning("⚠️ 本地模型載入失敗，嘗試使用線上備用模型...")
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="GanymedeNil/text2vec-large-chinese",
                model_kwargs={'device': device}
            )
            logging.info("✅ 使用線上模型初始化成功")
            return embeddings
        except Exception as e2:
            logging.error(f"線上模型也失敗: {e2}")
            st.error(f"❌ 所有模型載入方式都失敗了")
            st.error("請嘗試以下解決方案：")
            st.error("1. 刪除 './text2vec-large-chinese' 資料夾後重新啟動")
            st.error("2. 檢查網路連接")
            st.error("3. 確保有足夠的磁碟空間")
            st.stop()

@st.cache_resource
def get_llm() -> Ollama:
    logging.info("正在初始化 LLM 模型...")
    try:
        llm = Ollama(model="gemma3n:latest")
        logging.info("✅ LLM 模型初始化成功")
        return llm
    except Exception as e:
        logging.error(f"LLM 初始化失敗: {e}")
        st.error("❌ LLM 模型初始化失敗，請確保 Ollama 服務正在運行")
        st.error("請執行：ollama serve")
        st.stop()

class SoochowQASystem:
    RELEVANCE_SCORE_THRESHOLD = -600.0

    FALLBACK_MAP = {
        ('註冊', '選課', '成績', '學分', '畢業', '雙主修', '輔系', '招生', '課務'): {
            'name': '東吳大學教務處',
            'url': 'http://www.scu.edu.tw/acad/'
        },
        ('宿舍', '住宿', '床位', '社團', '請假', '操行', '德育', '群育', '心理諮商', '健康'): {
            'name': '東吳大學學生事務處',
            'url': 'https://www.scu.edu.tw/osa/'
        },
        ('採購', '繳費', '出納', '場地', '營繕', '環安', '校園安全'): {
            'name': '東吳大學總務處',
            'url': 'https://www.scu.edu.tw/oga/'
        },
        ('研究計畫', '學術倫理', '校務發展', '評鑑'): {
            'name': '東吳大學研究發展處',
            'url': 'https://www.scu.edu.tw/ord/'
        },
        ('圖書館', '借書', '還書', '自習', '資料庫'): {
            'name': '東吳大學圖書館',
            'url': 'https://www.library.scu.edu.tw/'
        },
        ('資料科學系', '資科系', '大數據', '巨量資料'): {
            'name': '東吳大學巨量資料管理學院',
            'url': 'https://bigdata.scu.edu.tw/'
        }
    }
    
    DEFAULT_FALLBACK = {
        'name': '東吳大學',
        'url': 'http://www.scu.edu.tw/'
    }

    def __init__(self):
        try:
            self.embeddings = get_embeddings()
            self.llm = get_llm()
            self.vector_store = self._init_vector_store()
            
            # 初始化對話記憶
            if "conversation_memory" not in st.session_state:
                st.session_state.conversation_memory = ConversationMemory()
            self.memory = st.session_state.conversation_memory
            
        except Exception as e:
            logging.error(f"SoochowQASystem 初始化失敗: {e}")
            raise

    def _init_vector_store(self) -> Chroma:
        if "vector_store" not in st.session_state:
            logging.info("正在載入 Chroma 向量資料庫...")
            try:
                if not os.path.exists("chroma_db"):
                    st.error("❌ 找不到 'chroma_db' 資料庫目錄")
                    st.error("請先執行 test1.py 來建立向量資料庫")
                    st.stop()
                
                st.session_state["vector_store"] = Chroma(
                    persist_directory="chroma_db",
                    embedding_function=self.embeddings
                )
                logging.info("✅ 向量資料庫載入成功")
            except Exception as e:
                logging.error(f"向量資料庫載入失敗: {e}")
                st.error("❌ 向量資料庫載入失敗")
                st.error("請確保 'chroma_db' 目錄存在且完整")
                st.stop()
                
        return st.session_state["vector_store"]
        
    def _get_fallback_suggestion(self, query: str) -> Dict[str, str]:
        for keywords, suggestion in self.FALLBACK_MAP.items():
            if any(keyword in query for keyword in keywords):
                logging.info(f"觸發回退機制，匹配關鍵字: {keywords}")
                return suggestion
        logging.info("觸發回退機制，使用預設建議")
        return self.DEFAULT_FALLBACK
        
    async def retrieve_relevant_documents(self, query: str) -> List[Tuple[Document, float]]:
        try:
            results_with_scores = await self.vector_store.asimilarity_search_with_relevance_scores(
                query=query,
                k=10
            )
            return results_with_scores
        except Exception as e:
            logging.error(f"文檔檢索錯誤: {e}", exc_info=True)
            raise

    async def generate_response_with_memory(self, query: str, context: str) -> str:
        """生成帶有對話記憶的回應"""
        
        # 獲取對話歷史上下文
        conversation_context = self.memory.get_conversation_context(query)
        
        # 獲取相關的歷史問答
        related_previous_qa = self.memory.get_related_previous_qa(query)
        
        # 構建增強的提示
        memory_context = ""
        if related_previous_qa:
            memory_context = "\n【相關歷史對話】:\n"
            for i, qa in enumerate(related_previous_qa, 1):
                memory_context += f"歷史問題{i}: {qa['user_input']}\n"
                memory_context += f"歷史回答{i}: {qa['system_response'][:150]}...\n\n"
        
        prompt = f"""
===== 東吳大學對話式智能知識引擎 =====

你是東吳大學的AI助理，能夠記住對話歷史並提供連貫的回應。

{conversation_context}

{memory_context}

**增強對話指導原則：**
1. **對話連貫性**：參考對話歷史，建立問題間的連接
2. **個人化回應**：根據用戶提問歷史調整回答詳細程度
3. **上下文理解**：理解代詞和指示詞在對話中的具體指向
4. **追問處理**：識別並適當回應追問性質的問題
5. **對話自然性**：使用自然、口語化的表達方式

【當前檢索資料】：
{context}

【當前使用者提問】：
{query}

請基於以上對話歷史和檢索資料，提供一個連貫、自然、個人化的回應。
"""
        
        try:
            response = await self.llm.ainvoke(prompt)
            logging.info(f"生成回應長度: {len(response)}")
            return response
        except Exception as e:
            logging.error(f"回應生成錯誤: {e}", exc_info=True)
            raise

    async def process_query_with_memory(self, query: str) -> str:
        """帶有記憶功能的查詢處理"""
        try:
            logging.info(f"處理帶記憶的查詢: {query}")
            
            results_with_scores = await self.retrieve_relevant_documents(query)
            filtered_results = [doc for doc, score in results_with_scores if score > self.RELEVANCE_SCORE_THRESHOLD]
            
            if results_with_scores:
                best_score = results_with_scores[0][1]
                logging.info(f"檢索到 {len(results_with_scores)} 個文檔，最佳分數: {best_score:.4f}")

            if not filtered_results:
                logging.warning(f"查詢 '{query}' 在相關性過濾後無結果")
                suggestion = self._get_fallback_suggestion(query)
                
                # 檢查是否為追問
                if len(self.memory.conversation_history) > 0:
                    if any(word in query.lower() for word in ['更多', '其他', '還有', '詳細', '那', '這']):
                        fallback_response = (f"根據我們剛才的對話，我了解您想知道更多資訊。\n\n"
                                          f"雖然我在資料庫中找不到更詳細的相關資料，但建議您直接聯繫 **{suggestion['name']}** "
                                          f"以獲取最完整的資訊：\n🔗 [{suggestion['name']}]({suggestion['url']})")
                    else:
                        fallback_response = (f"⚠ 抱歉，我在資料庫中找不到與此問題高度相關的答案。\n\n"
                                          f"建議您參考 **{suggestion['name']}** 獲取準確資訊：\n"
                                          f"🔗 [{suggestion['name']}]({suggestion['url']})")
                else:
                    fallback_response = (f"⚠ 系統中查無此問題的明確答案。\n\n"
                                      f"建議您參考 **{suggestion['name']}** 獲取準確資訊：\n"
                                      f"🔗 [{suggestion['name']}]({suggestion['url']})")
                
                # 記錄到對話歷史
                self.memory.add_exchange(query, fallback_response, "fallback")
                return fallback_response
            
            context = "\n\n".join([
                f"【來源】{doc.metadata.get('source_url', '無網址')}\n{doc.page_content}"
                for doc in filtered_results
            ])
            
            logging.info(f"使用 {len(filtered_results)} 個文檔作為上下文")
            
            # 使用帶記憶的回應生成
            response = await self.generate_response_with_memory(query, context)
            
            # 記錄到對話歷史
            self.memory.add_exchange(query, response, f"檢索到 {len(filtered_results)} 個文檔")
            
            return response

        except Exception as e:
            error_msg = f"處理您的問題時發生錯誤，請稍後再試。"
            logging.error(f"查詢處理錯誤: {e}", exc_info=True)
            return error_msg


async def main():
    st.set_page_config(
        page_title="東吳大學對話式智能問答系統",
        page_icon="🎓",
        layout="wide"
    )
    
    st.title("東吳大學對話式智能問答系統 🎓")
    st.caption("💬 具備對話記憶功能的智能助理")

    # 系統狀態檢查
    with st.expander("🔍 系統狀態檢查", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            model_exists = os.path.exists("./text2vec-large-chinese")
            if model_exists:
                required_files = ['config.json']
                model_files = os.listdir("./text2vec-large-chinese") if model_exists else []
                model_complete = any(f in model_files for f in ['pytorch_model.bin', 'model.safetensors'])
                st.success("✅ 嵌入模型" if model_complete else "⚠️ 模型不完整")
            else:
                st.error("❌ 嵌入模型缺失")
        
        with col2:
            db_exists = os.path.exists("chroma_db")
            st.success("✅ 向量資料庫") if db_exists else st.error("❌ 向量資料庫缺失")
        
        with col3:
            device_info = "🚀 GPU (MPS)" if torch.backends.mps.is_available() else "🚀 GPU (CUDA)" if torch.cuda.is_available() else "💻 CPU"
            st.info(f"運算設備: {device_info}")

    try:
        qa_system = SoochowQASystem()
        
        # 側邊欄功能
        with st.sidebar:
            st.header("💬 對話管理")
            
            # 顯示對話歷史統計
            history_count = len(qa_system.memory.conversation_history)
            st.metric("對話輪數", history_count)
            
            # 清除對話歷史按鈕
            if st.button("🗑️ 清除對話歷史"):
                qa_system.memory.clear_history()
                st.success("對話歷史已清除！")
                st.rerun()
            
            # 顯示最近的對話
            if history_count > 0:
                st.subheader("📋 最近對話")
                for i, exchange in enumerate(qa_system.memory.conversation_history[-3:], 1):
                    with st.expander(f"第{i}輪: {exchange['user_input'][:20]}..."):
                        st.write(f"**問題:** {exchange['user_input']}")
                        st.write(f"**回答:** {exchange['system_response'][:200]}...")
                        st.caption(f"⏰ {exchange['timestamp']}")

        # 主要輸入區域
        query = st.text_input(
            "💭 請輸入您的問題",
            placeholder="例如：東吳大學資料科學系在哪？",
            help="您可以進行多輪對話，系統會記住我們的談話內容"
        )

        if query:
            with st.spinner('🤔 正在思考並生成回答...'):
                response = await qa_system.process_query_with_memory(query)
                
                # 顯示回答
                st.markdown("### 📝 回答：")
                st.markdown(response, unsafe_allow_html=True)
                
                # 如果有對話歷史，顯示相關的歷史對話
                related_qa = qa_system.memory.get_related_previous_qa(query)
                if related_qa:
                    with st.expander("📚 相關的歷史對話"):
                        for i, qa in enumerate(related_qa, 1):
                            st.write(f"**相關問題{i}:** {qa['user_input']}")
                            st.write(f"**當時回答:** {qa['system_response'][:150]}...")
                            st.markdown("---")

    except Exception as e:
        st.error("❌ 系統發生未預期的錯誤")
        with st.expander("錯誤詳情"):
            st.code(str(e))
        logging.error(f"主程式未預期錯誤: {e}", exc_info=True)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 系統已關閉")
        pass