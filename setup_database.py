from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from huggingface_hub import snapshot_download
import os
import shutil
from tqdm.asyncio import tqdm
import asyncio
import torch  # <-- 新增導入

def download_embedding_model():
    model_name = "GanymedeNil/text2vec-large-chinese"
    local_dir = "./text2vec-large-chinese"
    print(f"準備下載模型 '{model_name}' 到 '{local_dir}'...")
    try:
        snapshot_download(repo_id=model_name, local_dir=local_dir)
        print(f"模型 '{model_name}' 已成功下載到 '{local_dir}'")
    except Exception as e:
        print(f"下載模型時發生嚴重錯誤: {e}")
        exit(1)

async def load_text_file(file_path: str) -> list:
    try:
        loader = TextLoader(file_path, encoding='utf-8')
        raw_docs = loader.load()

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            possible_url = lines[-1].strip() if lines else ''

        return [
            Document(
                page_content=doc.page_content,
                metadata={"source_url": possible_url}
            ) for doc in raw_docs
        ]
    except Exception as e:
        print(f"讀取文件時發生錯誤：{file_path} - {e}")
        return []

async def load_documents_from_folder(folder_path: str) -> list:
    documents = []
    tasks = []

    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.txt'):
                file_path = os.path.join(root, filename)
                tasks.append(load_text_file(file_path))

    for result in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="載入文件"):
        try:
            file_documents = await result
            if file_documents:
                documents.extend(file_documents)
        except Exception as e:
            print(f"處理文件時發生錯誤: {e}")
            continue

    return documents

def split_documents(documents: list, chunk_size: int = 600, chunk_overlap: int = 100) -> list:
    """
    將 Document 物件列表分割成更小的文本塊。
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    texts = text_splitter.split_documents(documents)
    return texts

# ===== 修改後的函數：智能檢測硬體設備 =====
async def create_embeddings_and_store(texts: list, persist_directory: str):
    """
    初始化嵌入模型並創建/更新向量資料庫。
    此函數會自動檢測可用的硬體 (CUDA for Nvidia, MPS for Apple Silicon, CPU as fallback)。
    """
    # 步驟 1: 智能檢測可用的硬體設備
    if torch.cuda.is_available():
        device = 'cuda'
        print("檢測到 NVIDIA GPU，將使用 CUDA 進行加速。")
    elif torch.backends.mps.is_available():
        device = 'mps'
        print("檢測到 Apple Silicon GPU，將使用 MPS 進行加速。")
    else:
        device = 'cpu'
        print("未檢測到可用 GPU，將使用 CPU。處理速度會較慢。")

    # 步驟 2: 初始化嵌入模型客戶端
    try:
        print("正在初始化嵌入模型...")
        # 使用檢測到的 device
        embeddings_client = HuggingFaceEmbeddings(
            model_name="./text2vec-large-chinese",
            model_kwargs={'device': device} # 使用動態檢測到的設備
        )
        print("嵌入模型初始化完成。")
    except Exception as e:
        print(f"建立 embeddings 時發生錯誤: {e}")
        # 增加提示：檢查模型文件是否存在
        print("提示：請確保 './text2vec-large-chinese' 資料夾存在且包含完整的模型文件。如果問題持續，請嘗試手動刪除該資料夾後重新執行此腳本。")
        exit(1)

    # 步驟 3: 清理舊的資料庫
    if os.path.exists(persist_directory):
        try:
            print(f"檢測到舊的向量資料庫，正在刪除 '{persist_directory}'...")
            shutil.rmtree(persist_directory)
            print(f"已成功刪除舊資料庫。")
        except Exception as e:
            print(f"刪除舊向量資料庫時發生錯誤：{str(e)}")
            exit(1)

    # 步驟 4: 創建新的向量資料庫
    try:
        print("開始建立新的向量資料庫 (ChromaDB)...")
        vectordb = await Chroma.afrom_documents(
            documents=texts,
            embedding=embeddings_client,
            persist_directory=persist_directory
        )
        print(f"向量資料庫建立完成並儲存於 '{persist_directory}'。")
    except Exception as e:
        print(f"建立向量資料庫時發生嚴重錯誤：{str(e)}")
        exit(1)
# ===== 修改結束 =====

async def main():
    backup_folder = 'test'
    persist_directory = "chroma_db"
    model_dir = "./text2vec-large-chinese"

    # 檢查模型目錄是否存在，如果不存在則下載
    if not os.path.exists(model_dir):
        download_embedding_model()

    if not os.path.exists(backup_folder):
        print(f"錯誤：找不到來源資料夾 '{backup_folder}'")
        exit(1)

    documents = await load_documents_from_folder(backup_folder)
    if not documents:
        print(f"錯誤：在 '{backup_folder}' 資料夾中沒有找到任何可讀取的 .txt 文件")
        exit(1)
    print(f"成功載入 {len(documents)} 份原始文件。")
    
    texts = split_documents(documents)
    print(f"文件被分割成 {len(texts)} 個文本塊。")
    
    await create_embeddings_and_store(texts, persist_directory)

if __name__ == "__main__":
    # 提醒用戶 LangChain 棄用警告
    print("提示：如果看到 'LangChainDeprecationWarning'，這是正常現象，表示某些函式庫版本將更新，暫不影響當前功能。")
    asyncio.run(main())