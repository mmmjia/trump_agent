#!/usr/bin/env python
# scripts/build_psychology_vectorstores.py
"""
为三个心理学流派构建向量数据库，输出格式与 VectorRetriever 完全匹配。

目录结构（脚本运行后）：
  <同级目录>/
  ├── AI_AGENT_PERSONAL/        ← 本项目
  │   └── scripts/
  │       └── build_psychology_vectorstores.py  (本文件)
  └── knowledge/
      ├── psychology_books/     ← 原始文档放这里
      │   ├── trait_psychology/
      │   ├── social_cognitive/
      │   └── Behaviorism_Learning_Theory/
      └── vector_stores/        ← 构建后的向量库输出到这里
          ├── trait_psychology/
          ├── social_cognitive/
          └── behaviorism_learning_theory/

用法：
  cd AI_AGENT_PERSONAL
  python scripts/build_psychology_vectorstores.py
"""

import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent        # AI_AGENT_PERSONAL/
KNOWLEDGE_ROOT = PROJECT_ROOT.parent / "knowledge"           # 同级 knowledge/
SOURCE_ROOT = KNOWLEDGE_ROOT / "psychology_books"            # 原始文档
VECTOR_STORE_ROOT = KNOWLEDGE_ROOT / "vector_stores"         # 向量库输出

sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from rag.vector_retriever import VectorRetriever

# 三个流派：源文件夹名 → 向量库子目录名（统一小写）
DISCIPLINES = {
    "trait_psychology":           "trait_psychology",
    "social_cognitive":           "social_cognitive",
    "Behaviorism_Learning_Theory": "behaviorism_learning_theory",
}

# 文本分块参数（1024 维模型上下文窗口更大，适当增大 chunk size）
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 100

# 需要删除的旧格式文件（Chroma 遗留）
STALE_FILES = ("chroma.sqlite3",)
STALE_DIRS  = ("chroma",)


# ------------------------------------------------------------------ #
#  清理旧索引
# ------------------------------------------------------------------ #

def clean_store_dir(store_dir: Path):
    """删除旧格式文件，保证本次重建的 FAISS 索引不被旧 Chroma 抢先加载"""
    removed = []
    for fname in STALE_FILES:
        f = store_dir / fname
        if f.exists():
            f.unlink()
            removed.append(fname)
    for dname in STALE_DIRS:
        d = store_dir / dname
        if d.exists():
            shutil.rmtree(d)
            removed.append(dname + "/")
    # 也清除旧 FAISS 文件，确保完全重建
    for fname in ("index.faiss", "docs.pkl"):
        f = store_dir / fname
        if f.exists():
            f.unlink()
            removed.append(fname)
    if removed:
        print(f"  已清除旧索引文件: {', '.join(removed)}")


# ------------------------------------------------------------------ #
#  文档加载
# ------------------------------------------------------------------ #

def load_documents_from_folder(folder_path: Path):
    """递归加载 .txt / .md / .pdf 文件，返回 LangChain Document 列表"""
    raw_docs = []
    for ext in ("*.txt", "*.md", "*.pdf"):
        for file_path in folder_path.glob(f"**/{ext}"):
            print(f"  加载: {file_path.name}")
            try:
                if file_path.suffix.lower() == ".pdf":
                    from langchain_community.document_loaders import PyPDFLoader
                    docs = PyPDFLoader(str(file_path)).load()
                else:
                    from langchain_community.document_loaders import TextLoader
                    docs = TextLoader(str(file_path), encoding="utf-8",
                                      autodetect_encoding=True).load()
                raw_docs.extend(docs)
            except Exception as e:
                print(f"  警告: 加载 {file_path.name} 失败: {e}")
    return raw_docs


def split_documents(raw_docs):
    """将文档切分为文本块"""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )
    split = splitter.split_documents(raw_docs)
    texts     = [doc.page_content for doc in split]
    metadatas = [{"source": doc.metadata.get("source", "")} for doc in split]
    return texts, metadatas


# ------------------------------------------------------------------ #
#  向量库构建
# ------------------------------------------------------------------ #

def build_vectorstore(source_dir: Path, store_dir: Path, shared_model):
    print(f"\n{'─'*60}")
    print(f"流派:     {source_dir.name}")
    print(f"源文件:   {source_dir}")
    print(f"输出目录: {store_dir}")

    if not source_dir.exists():
        print(f"  ✗ 源文件夹不存在，跳过")
        return False

    raw_docs = load_documents_from_folder(source_dir)
    if not raw_docs:
        print(f"  ✗ 未找到任何文档（.txt / .md / .pdf），跳过")
        return False
    print(f"  ✓ 加载 {len(raw_docs)} 个文件")

    texts, metadatas = split_documents(raw_docs)
    if not texts:
        print(f"  ✗ 切分后无文本块，跳过")
        return False
    print(f"  ✓ 切分为 {len(texts)} 个文本块（chunk_size={CHUNK_SIZE}）")

    # 清理旧索引，确保干净重建
    store_dir.mkdir(parents=True, exist_ok=True)
    clean_store_dir(store_dir)

    # 传入共享 embedding 模型，避免重复加载
    retriever = VectorRetriever(
        vector_store_path=store_dir,
        embedding_model=shared_model,
    )
    retriever.build_index(chunks=texts, metadatas=metadatas)
    retriever.save()
    print(f"  ✓ FAISS 向量库已保存: {store_dir}")
    return True


# ------------------------------------------------------------------ #
#  主流程
# ------------------------------------------------------------------ #

def main():
    print("=" * 60)
    print("心理学向量数据库构建工具（FAISS 格式）")
    print(f"Embedding 模型:   {settings.EMBEDDING_MODEL}")
    print(f"原始文档根目录:   {SOURCE_ROOT}")
    print(f"向量库输出目录:   {VECTOR_STORE_ROOT}")
    print(f"Chunk size:       {CHUNK_SIZE}  overlap: {CHUNK_OVERLAP}")
    print("=" * 60)

    VECTOR_STORE_ROOT.mkdir(parents=True, exist_ok=True)

    # 加载一次 embedding 模型，三个流派共享，避免重复加载
    print(f"\n正在加载 Embedding 模型: {settings.EMBEDDING_MODEL} ...")
    from sentence_transformers import SentenceTransformer
    shared_model = SentenceTransformer(settings.EMBEDDING_MODEL)

    # 快速验证向量维度
    test_vec = shared_model.encode(["test"])
    dim = test_vec.shape[1]
    print(f"向量维度确认: {dim} 维\n")

    results = {}
    for src_name, store_name in DISCIPLINES.items():
        ok = build_vectorstore(
            source_dir=SOURCE_ROOT / src_name,
            store_dir=VECTOR_STORE_ROOT / store_name,
            shared_model=shared_model,
        )
        results[store_name] = "✓" if ok else "✗"

    print("\n" + "=" * 60)
    print("构建结果：")
    for name, status in results.items():
        print(f"  {status}  {name}")
    print(f"\n向量维度: {dim} 维  模型: {settings.EMBEDDING_MODEL}")
    print("=" * 60)


if __name__ == "__main__":
    main()
