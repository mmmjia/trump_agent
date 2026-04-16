# rag/vector_retriever.py
"""
向量检索模块：自动检测 Chroma 或 FAISS 格式并加载。
- Chroma 格式：存在 chroma.sqlite3（由 build 脚本旧版生成）
- FAISS 格式：存在 index.faiss + docs.pkl（由新版 build 脚本生成）
新建索引统一使用 FAISS。
"""

import pickle
from typing import List, Dict, Any, Optional
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

from config.settings import settings


class VectorRetriever:
    """
    向量检索器：支持 Chroma 和 FAISS 两种存储格式，接口统一。
    传入共享 embedding_model 实例可避免重复加载，节省内存。
    """

    def __init__(self, vector_store_path=None, embedding_model=None):
        """
        Args:
            vector_store_path: 向量库路径（str 或 Path）
            embedding_model:   已加载的 SentenceTransformer（可选，共享用）
        """
        self.vector_store_path = (
            Path(vector_store_path) if vector_store_path is not None
            else settings.PSYCHOLOGY_VECTOR_STORE
        )

        self.embedding_model = embedding_model or self._load_embedding_model()

        # FAISS 专用状态
        self._faiss_index = None
        self._faiss_docs: List[Dict] = []

        # Chroma 专用状态
        self._chroma_collection = None

        # 自动加载
        self._backend = None  # "faiss" | "chroma" | None
        if self.vector_store_path.exists():
            self._auto_load()

    # ------------------------------------------------------------------ #
    #  初始化辅助
    # ------------------------------------------------------------------ #

    def _load_embedding_model(self) -> SentenceTransformer:
        print(f"正在加载 Embedding 模型: {settings.EMBEDDING_MODEL} ...")
        return SentenceTransformer(settings.EMBEDDING_MODEL)

    def _auto_load(self):
        """根据目录内容自动选择加载格式。
        优先 FAISS（新格式，维度与当前 embedding 模型一致），
        仅当不存在 FAISS 时才回退到旧 Chroma 格式。
        """
        if (self.vector_store_path / "index.faiss").exists():
            self._load_faiss()
        elif (self.vector_store_path / "chroma.sqlite3").exists():
            self._load_chroma()
        else:
            print(f"未找到索引文件，请先构建索引。路径: {self.vector_store_path}")

    # ------------------------------------------------------------------ #
    #  Chroma 后端
    # ------------------------------------------------------------------ #

    def _load_chroma(self):
        """加载已有的 Chroma 向量库"""
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            client = chromadb.PersistentClient(
                path=str(self.vector_store_path),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            collections = client.list_collections()
            if not collections:
                print(f"Chroma 库存在但无集合: {self.vector_store_path}")
                return
            # 取第一个集合（每个流派只有一个）
            self._chroma_collection = client.get_collection(collections[0].name)
            self._backend = "chroma"
            count = self._chroma_collection.count()
            print(f"已加载 Chroma 索引（{count} 个向量）: {self.vector_store_path.name}")
        except ImportError:
            print("未安装 chromadb，无法加载 Chroma 格式。请运行: pip install chromadb")
        except Exception as e:
            print(f"加载 Chroma 失败: {e}")

    # ------------------------------------------------------------------ #
    #  FAISS 后端
    # ------------------------------------------------------------------ #

    def _load_faiss(self):
        """加载 FAISS 索引"""
        index_path = self.vector_store_path / "index.faiss"
        docs_path  = self.vector_store_path / "docs.pkl"
        if not index_path.exists() or not docs_path.exists():
            print(f"未找到 FAISS 文件: {self.vector_store_path}")
            return
        self._faiss_index = faiss.read_index(str(index_path))
        with open(docs_path, "rb") as f:
            self._faiss_docs = pickle.load(f)
        self._backend = "faiss"
        print(f"已加载 FAISS 索引（{self._faiss_index.ntotal} 个向量）: {self.vector_store_path.name}")

    def build_index(self, chunks: List[str], metadatas: Optional[List[Dict]] = None):
        """从文本块构建 FAISS 索引（新建时始终使用 FAISS）"""
        if not chunks:
            raise ValueError("chunks 不能为空")
        print(f"正在为 {len(chunks)} 个文档块生成嵌入向量...")
        embeddings = self.embedding_model.encode(chunks, show_progress_bar=True)
        embeddings = np.array(embeddings).astype("float32")
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        self._faiss_index = faiss.IndexFlatIP(dim)
        self._faiss_index.add(embeddings)
        self._faiss_docs = [
            {"text": chunk, "metadata": metadatas[i] if metadatas else {}}
            for i, chunk in enumerate(chunks)
        ]
        self._backend = "faiss"
        print(f"索引构建完成，包含 {self._faiss_index.ntotal} 个向量。")

    def save(self):
        """保存 FAISS 索引到磁盘"""
        if self._faiss_index is None:
            raise RuntimeError("FAISS 索引未构建，无法保存")
        self.vector_store_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._faiss_index, str(self.vector_store_path / "index.faiss"))
        with open(self.vector_store_path / "docs.pkl", "wb") as f:
            pickle.dump(self._faiss_docs, f)
        print(f"FAISS 索引已保存至 {self.vector_store_path}")

    # ------------------------------------------------------------------ #
    #  统一检索接口
    # ------------------------------------------------------------------ #

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        检索最相关文档块。后端为 Chroma 或 FAISS，接口相同。
        若索引未加载，返回空列表（LLM 依赖自身知识继续运行）。
        """
        if self._backend == "chroma":
            return self._retrieve_chroma(query, top_k)
        if self._backend == "faiss":
            return self._retrieve_faiss(query, top_k)
        return []

    def _retrieve_chroma(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        query_vec = self.embedding_model.encode([query]).tolist()
        results = self._chroma_collection.query(
            query_embeddings=query_vec,
            n_results=min(top_k, self._chroma_collection.count()),
        )
        docs, distances = results.get("documents", [[]])[0], results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        return [
            {"text": doc, "metadata": meta, "score": float(1 - dist)}
            for doc, meta, dist in zip(docs, metadatas, distances)
        ]

    def _retrieve_faiss(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        query_vec = np.array(self.embedding_model.encode([query])).astype("float32")
        faiss.normalize_L2(query_vec)
        scores, indices = self._faiss_index.search(query_vec, top_k)
        return [
            {
                "text": self._faiss_docs[idx]["text"],
                "metadata": self._faiss_docs[idx]["metadata"],
                "score": float(score),
            }
            for score, idx in zip(scores[0], indices[0])
            if idx != -1
        ]

    def add_documents(self, chunks: List[str], metadatas: Optional[List[Dict]] = None):
        """动态追加文档块（FAISS 后端）"""
        if self._faiss_index is None:
            self.build_index(chunks, metadatas)
            return
        embeddings = np.array(self.embedding_model.encode(chunks)).astype("float32")
        faiss.normalize_L2(embeddings)
        self._faiss_index.add(embeddings)
        for i, chunk in enumerate(chunks):
            self._faiss_docs.append({
                "text": chunk,
                "metadata": metadatas[i] if metadatas else {},
            })
        print(f"已添加 {len(chunks)} 个文档块，当前总数: {self._faiss_index.ntotal}")

    # ------------------------------------------------------------------ #
    #  向后兼容属性（旧代码用 .index 和 .documents 的地方不报错）
    # ------------------------------------------------------------------ #

    @property
    def index(self):
        return self._faiss_index

    @property
    def documents(self):
        return self._faiss_docs
