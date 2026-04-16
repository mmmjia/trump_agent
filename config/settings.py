# config/settings.py
"""
全局配置：加载环境变量，管理模型、路径、RAG 参数等。
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# 加载 .env 文件
load_dotenv()

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings:
    """应用程序设置"""
    
    # ---------- LLM 配置 ----------
    # 主模型：DeepSeek
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    
    # 备用模型（可选）
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    # 推理参数
    TEMPERATURE: float = 0.3       # 较低温度保证分析一致性
    MAX_TOKENS: int = 4096
    
    # ---------- RAG 配置 ----------
    # 知识库根目录：优先读取 .env 中的 KNOWLEDGE_DIR，否则自动推断为 ai_agent_personal 的同级目录
    KNOWLEDGE_DIR: Path = Path(
        os.getenv("KNOWLEDGE_DIR") or str(BASE_DIR.parent / "knowledge" / "vector_stores")
    )
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # 检索参数
    TOP_K_RETRIEVAL: int = 5       # 每次检索返回的最相关文档块数量
    CHUNK_SIZE: int = 512          # 文本分块大小（字符数）
    CHUNK_OVERLAP: int = 50        # 分块重叠大小

    # 三个专项知识库（对应三大心理学理论）
    TRAIT_PSYCHOLOGY_STORE: Path = KNOWLEDGE_DIR / "trait_psychology"        # 大五人格 + 布迪厄
    SOCIAL_COGNITIVE_STORE: Path = KNOWLEDGE_DIR / "social_cognitive"        # 班杜拉社会认知
    TPB_STORE: Path = KNOWLEDGE_DIR / "behaviorism_learning_theory"          # 计划行为理论（TPB）

    # 保留旧路径别名，向后兼容
    VECTOR_STORE_PATH: Path = KNOWLEDGE_DIR
    PSYCHOLOGY_VECTOR_STORE: Path = TRAIT_PSYCHOLOGY_STORE
    BIOGRAPHY_VECTOR_STORE: Path = KNOWLEDGE_DIR / "figure_biography"
    
    # ---------- 知识图谱（可选）----------
    NEO4J_URI: str = os.getenv("NEO4J_URI", "")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "")
    USE_KNOWLEDGE_GRAPH: bool = bool(NEO4J_URI and NEO4J_USER)  # 如果配置了才启用
    
    # ---------- 数据路径 ----------
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    PIPELINE_STATE_DIR: Path = DATA_DIR / "pipeline_state"
    
    # ---------- 日志 ----------
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # ---------- 辅助方法 ----------
    @classmethod
    def ensure_directories(cls):
        """确保所有必要的目录存在"""
        for dir_path in [
            cls.KNOWLEDGE_DIR,
            cls.TRAIT_PSYCHOLOGY_STORE,
            cls.SOCIAL_COGNITIVE_STORE,
            cls.TPB_STORE,
            cls.BIOGRAPHY_VECTOR_STORE,
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.PIPELINE_STATE_DIR,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_llm_config(cls) -> dict:
        """返回 LLM 初始化参数（兼容 OpenAI SDK）"""
        return {
            "api_key": cls.DEEPSEEK_API_KEY,
            "base_url": cls.DEEPSEEK_BASE_URL,
            "model": cls.DEEPSEEK_MODEL,
            "temperature": cls.TEMPERATURE,
            "max_tokens": cls.MAX_TOKENS,
        }

# 创建全局单例
settings = Settings()