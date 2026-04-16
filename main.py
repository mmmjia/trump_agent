"""
心理传记学 AI Agent 主入口
Psychobiography AI Agent — Main Entry Point

用法 / Usage:
  python main.py --figure "Donald Trump" --scenario "如果此人面临AI监管决策，他会怎么做？"
  python main.py --figure "Steve Jobs" --biography path/to/biography.txt --scenario "..."
"""

import os
import argparse

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from sentence_transformers import SentenceTransformer

from agents.orchestrator import OrchestratorAgent
from rag.vector_retriever import VectorRetriever
from config.settings import settings

load_dotenv()


def build_llm() -> ChatOpenAI:
    """创建共享 LLM 实例（使用 DeepSeek，兼容 OpenAI API）
    Create a shared LLM instance (using DeepSeek with OpenAI-compatible API)"""
    return ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        temperature=float(os.getenv("TEMPERATURE", "0.3")),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
    )


def main():
    parser = argparse.ArgumentParser(description="心理传记学 AI Agent — 公众人物分析 / Psychobiography AI Agent — Public Figure Analysis")
    parser.add_argument("--figure",    default="Steve Jobs",
                        help="分析对象姓名（英文）/ Figure name to analyze (English)")
    parser.add_argument("--scenario",  default="如果此人面临重大战略决策，他会怎么做？",
                        help="预测情境描述 / Hypothetical scenario description for behavioral prediction")
    parser.add_argument("--biography", default=None,
                        help="本地传记文本路径（可选；不提供则自动从网络采集）/ Local biography file path (optional; auto-collected from web if omitted)")
    parser.add_argument("--refresh",   action="store_true",
                        help="强制重新分析，忽略缓存 / Force re-analysis, ignoring cached results")
    args = parser.parse_args()

    # 单一共享 LLM（所有 agent 复用同一实例）
    # Single shared LLM instance reused by all agents
    llm = build_llm()

    # 共享 Embedding 模型（只加载一次，三个检索器复用）
    # Shared embedding model loaded once and reused by all three retrievers
    print(f"正在加载 Embedding 模型 / Loading embedding model: {settings.EMBEDDING_MODEL} ...")
    shared_embedding = SentenceTransformer(settings.EMBEDDING_MODEL)

    # 三个专项知识库检索器（对应三大心理学理论）
    # Three specialized knowledge-base retrievers (one per psychological theory)
    trait_retriever            = VectorRetriever(settings.TRAIT_PSYCHOLOGY_STORE,  shared_embedding)
    social_cognitive_retriever = VectorRetriever(settings.SOCIAL_COGNITIVE_STORE,  shared_embedding)
    tpb_retriever              = VectorRetriever(settings.TPB_STORE,               shared_embedding)

    orchestrator = OrchestratorAgent(
        llm=llm,
        trait_retriever=trait_retriever,
        social_cognitive_retriever=social_cognitive_retriever,
        tpb_retriever=tpb_retriever,
    )

    # 读取本地传记（可选）/ Load local biography file (optional)
    biography_text = None
    if args.biography:
        with open(args.biography, encoding="utf-8") as f:
            biography_text = f.read()

    # 运行完整分析流水线 / Run the full analysis pipeline
    result = orchestrator.run(
        figure_name=args.figure,
        new_scenario=args.scenario,
        biography_text=biography_text,
        force_refresh=args.refresh,
    )

    # 输出报告 / Print report
    print("\n" + "=" * 60)
    print("心理传记学分析报告 / Psychobiography Analysis Report")
    print("=" * 60)

    print("\n【行为倾向预测 / Behavioral Tendency Prediction】")
    prediction = result.get("prediction", {})
    if isinstance(prediction, dict):
        print(prediction.get("prediction", prediction))
    else:
        print(prediction)


if __name__ == "__main__":
    main()
