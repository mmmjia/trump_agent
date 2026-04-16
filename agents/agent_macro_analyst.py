# agents/agent_macro_analyst.py
"""
Agent 3: 宏观分析（大五人格 + 布迪厄）— 集体侧写模式
Agent 3: Macro Analysis (Big Five Personality + Bourdieu) — Collective Profiling Mode

输入：全部事件列表
Input: Full list of biographical events

输出：一次性综合心理人格侧写（bigfive + bourdieu + behavioral_patterns + psychological_portrait）
Output: One-shot comprehensive psychological portrait (Big Five + Bourdieu + behavioral patterns + portrait)

结果自动保存至 data/processed/{figure_name}/macro_analysis.json
Results are automatically saved to data/processed/{figure_name}/macro_analysis.json
"""

import json
from pathlib import Path
from typing import List, Dict, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough

from config.prompts import MACRO_ANALYST_SYSTEM, MACRO_ANALYSIS_TASK_TEMPLATE
from config.settings import settings

# 每个事件在传给 LLM 时保留的字段（控制 token 用量）
# Fields retained per event when passing to LLM (controls token usage)
_EVENT_FIELDS = ("event_id", "time_period", "category", "title", "summary",
                 "impact_level", "outcome", "key_actors")

# summary 最大保留字符数 / Maximum characters retained from each event summary
_SUMMARY_LIMIT = 300


class MacroAnalystAgent:
    def __init__(self, llm, trait_retriever, data_dir: Optional[str] = None):
        self.llm = llm
        self.retriever = trait_retriever
        self.data_dir = Path(data_dir) if data_dir else settings.PROCESSED_DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chain = self._build_chain()

    def _build_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", MACRO_ANALYST_SYSTEM),
            ("human", MACRO_ANALYSIS_TASK_TEMPLATE),
        ])
        # RAG 注入：在调用 LLM 前自动检索理论参考文献
        # RAG injection: automatically retrieve theory references before calling LLM
        return (
            RunnablePassthrough.assign(
                theory_context=lambda x: self._retrieve_theory(x["figure_name"])
            )
            | prompt
            | self.llm
            | JsonOutputParser()
        )

    def _retrieve_theory(self, figure_name: str) -> str:
        """从大五人格 / 布迪厄知识库检索相关理论片段
        Retrieve relevant theory excerpts from the Big Five / Bourdieu knowledge base"""
        query = f"大五人格 布迪厄 场域 资本 惯习 心理传记 人格侧写 {figure_name}"
        try:
            results = self.retriever.retrieve(query, top_k=3)
            if isinstance(results, list):
                return "\n\n".join([doc.get("text", str(doc)) for doc in results])
            return str(results)
        except Exception as e:
            print(f"[MacroAnalyst] RAG 检索失败（将跳过理论参考）/ RAG retrieval failed (skipping theory context): {e}")
            return ""

    def _condense_events(self, events: List[Dict]) -> str:
        """压缩事件列表以控制 token 用量，按 event_id 排序
        Condense event list to control token usage, sorted by event_id"""
        condensed = []
        for ev in sorted(events, key=lambda e: e.get("event_id", 0)):
            item = {k: ev.get(k, "") for k in _EVENT_FIELDS}
            if isinstance(item.get("summary"), str):
                item["summary"] = item["summary"][:_SUMMARY_LIMIT]
            condensed.append(item)
        return json.dumps(condensed, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------ #
    #  缓存 I/O / Cache I/O
    # ------------------------------------------------------------------ #

    def _get_file_path(self, figure_name: str) -> Path:
        # 文件名不含人名前缀，因为文件夹已经是人名目录
        # Filename has no person-name prefix; the parent folder already identifies the figure
        return self.data_dir / "macro_analysis.json"

    def save_analysis(self, figure_name: str, result: Dict):
        file_path = self._get_file_path(figure_name)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[MacroAnalyst] 已保存宏观分析至 / Saved macro analysis to {file_path}")

    def load_analysis(self, figure_name: str) -> Optional[Dict]:
        file_path = self._get_file_path(figure_name)
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                result = json.load(f)
            print(f"[MacroAnalyst] 已加载宏观分析 / Loaded macro analysis from {file_path}")
            return result
        return None

    # ------------------------------------------------------------------ #
    #  主分析入口（集体侧写）/ Main Analysis Entry Point (Collective Profiling)
    # ------------------------------------------------------------------ #

    def analyze_events(
        self,
        events: List[Dict],
        figure_name: str,
        force_refresh: bool = False,
    ) -> Dict:
        """
        对全部事件做一次性整体分析，输出综合心理人格侧写。
        Performs a single comprehensive analysis over all events and returns a psychological portrait.

        结果自动保存至 data/processed/{figure_name}/。
        Results are automatically saved to data/processed/{figure_name}/.
        """
        # 命中缓存则直接返回，跳过 LLM 调用
        # Return cached result immediately if available, skipping LLM call
        if not force_refresh:
            cached = self.load_analysis(figure_name)
            if cached is not None:
                return cached

        print(f"[MacroAnalyst] 开始集体侧写 / Starting collective profiling: {figure_name}（共 {len(events)} 个事件 / {len(events)} events）...")
        events_json = self._condense_events(events)
        try:
            result = self.chain.invoke({
                "figure_name": figure_name,
                "event_count": len(events),
                "events_json": events_json,
            })
        except Exception as e:
            print(f"[MacroAnalyst] 分析失败 / Analysis failed: {e}")
            result = {"error": str(e)}

        output = {"figure_name": figure_name, **result}
        self.save_analysis(figure_name, output)
        return output
