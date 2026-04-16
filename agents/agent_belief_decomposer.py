# agents/agent_belief_decomposer.py
"""
Agent 5: 信念侧写（计划行为理论 TPB）— 集体侧写模式
Agent 5: Belief Profiling (Theory of Planned Behavior, TPB) — Collective Profiling Mode

输入：全部事件列表
Input: Full list of biographical events

输出：一次性综合信念体系侧写（core_behavioral_beliefs + normative_beliefs + control_beliefs + evolution + portrait）
Output: One-shot comprehensive belief system profile (behavioral, normative, and control beliefs + evolution + portrait)

结果自动保存至 data/processed/{figure_name}/belief_analysis.json
Results are automatically saved to data/processed/{figure_name}/belief_analysis.json
"""

import json
from pathlib import Path
from typing import List, Dict, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough

from config.prompts import (
    BELIEF_DECOMPOSER_SYSTEM,
    BELIEF_DECOMPOSITION_TASK_TEMPLATE,
    DECOMPOSE_EVENT_PROMPT,
)
from config.settings import settings

# 每个事件传给 LLM 时保留的字段 / Fields retained per event when passed to LLM
_EVENT_FIELDS = ("event_id", "time_period", "category", "title", "summary",
                 "impact_level", "outcome", "key_actors")

# summary 最大保留字符数 / Maximum characters retained from each event summary
_SUMMARY_LIMIT = 300


class BeliefDecomposerAgent:
    def __init__(self, llm, tpb_retriever, data_dir: Optional[str] = None):
        self.llm = llm
        self.retriever = tpb_retriever
        self.data_dir = Path(data_dir) if data_dir else settings.PROCESSED_DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chain = self._build_chain()
        # 保留单事件拆解链（可按需调用，不再用于批量分析）
        # Keep single-event decomposition chain (available on demand, not used in batch flow)
        self._decompose_chain = self._build_decompose_chain()

    def _build_chain(self):
        """集体信念侧写主链 / Main chain for collective belief profiling"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", BELIEF_DECOMPOSER_SYSTEM),
            ("human", BELIEF_DECOMPOSITION_TASK_TEMPLATE),
        ])
        # RAG 注入：调用 LLM 前自动检索 TPB 理论参考
        # RAG injection: automatically retrieve TPB theory references before LLM call
        return (
            RunnablePassthrough.assign(
                tpb_context=lambda x: self._retrieve_tpb(x["figure_name"])
            )
            | prompt
            | self.llm
            | JsonOutputParser()
        )

    def _build_decompose_chain(self):
        """单事件决策拆解链（保留备用）
        Single-event decision decomposition chain (kept as fallback)"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一位事件分析专家。请将大事件拆解为2-4个该人物的具体决策行为，严格输出JSON数组。"),
            ("human", DECOMPOSE_EVENT_PROMPT),
        ])
        return prompt | self.llm | JsonOutputParser()

    def _retrieve_tpb(self, figure_name: str) -> str:
        """从 TPB 知识库检索相关理论片段
        Retrieve relevant excerpts from the TPB knowledge base"""
        query = f"计划行为理论 行为信念 规范信念 控制信念 跨情境信念 核心动机 {figure_name}"
        try:
            results = self.retriever.retrieve(query, top_k=3)
            if isinstance(results, list):
                return "\n\n".join([doc.get("text", str(doc)) for doc in results])
            return str(results)
        except Exception as e:
            print(f"[BeliefDecomposer] RAG 检索失败（将跳过理论参考）/ RAG retrieval failed (skipping theory context): {e}")
            return ""

    def _condense_events(self, events: List[Dict]) -> str:
        """压缩事件列表以控制 token 用量 / Condense event list to control token usage"""
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
        return self.data_dir / "belief_analysis.json"

    def save_analysis(self, figure_name: str, result: Dict):
        file_path = self._get_file_path(figure_name)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[BeliefDecomposer] 已保存信念分析至 / Saved belief analysis to {file_path}")

    def load_analysis(self, figure_name: str) -> Optional[Dict]:
        file_path = self._get_file_path(figure_name)
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                result = json.load(f)
            print(f"[BeliefDecomposer] 已加载信念分析 / Loaded belief analysis from {file_path}")
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
        对全部事件做一次性整体分析，输出综合信念体系侧写。
        Performs a single comprehensive belief system analysis over all events.

        结果自动保存至 data/processed/{figure_name}/。
        Results are automatically saved to data/processed/{figure_name}/.
        """
        # 命中缓存则直接返回，跳过 LLM 调用
        # Return cached result immediately if available, skipping LLM call
        if not force_refresh:
            cached = self.load_analysis(figure_name)
            if cached is not None:
                return cached

        print(f"[BeliefDecomposer] 开始集体侧写 / Starting collective profiling: {figure_name}（共 {len(events)} 个事件 / {len(events)} events）...")
        events_json = self._condense_events(events)
        try:
            result = self.chain.invoke({
                "figure_name": figure_name,
                "event_count": len(events),
                "events_json": events_json,
            })
        except Exception as e:
            print(f"[BeliefDecomposer] 分析失败 / Analysis failed: {e}")
            result = {"error": str(e)}

        output = {"figure_name": figure_name, **result}
        self.save_analysis(figure_name, output)
        return output

    # ------------------------------------------------------------------ #
    #  单事件拆解（保留备用，不再用于批量流程）
    #  Single-event decomposition (kept as fallback, no longer used in batch flow)
    # ------------------------------------------------------------------ #

    def decompose_and_analyze(self, event: Dict, figure_name: str = "") -> Dict:
        """对单个事件进行决策单元拆解和信念分析（备用接口）
        Decompose a single event into decision units and analyze beliefs (fallback interface)"""
        summary = event.get("summary", "")
        if not summary:
            return {"event_id": event.get("event_id", "?"), "event_title": event.get("title", ""), "belief_units": []}
        try:
            decisions = self._decompose_chain.invoke({
                "figure_name": figure_name,
                "event_summary": summary,
            })
        except Exception as e:
            return {"event_id": event.get("event_id", "?"), "event_title": event.get("title", ""), "error": str(e)}
        if not isinstance(decisions, list):
            decisions = []
        return {
            "event_id": event.get("event_id", "?"),
            "event_title": event.get("title", ""),
            "belief_units": decisions,
        }
