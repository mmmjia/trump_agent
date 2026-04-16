# agents/agent_event_refiner.py
"""
Agent 2b: 事件精炼器（纯评估模式）
职责：评估事件覆盖度与质量，返回 sufficient 标志和 search_queries。
      不再自己调用 info_collector 或 extractor——补充循环由 LangGraph workflow 驱动。

evaluate() 返回：
  {
    "sufficient":      bool,
    "search_queries":  List[str],   # 时间空白 → 传回 collect_info 触发补充搜索
    "events":          List[Dict],  # needs_update 事件已由 LLM 直接增强（无需外部搜索）
  }
"""

import json
import re
import datetime
from typing import List, Dict, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from config.prompts import EVENT_VALIDATOR_SYSTEM, EVENT_VALIDATION_TASK_TEMPLATE

TEMPORAL_GAP_YEARS = 3


class EventRefinerAgent:
    def __init__(self, llm, extractor=None, info_collector=None):
        self.llm = llm
        self.extractor = extractor          # 保留备用，evaluate() 不使用
        self.info_collector = info_collector  # 保留备用，evaluate() 不使用

        validator_prompt = ChatPromptTemplate.from_messages([
            ("system", EVENT_VALIDATOR_SYSTEM),
            ("human", EVENT_VALIDATION_TASK_TEMPLATE),
        ])
        self.validator_chain = validator_prompt | self.llm | JsonOutputParser()

    # ------------------------------------------------------------------ #
    #  时间覆盖度工具
    # ------------------------------------------------------------------ #

    def _parse_year(self, time_period: str) -> Optional[int]:
        if not time_period:
            return None
        m = re.search(r'\b(1[89]\d{2}|20[0-2]\d)\b', str(time_period))
        return int(m.group(1)) if m else None

    def _detect_temporal_gaps(
        self, events: List[Dict], birth_year: Optional[int] = None
    ) -> List[Tuple[int, int]]:
        all_years = sorted({
            y for ev in events
            if (y := self._parse_year(ev.get("time_period", ""))) is not None
        })
        if not all_years:
            return []
        current_year = datetime.datetime.now().year

        # 只关心 birth_year 之后的时间段，忽略祖先/家族史
        floor = birth_year if birth_year else all_years[0]
        years = [y for y in all_years if y >= floor]
        if not years:
            return []

        gaps, prev = [], floor
        for y in years:
            if y - prev > TEMPORAL_GAP_YEARS:
                gaps.append((prev, y))
            prev = y
        if current_year - prev > TEMPORAL_GAP_YEARS:
            gaps.append((prev, current_year))
        return gaps

    def print_coverage(self, events: List[Dict]):
        years = sorted({
            y for ev in events
            if (y := self._parse_year(ev.get("time_period", ""))) is not None
        })
        cats: Dict[str, int] = {}
        for ev in events:
            c = ev.get("category", "unknown")
            cats[c] = cats.get(c, 0) + 1
        year_range = f"{years[0]}-{years[-1]}" if years else "?"
        print(f"[Refiner] 事件总数: {len(events)} | 年份跨度: {year_range}")
        print("[Refiner] 类别分布:", " | ".join(f"{k}:{v}" for k, v in sorted(cats.items())))

    # ------------------------------------------------------------------ #
    #  精准搜索词生成（时间空白 → LLM 生成具体关键词）
    # ------------------------------------------------------------------ #

    def _generate_gap_queries(
        self,
        figure_name: str,
        gaps: List[Tuple[int, int]],
        events: List[Dict],
    ) -> List[str]:
        """
        用 LLM 为每个时间空白生成 2-3 条具体的英文搜索关键词。
        参考相邻事件的标题/类别，让关键词足够精准，而非泛泛的 "biography events"。
        """
        # 收集相邻事件作为上下文提示
        year_to_events: Dict[int, List[str]] = {}
        for ev in events:
            y = self._parse_year(ev.get("time_period", ""))
            if y is not None:
                year_to_events.setdefault(y, []).append(
                    ev.get("title") or ev.get("category") or "event"
                )

        queries: List[str] = []
        for start_y, end_y in gaps:
            # 找紧邻空白前后的事件标题作为参考
            before = [
                t for y, titles in year_to_events.items()
                if start_y - 5 <= y <= start_y
                for t in titles
            ][-3:]
            after = [
                t for y, titles in year_to_events.items()
                if end_y <= y <= end_y + 5
                for t in titles
            ][:3]

            context_str = ""
            if before:
                context_str += f"Events just before the gap: {'; '.join(before)}. "
            if after:
                context_str += f"Events just after the gap: {'; '.join(after)}."

            prompt = (
                f"You are a biographical research assistant.\n"
                f"Person: {figure_name}\n"
                f"Time gap with no recorded events: {start_y}–{end_y}\n"
                f"{context_str}\n\n"
                f"Generate exactly 3 specific English Google search queries to find important "
                f"biographical events for {figure_name} during {start_y}–{end_y}. "
                f"Each query should be precise and targeted (NOT generic like 'biography life history'). "
                f"Output ONLY the 3 queries, one per line, no numbering or extra text."
            )
            try:
                response = self.llm.invoke(prompt)
                text = response.content if hasattr(response, "content") else str(response)
                for line in text.strip().splitlines():
                    line = line.strip().lstrip("•-*0123456789.) ")
                    if line:
                        queries.append(line)
            except Exception as e:
                print(f"[Refiner] 搜索词生成失败 ({start_y}-{end_y}): {e}")
                # 保底：比泛型稍具体的关键词
                queries.append(f"{figure_name} {start_y} {end_y} major events decisions")

        return queries

    # ------------------------------------------------------------------ #
    #  LLM 评估器
    # ------------------------------------------------------------------ #

    def _llm_evaluate(self, figure_name: str, events: List[Dict]) -> Dict:
        events_json = json.dumps(events, ensure_ascii=False, indent=2)
        try:
            result = self.validator_chain.invoke({
                "figure_name": figure_name,
                "events_json": events_json,
            })
            return result if isinstance(result, dict) else {}
        except Exception as e:
            print(f"[Refiner] LLM 评估失败: {e}")
            return {}

    # ------------------------------------------------------------------ #
    #  LLM 直接增强 summary（无需外部搜索）
    # ------------------------------------------------------------------ #

    def _enrich_summary_via_llm(self, event: Dict, figure_name: str) -> str:
        """让 LLM 直接补充事件 summary 中缺失的心理反应与应对策略描述"""
        prompt = (
            f"人物：{figure_name}\n"
            f"事件标题：{event.get('title', '')}\n"
            f"当前摘要：{event.get('summary', '')}\n\n"
            f"请在保留原有内容的基础上，补充 {figure_name} 对此事件的具体情感反应、"
            f"态度立场与应对策略描述。不超过500字，直接输出完整摘要，不要加前缀。"
        )
        try:
            response = self.llm.invoke(prompt)
            text = response.content if hasattr(response, "content") else str(response)
            return text.strip()
        except Exception as e:
            print(f"[Refiner] 摘要增强失败 (event_id={event.get('event_id')}): {e}")
            return ""

    # ------------------------------------------------------------------ #
    #  主评估入口（供 workflow 调用）
    # ------------------------------------------------------------------ #

    def evaluate(self, figure_name: str, events: List[Dict]) -> Dict:
        """
        纯评估，不做任何网络请求或文件 I/O。

        1. 本地时间空白检测 → 若有空白，返回 search_queries 让 workflow 循环回 collect_info
        2. LLM 质量评估 → needs_update 的事件直接用 LLM 增强 summary（无需外部搜索）
        3. 返回 sufficient 标志 + search_queries（仅时间空白触发，质量问题已本地处理）
        """
        # ── 1. 时间空白检测（快速，无 LLM）──
        # 自动推断出生年：找标题/摘要含 "birth" / "born" / "出生" 的最早事件年份
        birth_year: Optional[int] = None
        birth_keywords = ("birth", "born", "出生", "诞生")
        for ev in events:
            text = (ev.get("title", "") + " " + ev.get("summary", "")).lower()
            if any(kw in text for kw in birth_keywords):
                y = self._parse_year(ev.get("time_period", ""))
                if y and (birth_year is None or y < birth_year):
                    birth_year = y

        gaps = self._detect_temporal_gaps(events, birth_year=birth_year)
        if gaps:
            queries = self._generate_gap_queries(figure_name, gaps, events)
            print(f"[Refiner] 发现时间空白 {gaps}，生成 {len(queries)} 个精准搜索词")
            return {"sufficient": False, "search_queries": queries, "events": events}

        # ── 2. LLM 质量评估 ──
        validation = self._llm_evaluate(figure_name, events)
        sufficient = str(validation.get("sufficient", "no")).lower() == "yes"

        # ── 3. 直接增强缺少心理反应描述的事件（LLM，无外部搜索）──
        updated_events = list(events)
        needs_update = validation.get("needs_update", []) or []
        if needs_update:
            print(f"[Refiner] 增强 {len(needs_update)} 个事件的心理反应描述...")
        for item in needs_update:
            event_id = item.get("event_id")
            for ev in updated_events:
                if ev.get("event_id") == event_id:
                    enriched = self._enrich_summary_via_llm(ev, figure_name)
                    if enriched:
                        ev["summary"] = enriched
                    break

        # ── 4. 收集 LLM 认为仍有时间空白的搜索词 ──
        search_queries = [
            gap["search_query"]
            for gap in (validation.get("temporal_gaps") or [])
            if gap.get("search_query")
        ]

        return {
            "sufficient": sufficient,
            "search_queries": search_queries,
            "events": updated_events,
        }
