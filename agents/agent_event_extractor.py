# agents/agent_event_extractor.py
"""
Agent 2a: 事件提取器（纯提取）
职责：将传记文本切分为块，调用 LLM 逐块提取心理行为学事件，合并去重后返回事件列表。
不包含验证或补全逻辑——这部分由 Agent 2b（EventRefinerAgent）负责。
"""

import json
import re
from typing import List, Dict, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from config.settings import settings
from config.prompts import EVENT_EXTRACTOR_SYSTEM, EVENT_EXTRACTION_TASK_TEMPLATE

# 长文本分块参数
TEXT_CHUNK_SIZE = 12000     # 每块字符数
TEXT_CHUNK_OVERLAP = 1500   # 块间重叠，避免事件被切断


class EventExtractorAgent:
    """
    纯事件提取器：给定文本 → 输出去重后的事件列表。
    不调用任何外部资源，无网络 I/O，无验证循环。
    """

    def __init__(self, llm, data_dir: Optional[str] = None):
        self.llm = llm
        self.data_dir = Path(data_dir) if data_dir else settings.PROCESSED_DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chain = self._build_chain()

    def _build_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", EVENT_EXTRACTOR_SYSTEM),
            ("human", EVENT_EXTRACTION_TASK_TEMPLATE),
        ])
        return prompt | self.llm | JsonOutputParser()

    # ------------------------------------------------------------------ #
    #  缓存 I/O
    # ------------------------------------------------------------------ #

    def _get_events_file_path(self, figure_name: str) -> Path:
        return self.data_dir / "events.json"

    def load_existing_events(self, figure_name: str) -> List[Dict]:
        file_path = self._get_events_file_path(figure_name)
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                events = json.load(f)
            print(f"[Extractor] 已加载 {len(events)} 个已有事件 from {file_path}")
            return events
        return []

    def save_events(self, figure_name: str, events: List[Dict]):
        file_path = self._get_events_file_path(figure_name)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(events, f, indent=2, ensure_ascii=False)
        print(f"[Extractor] 已保存 {len(events)} 个事件至 {file_path}")

    # ------------------------------------------------------------------ #
    #  文本分块
    # ------------------------------------------------------------------ #

    def _chunk_text(self, text: str) -> List[str]:
        """将长文本切分为带重叠的块"""
        if len(text) <= TEXT_CHUNK_SIZE:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = start + TEXT_CHUNK_SIZE
            chunks.append(text[start:end])
            start += TEXT_CHUNK_SIZE - TEXT_CHUNK_OVERLAP
        return chunks

    # ------------------------------------------------------------------ #
    #  去重
    # ------------------------------------------------------------------ #

    def _text_similarity(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        sa, sb = set(a.split()), set(b.split())
        return len(sa & sb) / len(sa | sb) if sa and sb else 0.0

    def is_duplicate(self, new_event: Dict, existing_events: List[Dict]) -> bool:
        if not isinstance(new_event, dict):
            return False
        new_title   = new_event.get("title", "").lower()
        new_summary = new_event.get("summary", "")[:500].lower()
        for ev in existing_events:
            if ev.get("title", "").lower() == new_title:
                return True
            if self._text_similarity(new_summary, ev.get("summary", "")[:500].lower()) > 0.8:
                return True
        return False

    # ------------------------------------------------------------------ #
    #  核心提取
    # ------------------------------------------------------------------ #

    def _extract_chunk(self, chunk: str, idx: int, total: int, num_events: int) -> List[Dict]:
        """提取单个文本块中的事件（供并行调用）。"""
        print(f"[Extractor] 提取块 {idx+1}/{total}（{len(chunk)} 字符）")
        try:
            result = self.chain.invoke({
                "num_events": num_events,
                "figure_intro": "",
                "biography_text": chunk,
            })
            if isinstance(result, str):
                clean = re.sub(r'^```json\s*', '', result)
                clean = re.sub(r'\s*```$', '', clean)
                result = json.loads(clean)
            if not isinstance(result, list):
                print(f"[Extractor] 警告: 块 {idx+1} LLM 返回 {type(result)}，跳过")
                return []
            return [ev for ev in result if isinstance(ev, dict)]
        except Exception as e:
            print(f"[Extractor] 块 {idx+1} 提取失败: {e}")
            return []

    def extract_events_from_text(
        self, text: str, num_events: int = 30, max_chunks: int = 50, max_workers: int = 8
    ) -> List[Dict]:
        """
        从文本中提取事件。自动分块，并行调用 LLM，去重后返回。
        max_chunks:  最多处理的文本块数，超出时均匀采样以覆盖全文。
        max_workers: 并行 LLM 调用数（建议 5-10，避免触发 API 速率限制）。
        """
        if not text.strip():
            return []
        chunks = self._chunk_text(text)

        # 均匀采样，确保覆盖全文而非只截断前 N 块
        if len(chunks) > max_chunks:
            total = len(chunks)
            step = total / max_chunks
            chunks = [chunks[int(i * step)] for i in range(max_chunks)]
            print(f"[Extractor] 文本块总数 {total} → 均匀采样 {max_chunks} 块")

        total = len(chunks)
        print(f"[Extractor] 并行提取 {total} 个文本块（max_workers={max_workers}）...")

        # 并行执行 LLM 调用，按原始顺序收集结果
        chunk_results: Dict[int, List[Dict]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self._extract_chunk, chunk, i, total, num_events): i
                for i, chunk in enumerate(chunks)
            }
            for future in as_completed(futures):
                i = futures[future]
                chunk_results[i] = future.result()

        # 按块顺序合并去重（保持时间顺序）
        all_events: List[Dict] = []
        for i in range(total):
            for ev in chunk_results.get(i, []):
                if not self.is_duplicate(ev, all_events):
                    all_events.append(ev)

        return all_events

    # ------------------------------------------------------------------ #
    #  主入口（提取 + 编号 + 保存）
    # ------------------------------------------------------------------ #

    def extract_events(
        self,
        figure_name: str,
        biography_text: str,
        num_events: int = 50,
        max_chunks: int = 50,
    ) -> List[Dict]:
        """
        从给定传记文本中提取事件，分配 event_id，保存并返回。
        max_chunks: 最多处理的文本块数（默认 50）。
        不包含验证或补全逻辑。
        """
        events = self.extract_events_from_text(biography_text, num_events, max_chunks)
        for i, ev in enumerate(events, 1):
            ev["event_id"] = i
        self.save_events(figure_name, events)
        print(f"[Extractor] 提取完成，共 {len(events)} 个事件")
        return events

    # ------------------------------------------------------------------ #
    #  增量更新（外部调用）
    # ------------------------------------------------------------------ #

    def merge_new_events(
        self, figure_name: str, new_events: List[Dict], existing: List[Dict]
    ) -> List[Dict]:
        """
        将新事件合并入已有列表（去重 + 重新编号 + 保存）。
        供 EventRefinerAgent 在补充搜索后调用。
        """
        added = []
        max_id = max((e.get("event_id", 0) for e in existing), default=0)
        for ev in new_events:
            if not self.is_duplicate(ev, existing + added):
                max_id += 1
                ev["event_id"] = max_id
                added.append(ev)
        if added:
            merged = sorted(existing + added, key=lambda x: x.get("event_id", 0))
            self.save_events(figure_name, merged)
            print(f"[Extractor] 合并新增 {len(added)} 个事件，总计 {len(merged)} 个")
            return merged
        print("[Extractor] 无新增唯一事件")
        return existing
