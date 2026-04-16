# agents/orchestrator.py
"""
协调器：流程编排与任务分发

流程图：
                            ┌─────────────────────────────────┐
                            │  (insufficient & round < MAX)   │
                            ▼                                 │
  collect_info ──► extract_events ──► refine_events ──────────┘
                                           │
                                 (sufficient or MAX rounds)
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    ▼                      ▼                      ▼
             macro_analysis       social_cognitive       belief_decomposition
                    │                      │                      │
                    └──────────────────────┴──────────────────────┘
                                           │ (fan-in: 全部完成后)
                                           ▼
                                    predict_behavior
                                           │
                                          END

关键设计：
  - refine_events 是纯评估节点，不做任何网络请求
  - 若评估不足 → 把 search_queries 写入 state → 路由回 collect_info（有针对性地补充）
  - collect_info 区分"初次采集"和"补充采集"
  - extract_events 区分"初次提取"和"增量合并"
  - 三路分析并行执行（LangGraph fan-out），全部完成后汇聚到 predict_behavior
"""

from typing import Dict, Any, List, Optional, TypedDict

from langgraph.graph import StateGraph, END


class PipelineState(TypedDict, total=False):
    """
    TypedDict 状态：每个 key 是独立 channel，支持并行节点同时写入不同 key。
    必须用 TypedDict 而非 dict，否则并行节点写 __root__ 时会 InvalidUpdateError。
    """
    figure_name:              str
    biography_text:           Optional[str]
    new_scenario:             str
    force_refresh:            bool
    max_chunks:               int
    supplemental_text:        str
    events:                   Optional[list]
    refine_round:             int
    refine_sufficient:        bool
    refine_search_queries:    list
    macro_analysis:           Optional[dict]
    social_cognitive_analysis: Optional[dict]
    beliefs:                  Optional[dict]
    prediction:               Optional[dict]

import json
from pathlib import Path

from agents.agent_info_collector import InfoCollectorAgent
from agents.agent_event_extractor import EventExtractorAgent
from agents.agent_event_refiner import EventRefinerAgent
from agents.agent_macro_analyst import MacroAnalystAgent
from agents.agent_social_cognitive import SocialCognitiveAgent
from agents.agent_belief_decomposer import BeliefDecomposerAgent
from agents.agent_behavior_predictor import BehaviorPredictorAgent
from config.settings import settings

# refine_events 评估最多循环几次后强制进入分析阶段
MAX_REFINE_ROUNDS = 3

# 缓存的四个分析文件（behavior_prediction 依赖 scenario，不算强制缓存）
ANALYSIS_CACHE_FILES = ("events.json", "macro_analysis.json", "social_cognitive.json", "belief_analysis.json")


class OrchestratorAgent:
    def __init__(self, llm, trait_retriever, social_cognitive_retriever, tpb_retriever):
        self.info_collector     = InfoCollectorAgent()
        self.event_extractor    = EventExtractorAgent(llm)
        self.event_refiner      = EventRefinerAgent(llm)
        self.macro_analyst      = MacroAnalystAgent(llm, trait_retriever)
        self.social_cognitive   = SocialCognitiveAgent(llm, social_cognitive_retriever)
        self.belief_decomposer  = BeliefDecomposerAgent(llm, tpb_retriever)
        self.behavior_predictor = BehaviorPredictorAgent(llm)
        self.graph = self._build_graph()

    # ------------------------------------------------------------------ #
    #  图构建
    # ------------------------------------------------------------------ #

    def _build_graph(self):
        workflow = StateGraph(PipelineState)

        # 节点注册
        workflow.add_node("collect_info",        self._step_collect)
        workflow.add_node("extract_events",       self._step_extract)
        workflow.add_node("refine_events",        self._step_refine)
        workflow.add_node("macro_analysis",       self._step_macro)
        workflow.add_node("social_cognitive",     self._step_social_cognitive)
        workflow.add_node("belief_decomposition", self._step_belief)
        workflow.add_node("predict_behavior",     self._step_predict)

        # 固定边：采集 → 提取 → 评估
        workflow.set_entry_point("collect_info")
        workflow.add_edge("collect_info",   "extract_events")
        workflow.add_edge("extract_events", "refine_events")

        # 条件边：refine_events → 循环 or 并行分析
        workflow.add_conditional_edges("refine_events", self._route_after_refine)

        # 并行分析 → 汇聚到预测（fan-in：三路全部完成后才触发）
        workflow.add_edge("macro_analysis",       "predict_behavior")
        workflow.add_edge("social_cognitive",     "predict_behavior")
        workflow.add_edge("belief_decomposition", "predict_behavior")
        workflow.add_edge("predict_behavior",     END)

        return workflow.compile()

    def _route_after_refine(self, state: Dict):
        """
        路由函数：
          - 返回字符串 "collect_info"  → 单路循环回采集
          - 返回列表   [三个分析节点]  → 并行 fan-out
        """
        round_num = state.get("refine_round", 0)
        sufficient = state.get("refine_sufficient", False)
        has_queries = bool(state.get("refine_search_queries"))

        if sufficient or round_num >= MAX_REFINE_ROUNDS or not has_queries:
            if round_num >= MAX_REFINE_ROUNDS and not sufficient:
                print(f"[Refiner] 已达最大循环轮次 {MAX_REFINE_ROUNDS}，强制进入并行分析")
            else:
                print(f"[Refiner] 评估通过（第 {round_num} 轮），进入并行分析阶段")
            return ["macro_analysis", "social_cognitive", "belief_decomposition"]

        print(f"[Refiner] 第 {round_num} 轮未通过，返回信息采集（{len(state['refine_search_queries'])} 个目标查询）")
        return "collect_info"

    # ------------------------------------------------------------------ #
    #  步骤：采集（初次 or 补充）
    # ------------------------------------------------------------------ #

    def _step_collect(self, state: Dict) -> Dict:
        queries: List[str] = state.get("refine_search_queries", [])

        if queries:
            print(f"[Collector] 补充采集，共 {len(queries)} 个目标查询...")
            texts = []
            for q in queries:
                text = self.info_collector.collect_supplemental(state["figure_name"], q)
                if text:
                    texts.append(text)
            return {
                "supplemental_text":     "\n\n---\n\n".join(texts) if texts else "",
                "refine_search_queries": [],   # 消费后清空
            }
        else:
            print("[Collector] 初次采集人物信息...")
            updates: Dict = {"supplemental_text": ""}
            if not state.get("biography_text"):
                updates["biography_text"] = self.info_collector.collect(state["figure_name"])
            return updates

    # ------------------------------------------------------------------ #
    #  步骤：提取（初次 or 增量合并）
    # ------------------------------------------------------------------ #

    def _step_extract(self, state: Dict) -> Dict:
        figure_name       = state["figure_name"]
        supplemental_text = state.get("supplemental_text", "")
        max_chunks        = state.get("max_chunks", 50)

        if supplemental_text:
            print("[Extractor] 增量提取补充事件...")
            new_events = self.event_extractor.extract_events_from_text(
                supplemental_text, num_events=20, max_chunks=max_chunks
            )
            existing = state.get("events") or []
            return {"events": self.event_extractor.merge_new_events(figure_name, new_events, existing)}
        else:
            if not state.get("force_refresh") and self.event_extractor.load_existing_events(figure_name):
                events = self.event_extractor.load_existing_events(figure_name)
                print(f"[Extractor] 使用缓存事件（{figure_name}）")
            else:
                print("[Extractor] 初次提取事件...")
                events = self.event_extractor.extract_events(
                    figure_name=figure_name,
                    biography_text=state.get("biography_text", ""),
                    num_events=50,
                    max_chunks=max_chunks,
                )
            return {"events": events}

    # ------------------------------------------------------------------ #
    #  步骤：评估（纯评估，不做网络请求）
    # ------------------------------------------------------------------ #

    def _step_refine(self, state: Dict) -> Dict:
        round_num = state.get("refine_round", 0) + 1
        print(f"[Refiner] 第 {round_num} 轮评估（共 {len(state.get('events', []))} 个事件）...")

        result = self.event_refiner.evaluate(
            figure_name=state["figure_name"],
            events=state.get("events", []),
        )

        if result["sufficient"]:
            self.event_refiner.print_coverage(result["events"])

        return {
            "events":                result["events"],
            "refine_sufficient":     result["sufficient"],
            "refine_search_queries": result.get("search_queries", []),
            "refine_round":          round_num,
        }

    # ------------------------------------------------------------------ #
    #  步骤：并行分析（三路独立，互不依赖）
    #  重要：只返回本节点写入的 key，不返回整个 state
    # ------------------------------------------------------------------ #

    def _step_macro(self, state: Dict) -> Dict:
        print("[MacroAnalyst] 开始宏观分析...")
        return {
            "macro_analysis": self.macro_analyst.analyze_events(
                events=state["events"],
                figure_name=state["figure_name"],
            )
        }

    def _step_social_cognitive(self, state: Dict) -> Dict:
        print("[SocialCognitive] 开始社会认知分析...")
        return {
            "social_cognitive_analysis": self.social_cognitive.analyze_events(
                events=state["events"],
                figure_name=state["figure_name"],
            )
        }

    def _step_belief(self, state: Dict) -> Dict:
        print("[BeliefDecomposer] 开始信念侧写...")
        return {
            "beliefs": self.belief_decomposer.analyze_events(
                events=state["events"],
                figure_name=state["figure_name"],
            )
        }

    # ------------------------------------------------------------------ #
    #  步骤：预测（汇聚三路结果）
    # ------------------------------------------------------------------ #

    def _step_predict(self, state: Dict) -> Dict:
        print("[Predictor] 综合三路分析，进行行为倾向预测...")
        return {
            "prediction": self.behavior_predictor.predict(
                new_scenario=state["new_scenario"],
                macro_analysis=state["macro_analysis"],
                belief_patterns=[state["beliefs"]],
                social_cognitive_analysis=state.get("social_cognitive_analysis"),
                figure_name=state["figure_name"],
            )
        }

    # ------------------------------------------------------------------ #
    #  人物文件夹 / 注册表工具
    # ------------------------------------------------------------------ #

    @staticmethod
    def _safe_name(figure_name: str) -> str:
        return figure_name.lower().replace(" ", "_")

    def _figure_dir(self, figure_name: str) -> Path:
        """返回该人物的专属数据目录，自动创建。"""
        d = settings.PROCESSED_DATA_DIR / self._safe_name(figure_name)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _registry_path(self) -> Path:
        return settings.PROCESSED_DATA_DIR / "figures_registry.json"

    def _register_figure(self, figure_name: str):
        """将人物名加入注册表（去重）。"""
        path = self._registry_path()
        registry: List[str] = []
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                registry = json.load(f)
        if figure_name not in registry:
            registry.append(figure_name)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(registry, f, indent=2, ensure_ascii=False)
            print(f"[Orchestrator] 已注册新人物：{figure_name}")

    def _is_analysis_cached(self, figure_dir: Path) -> bool:
        """四个核心分析文件全部存在则视为缓存命中。"""
        return all((figure_dir / fname).exists() for fname in ANALYSIS_CACHE_FILES)

    def _load_cached_analyses(self, figure_dir: Path) -> Dict:
        """从人物目录加载所有缓存的分析结果。"""
        result = {}
        for fname in ANALYSIS_CACHE_FILES:
            key = fname.replace(".json", "")   # "events", "macro_analysis", ...
            with open(figure_dir / fname, "r", encoding="utf-8") as f:
                result[key] = json.load(f)
        return result

    def _set_agents_data_dir(self, figure_dir: Path):
        """将所有 agent 的 data_dir 指向该人物目录。"""
        for agent in (
            self.event_extractor,
            self.macro_analyst,
            self.social_cognitive,
            self.belief_decomposer,
            self.behavior_predictor,
        ):
            agent.data_dir = figure_dir
            figure_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  主入口
    # ------------------------------------------------------------------ #

    def run(
        self,
        figure_name: str,
        new_scenario: str,
        biography_text: Optional[str] = None,
        force_refresh: bool = False,
        max_chunks: int = 50,
    ) -> Dict[str, Any]:
        figure_dir = self._figure_dir(figure_name)
        self._set_agents_data_dir(figure_dir)

        # ── 缓存命中：跳过采集和分析，仅重跑 behavior_predictor ──
        if not force_refresh and self._is_analysis_cached(figure_dir):
            print(f"[Orchestrator] 命中缓存（{figure_name}），直接加载已有分析结果...")
            cached = self._load_cached_analyses(figure_dir)
            print("[Predictor] 使用缓存结果，综合三路分析，进行行为倾向预测...")
            prediction = self.behavior_predictor.predict(
                new_scenario=new_scenario,
                macro_analysis=cached["macro_analysis"],
                belief_patterns=[cached["belief_analysis"]],
                social_cognitive_analysis=cached["social_cognitive"],
                figure_name=figure_name,
            )
            return {
                "figure_name":              figure_name,
                "events":                   cached["events"],
                "macro_analysis":           cached["macro_analysis"],
                "social_cognitive_analysis": cached["social_cognitive"],
                "beliefs":                  cached["belief_analysis"],
                "prediction":               prediction,
            }

        # ── 缓存未命中：运行完整 workflow ──
        initial_state = {
            "figure_name":               figure_name,
            "biography_text":            biography_text,
            "new_scenario":              new_scenario,
            "force_refresh":             force_refresh,
            "max_chunks":                max_chunks,
            "supplemental_text":         "",
            "events":                    None,
            "refine_round":              0,
            "refine_sufficient":         False,
            "refine_search_queries":     [],
            "macro_analysis":            None,
            "social_cognitive_analysis": None,
            "beliefs":                   None,
            "prediction":                None,
        }
        result = self.graph.invoke(initial_state)
        self._register_figure(figure_name)
        return result
