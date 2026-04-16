# agents/agent_behavior_predictor.py
"""
Agent 6: 行为倾向预测器
Agent 6: Behavioral Tendency Predictor

职责：综合宏观分析、社会认知分析和信念拆解，对新情境下人物的行为倾向进行预测
Responsibility: Synthesize macro analysis, social cognitive analysis, and belief decomposition
                to predict a figure's behavioral tendency in a new hypothetical scenario.

输入：macro_analysis、social_cognitive_analysis、belief_patterns、new_scenario
Input: macro_analysis, social_cognitive_analysis, belief_patterns, new_scenario

输出：预测报告 + 所用核心信念
Output: Prediction report + core beliefs used

结果自动保存至 data/processed/{figure_name}/behavior_prediction.json
Results are automatically saved to data/processed/{figure_name}/behavior_prediction.json
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config.prompts import (
    BEHAVIOR_PREDICTOR_SYSTEM,
    BEHAVIOR_PREDICTION_TASK_TEMPLATE,
    CORE_BELIEF_SYNTHESIS_PROMPT,
)
from config.settings import settings


class BehaviorPredictorAgent:
    """
    综合宏观分析 + 社会认知分析 + 信念拆解，对新情境下人物的行为倾向进行预测。
    Synthesizes macro analysis, social cognitive analysis, and belief decomposition
    to predict a figure's behavioral tendency in a novel scenario.

    核心假设：人格特质和核心信念具有跨情境稳定性。
    Core assumption: Personality traits and core beliefs are stable across contexts.
    """

    def __init__(self, llm, data_dir: Optional[str] = None):
        self.llm = llm
        self.data_dir = Path(data_dir) if data_dir else settings.PROCESSED_DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.predict_chain = self._build_predict_chain()
        self.synthesis_chain = self._build_synthesis_chain()

    def _build_predict_chain(self):
        """构建行为预测链，使用 prompts.py 中的规范模板
        Build the behavior prediction chain using the standardized template in prompts.py"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", BEHAVIOR_PREDICTOR_SYSTEM),
            ("human", BEHAVIOR_PREDICTION_TASK_TEMPLATE),
        ])
        return prompt | self.llm | StrOutputParser()

    def _build_synthesis_chain(self):
        """构建核心信念归纳链，使用 prompts.py 中的规范模板
        Build the core belief synthesis chain using the standardized template in prompts.py"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一位心理学分析专家，擅长从多个案例中归纳稳定的核心信念模式。请严格按 JSON 格式输出。"),
            ("human", CORE_BELIEF_SYNTHESIS_PROMPT),
        ])
        return prompt | self.llm | StrOutputParser()

    # ------------------------------------------------------------------ #
    #  缓存 I/O / Cache I/O
    # ------------------------------------------------------------------ #

    def _get_file_path(self, figure_name: str) -> Path:
        # 文件名不含人名前缀，因为文件夹已经是人名目录
        # Filename has no person-name prefix; the parent folder already identifies the figure
        return self.data_dir / "behavior_prediction.json"

    def save_prediction(self, figure_name: str, result: Dict):
        file_path = self._get_file_path(figure_name)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[BehaviorPredictor] 已保存行为预测至 / Saved behavior prediction to {file_path}")

    def load_prediction(self, figure_name: str) -> Optional[Dict]:
        file_path = self._get_file_path(figure_name)
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                result = json.load(f)
            print(f"[BehaviorPredictor] 已加载行为预测 / Loaded behavior prediction from {file_path}")
            return result
        return None

    # ------------------------------------------------------------------ #
    #  主预测入口 / Main Prediction Entry Point
    # ------------------------------------------------------------------ #

    def predict(
        self,
        new_scenario: str,
        macro_analysis: Dict,
        belief_patterns: List[Dict],
        social_cognitive_analysis: Optional[Dict] = None,
        figure_name: Optional[str] = None,
    ) -> Dict:
        """
        预测人物在给定新情境下的行为倾向，结果自动保存至 data/processed/（需提供 figure_name）。
        Predict the figure's behavioral tendency in the given scenario.
        Results are automatically saved to data/processed/ when figure_name is provided.

        Args:
            new_scenario:               假设情境描述 / Hypothetical scenario description
            macro_analysis:             宏观分析结果（大五人格 + 布迪厄）/ Macro analysis (Big Five + Bourdieu)
            belief_patterns:            各事件的信念拆解列表 / Belief decomposition list from events
            social_cognitive_analysis:  社会认知分析结果（自我效能、结果预期等），可选
                                        Social cognitive analysis (self-efficacy, outcome expectancy, etc.), optional
            figure_name:                人物姓名，用于保存文件；不提供则不保存
                                        Figure name for saving output; omit to skip saving
        Returns:
            包含 scenario、prediction、core_beliefs_used 的字典
            Dict containing scenario, prediction text, and core_beliefs_used
        """
        # 先归纳跨事件稳定的核心信念
        # First synthesize stable core beliefs that persist across events
        core_beliefs = self._synthesize_core_beliefs(belief_patterns)

        prediction_text = self.predict_chain.invoke({
            "macro_analysis":            json.dumps(macro_analysis, ensure_ascii=False, indent=2),
            "social_cognitive_analysis": json.dumps(social_cognitive_analysis or {}, ensure_ascii=False, indent=2),
            "core_beliefs":              json.dumps(core_beliefs, ensure_ascii=False, indent=2),
            "new_scenario":              new_scenario,
        })

        result = {
            "figure_name":      figure_name or "",
            "scenario":         new_scenario,
            "prediction":       prediction_text,
            "core_beliefs_used": core_beliefs,
        }

        if figure_name:
            self.save_prediction(figure_name, result)

        return result

    def _synthesize_core_beliefs(self, belief_patterns: List[Dict]) -> Dict:
        """从多个信念分析中归纳跨情境稳定的核心模式
        Synthesize stable cross-context core belief patterns from multiple belief analyses"""
        response_text = self.synthesis_chain.invoke({
            "belief_patterns": json.dumps(belief_patterns, ensure_ascii=False, indent=2),
        })
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            # JSON 解析失败时保留原始文本，不抛出异常
            # Preserve raw text on JSON parse failure instead of raising
            return {"error": "Failed to parse JSON", "raw_response": response_text}
