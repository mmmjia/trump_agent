# Psychobiography AI Agent System
**心理传记学 AI 多智能体系统**

A multi-agent AI system that constructs psychological profiles of public figures and predicts their behavioral tendencies in hypothetical scenarios, powered by LangGraph, RAG, and three classical psychological theories.

一个基于 LangGraph、RAG 和三大经典心理学理论的 AI 多智能体系统，用于构建公众人物的心理侧写，并预测其在假设情境下的行为倾向。

---

## Overview / 系统概述

Given a public figure's name and a hypothetical scenario, the system:

1. Automatically collects biographical text (Wikipedia + web search)
2. Extracts structured psychological events from the biography
3. Evaluates event coverage and fills temporal gaps
4. Runs three parallel psychological analyses grounded in academic theory
5. Synthesizes all analyses into a behavior prediction for the given scenario

给定人物姓名和假设情境，系统将：

1. 自动采集传记文本（Wikipedia + 网络搜索）
2. 从传记中提取结构化心理行为事件
3. 评估事件覆盖度并填补时间空白
4. 并行执行三路心理学分析
5. 综合全部分析，生成对假设情境的行为倾向预测

---

## Pipeline Architecture / 流水线架构

```
collect_info
     │
     ▼
extract_events
     │
     ▼
refine_events ──── insufficient ──── collect_info (loop / 循环)
     │
  sufficient
     │
     ├──────────────────┬──────────────────┐
     ▼                  ▼                  ▼
macro_analysis   social_cognitive   belief_decomposition
     │                  │                  │
     └──────────────────┴──────────────────┘
                        │
                        ▼
                predict_behavior
                        │
                       END
```

### Agents / 智能体

| Agent | Role / 职责 | Theory / 理论基础 |
|-------|------------|-----------------|
| **InfoCollector** | Scrapes Wikipedia & DuckDuckGo for biography text / 采集传记文本 | — |
| **EventExtractor** | Chunks text, extracts psychological events via LLM in parallel / 并行提取心理行为事件 | — |
| **EventRefiner** | Evaluates temporal coverage, generates targeted gap-fill queries / 评估时间覆盖度，生成精准补充搜索词 | — |
| **MacroAnalyst** | Big Five personality + Bourdieu social capital / 大五人格 + 布迪厄社会资本 | Trait Psychology |
| **SocialCognitiveAgent** | Self-efficacy, outcome expectancy, observational learning / 自我效能、结果预期、观察学习 | Bandura (1986) |
| **BeliefDecomposer** | Behavioral, normative, and control beliefs / 行为信念、规范信念、控制信念 | Theory of Planned Behavior (TPB) |
| **BehaviorPredictor** | Synthesizes all analyses → behavioral prediction / 综合三路分析 → 行为预测 | — |

---

## Quick Start / 快速开始

### 1. Install Dependencies / 安装依赖

```bash
pip install langchain langchain-openai langgraph sentence-transformers faiss-cpu \
            python-dotenv requests beautifulsoup4 ddgs
```

### 2. Configure Environment / 配置环境变量

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

```env
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat

EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
```

### 3. Build Psychology Knowledge Bases / 构建心理学知识库（可选，增强分析质量）

Place PDF/TXT psychology books in the following directories (relative to this project):

```
../knowledge/psychology_books/
├── trait_psychology/        ← Big Five + Bourdieu texts
├── social_cognitive/        ← Bandura texts
└── Behaviorism_Learning_Theory/  ← TPB texts
```

Then build the FAISS vector stores:

```bash
python scripts/build_psychology_vectorstores.py
```

### 4. Run Analysis / 运行分析

```bash
# Basic usage / 基本用法
python main.py --figure "Donald Trump" \
               --scenario "If this person faced an AI regulation decision, what would they do?"

# With a local biography file / 使用本地传记文件
python main.py --figure "Steve Jobs" \
               --biography path/to/biography.txt \
               --scenario "How would this person respond to a major product failure?"

# Force refresh cached results / 强制刷新缓存
python main.py --figure "Elon Musk" --scenario "..." --refresh
```

---

## Caching / 缓存机制

The system caches results per figure to avoid redundant LLM calls:

- On first run: all 6 agents execute, results are saved to `data/processed/{figure_name}/`
- On subsequent runs with the **same figure**: events + 3 analyses are loaded from cache; only `BehaviorPredictor` re-runs with the new scenario
- Pass `--refresh` (or `force_refresh=True`) to force a full re-analysis

系统对每个人物缓存分析结果，避免重复 LLM 调用：

- 首次运行：所有 6 个 Agent 执行，结果保存至 `data/processed/{figure_name}/`
- 再次查询**同一人物**：直接加载缓存的事件和三路分析，仅重跑 BehaviorPredictor（因情境不同）
- 传入 `--refresh` 或 `force_refresh=True` 可强制重新分析

---

## Output Format / 输出格式

All analysis results are saved as JSON. Key output files:

| File | Contents |
|------|----------|
| `events.json` | List of extracted biographical events with fields: `event_id`, `time_period`, `category`, `title`, `summary`, `impact_level`, `outcome`, `key_actors` |
| `macro_analysis.json` | Big Five scores, Bourdieu capital analysis, behavioral patterns, psychological portrait |
| `social_cognitive.json` | Self-efficacy profile, outcome expectancy, observational learning history, triadic dynamics |
| `belief_analysis.json` | Core behavioral/normative/control beliefs, belief evolution over time, belief portrait |
| `behavior_prediction.json` | Scenario, prediction narrative, core beliefs used in prediction |

---

## Configuration Reference / 配置参考

| Variable | Default | Description |
|----------|---------|-------------|
| `DEEPSEEK_API_KEY` | — | DeepSeek API key (required) |
| `DEEPSEEK_MODEL` | `deepseek-chat` | Model name |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com/v1` | API base URL |
| `EMBEDDING_MODEL` | `BAAI/bge-large-zh-v1.5` | Sentence-transformer model for RAG (1024-dim, Chinese+English) |
| `KNOWLEDGE_DIR` | `../knowledge/vector_stores` | Path to FAISS vector stores |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

---



## Troubleshooting / 常见问题

**Google search always returns empty**
Google blocks headless requests. The system uses DuckDuckGo (`ddgs` package) as the primary search engine. Install with:
```bash
pip install ddgs
```

**Embedding dimension mismatch**
Old Chroma vector stores (384-dim) conflict with the new FAISS stores (1024-dim). Rebuild with:
```bash
python scripts/build_psychology_vectorstores.py
```

**`InvalidUpdateError` from LangGraph parallel branches**
Ensure all step methods return **only the keys they modify**, not the full state dict. See `orchestrator.py` step methods for the correct pattern.

**Too many extraction chunks**
Control the number of text chunks processed per run:
```python
orchestrator.run(..., max_chunks=30)  # default is 50
```
