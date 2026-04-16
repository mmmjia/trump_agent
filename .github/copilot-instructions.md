# AI Coding Agent Instructions for Psychobiography AI Agent Project

## Project Overview
This is a multi-agent AI system for psychobiographical analysis, written in Python using LangChain and LangGraph. The system analyzes historical figures through four sequential agents: event extraction, macro psychological analysis (Big Five + Bourdieu), belief decomposition, and behavior prediction. It incorporates RAG (Retrieval-Augmented Generation) for psychology theory knowledge.

## Architecture & Data Flow
- **Entry Point**: `main.py` initializes LLMs and orchestrator, runs analysis pipeline
- **Orchestrator**: `agents/orchestrator.py` uses LangGraph StateGraph for workflow management
- **Agent Sequence**: 
  1. `agent_event_extractor.py`: Extracts 10 key life events from biography text
  2. `agent_macro_analyst.py`: Analyzes personality traits (OCEAN) and social capital (Bourdieu)
  3. `agent_belief_decomposer.py`: Decomposes beliefs into behavioral, normative, and control components
  4. `agent_behavior_predictor.py`: Predicts behavior in hypothetical scenarios
- **Data Flow**: Biography text → JSON events → parallel macro/belief analysis → integrated prediction
- **RAG Integration**: `rag/vector_retriever.py` retrieves from FAISS vector stores of psychology texts

## Key Conventions & Patterns
- **Language**: All prompts and comments in Chinese; maintain this for consistency
- **LLM Setup**: Use separate ChatOpenAI instances per agent (DeepSeek API preferred)
- **Structured Output**: All agents use JsonOutputParser for consistent JSON responses
- **Prompt Management**: System prompts in `config/prompts.py`, task templates use f-string formatting
- **Data Persistence**: Events and analyses saved as JSON in `../data/processed/{figure}_events.json`
- **Vector Stores**: Built from `../knowledge/source_docs/` using `scripts/build_psychology_vectorstores.py`
- **Configuration**: Environment variables loaded via `config/settings.py` (DeepSeek API keys required)

## Critical Developer Workflows
- **Setup**: `pip install -r requirements.txt` (create if missing); copy `.env.example` to `.env` with API keys
- **Build Knowledge Base**: Run `python scripts/build_psychology_vectorstores.py` to create FAISS indices from psychology PDFs
- **Test Agents**: Use `try_extract.py` or similar scripts for individual agent testing
- **Run Full Pipeline**: `python main.py` analyzes configured figure (currently Steve Jobs example)
- **Debug**: Check `../data/processed/` for intermediate JSON outputs; agents print progress to console

## Integration Points
- **External APIs**: DeepSeek (primary) or OpenAI for LLM calls
- **Vector Search**: FAISS with sentence-transformers (BAAI/bge-large-zh-v1.5 for Chinese)
- **Document Loading**: Supports PDF/txt/md via langchain loaders in `utils/text_processor.py`
- **Web Scraping**: Wikipedia summaries fetched in `agent_event_extractor.py` (with User-Agent)

## Common Patterns & Examples
- **Agent Initialization**: `self.chain = prompt | self.llm | JsonOutputParser()` (see `agent_event_extractor.py`)
- **File Paths**: Use `pathlib.Path` for cross-platform compatibility
- **Error Handling**: Minimal; add try/catch for API failures
- **Caching**: Events cached in JSON; check `load_existing_events()` before re-processing
- **Parallel Processing**: LangGraph handles agent concurrency (macro & belief run in parallel)

## Dependencies & Environment
- **Core**: langchain, langgraph, faiss-cpu, sentence-transformers, python-dotenv
- **Document Processing**: pypdf, unstructured (for PDFs)
- **Environment**: Python 3.8+, macOS/Linux; requires DEEPSEEK_API_KEY in .env
- **Vector Stores**: Pre-build psychology theory databases before running analysis

## File Structure Reference
- `agents/`: Individual agent implementations
- `config/`: Prompts and settings (modify here for prompt engineering)
- `rag/`: Retrieval components
- `data/`: Raw/processed data and pipeline state (located at ../data/)
- `knowledge/`: Vector stores and source documents (located at ../knowledge/)
- `utils/`: Text processing utilities
- `scripts/`: Build and maintenance scripts</content>
<parameter name="filePath">/Volumes/Extreme SSD/ai_agent_personal/.github/copilot-instructions.md