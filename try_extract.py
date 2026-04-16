from agents.agent_event_extractor import EventExtractorAgent
import os
from dotenv import load_dotenv
# 新路径（官方标准）
from langchain_openai import ChatOpenAI
load_dotenv() 


llm = ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        temperature=0.3,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    )
    

agent = EventExtractorAgent(llm)

# 方式1：直接传入传记文本（一次性）
#events = agent.extract_events("Donald Trump", biography_text=long_text, num_events=15)

# 方式2：自动抓取并增量更新（检查缓存）
events = agent.extract_events("Donald Trump", force_refresh=False, num_events=50)  # 使用缓存
#events = agent.extract_events("Donald Trump", force_refresh=True)   # 强制重新抓取并合并