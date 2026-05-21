from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.agents import create_agent
from app.common.logger import logger
import sqlite3
import os

load_dotenv()
base_url = os.getenv("DASHSCOPE_BASE_URL")
api_key=os.getenv("DASHSCOPE_API_KEY")
llm=init_chat_model(
    model="qwen3.6-plus",
    model_provider="openai",
    base_url=base_url,
    api_key=api_key,
    timeout=60,
)

web_search=TavilySearch(
    max_results=5,
    topic="general"
)
# 创建数据库会话记忆存储
connection=sqlite3.connect("db/personal_chief.db", check_same_thread=False)
checkpointer=SqliteSaver(connection)
checkpointer.setup()

system_prompt = """
你是一名私人厨师。收到用户提供的食材照片或清单后，请按以下流程操作：
1.识别和评估食材：若用户提供照片，首先辨识所有可见食材。基于食材的外观状态，评估其新鲜度与可用量，整理出一份“当前可用食材清单”。
2.智能食谱检索：优先调用 web_search 工具，以“可用食材清单”为核心关键词，查找可行菜谱。
3.多维度评估与排序：从营养价值和制作难度两个维度对检索到的候选食谱进行量化打分，并根据得分排序，制作简单且营养丰富的排名靠前。
4.结构化方案输出：把排序后的食谱整理为一份结构清晰的建议报告，要包含食谱信息、得分、推荐理由、食谱的参考图片，帮助用户快速做出决策。

请严格按照流程，优先调用 web_search 工具搜索食谱，搜索不到的情况下才能自己发挥。
"""

agent=create_agent(
    model=llm,
    tools=[web_search],
    checkpointer=checkpointer,
    system_prompt=system_prompt
)

async def search_recipe(prompt: str,image: str,thread_id: str):
    try:
        if not image or image.strip()=="":
            message=HumanMessage(content=prompt)
        else:
            message=HumanMessage(content=[
                {"type":"image","url":image},
                {"type":"text","text":prompt}
            ])
        for chunk,metadata in agent.stream(
            {"messages":[message]},
             {"configurable": {"thread_id": thread_id}},
             stream_mode="messages"
        ):
            if isinstance(chunk,AIMessage) and chunk.content:
                yield chunk.content
    except Exception as e:
        logger.error(f"\n错误:{str(e)}")
        yield "抱歉，处理您的请求时发生了错误，请稍后再试。"

async def get_messages(thread_id: str):
    """获取历史消息"""
    logger.info(f"获取历史消息:{thread_id}")
    checkpoint=checkpointer.get({"configurable": {"thread_id": thread_id}})
    # 如果不存在，返回空列表
    if not checkpoint:
        return []

    # 安全获取 messages
    channel_values = checkpoint.get("channel_values")
    if not channel_values:
        return []

    messages = channel_values.get("messages", [])
    if not messages:
        return []

    # 转换消息格式
    result = []
    for msg in messages:
        if not msg.content:
            continue

        if isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            result.append({"role": "assistant", "content": msg.content})

    return result

async def clear_messages(thread_id: str):
    """清空历史消息"""
    logger.info(f"清空历史消息，thread_id: {thread_id}")
    checkpointer.delete_thread(thread_id)