from fastapi import APIRouter
from app.models.schemas import ChatRequest
from fastapi.responses import StreamingResponse
from app.agent.personal_agent import search_recipe,clear_messages,get_messages
router = APIRouter()


@router.post("/chat/stream")
async def chat_endpoint(request: ChatRequest):
    """流式对话"""
    return StreamingResponse(
        search_recipe(request.message,request.image_url,request.thread_id),
        media_type="text/event-stream"
    )


@router.get("/chat/messages")
async def get_chat_messages(thread_id: str):
    """获取历史消息"""
    messages=await get_messages(thread_id)
    return {"messages":messages}


@router.delete("/chat/messages")
async def clear_chat_messages(thread_id: str):
    """清空历史消息"""
    await clear_messages(thread_id)
    return {"success": True}