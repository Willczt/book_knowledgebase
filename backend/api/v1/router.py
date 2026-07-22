from fastapi import APIRouter

from api.v1.knowledge_base import router as kb_router
from api.v1.documents import router as docs_router
from api.v1.sessions import router as sessions_router
from api.v1.chat import router as chat_router
from api.v1.search import router as search_router
from api.v1.config_routes import router as config_router

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(kb_router)
api_router.include_router(docs_router)
api_router.include_router(sessions_router)
api_router.include_router(chat_router)
api_router.include_router(search_router)
api_router.include_router(config_router)
