# 多模态RAG系统主API文件
import sys
import os
import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# 获取项目根目录（rag_for_gradute）
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
# 将根目录加入sys.path
sys.path.insert(0, PROJECT_ROOT)

from src.api.routers import rag
from src.api.schemas import HealthCheck, ErrorResponse, SystemStatus
from config.settings import settings
from config.database import database
from src.database.vector_store import vector_store
from src.services.ollama_service import OllamaService

# 配置日志
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 全局变量
app_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    logger.info("Starting Multimodal RAG API...")
    
    try:
        # 初始化数据库连接
        await database.test_connection()
        await vector_store.connect()
        await vector_store.create_tables()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        # 不阻止启动，但记录错误
    
    try:
        # 测试Ollama连接
        async with OllamaService() as ollama_service:
            # 简单测试连接
            pass
        logger.info("Ollama service is available")
    except Exception as e:
        logger.warning(f"Ollama service may not be available: {e}")
    
    yield
    
    # 关闭时
    logger.info("Shutting down Multimodal RAG API...")
    await vector_store.close()
    logger.info("Cleanup completed")


# 创建FastAPI应用
app = FastAPI(
    title="多模态RAG系统API",
    description="基于文本-图像关联的本地知识库构建与多模态检索增强系统——以校园导览为例",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境中应该限制来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 全局异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理器"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=exc.detail.get("error", "Unknown error") if isinstance(exc.detail, dict) else str(exc.detail),
                detail=exc.detail.get("detail") if isinstance(exc.detail, dict) else None,
                request_id=request.headers.get("X-Request-ID")
            ).dict()
        )
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            request_id=request.headers.get("X-Request-ID")
        ).dict()
    )


# 中间件：请求日志
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """请求日志中间件"""
    start_time = time.time()
    request_id = request.headers.get("X-Request-ID", "N/A")
    
    logger.info(f"Request started: {request.method} {request.url.path} - ID: {request_id}")
    
    try:
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000
        
        logger.info(
            f"Request completed: {request.method} {request.url.path} "
            f"- Status: {response.status_code} "
            f"- Time: {process_time:.2f}ms "
            f"- ID: {request_id}"
        )
        
        response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
        return response
        
    except Exception as exc:
        process_time = (time.time() - start_time) * 1000
        logger.error(
            f"Request failed: {request.method} {request.url.path} "
            f"- Error: {exc} "
            f"- Time: {process_time:.2f}ms "
            f"- ID: {request_id}"
        )
        raise


# 注册路由
app.include_router(rag.router, prefix="/v1")  # 旧版本API
from src.api.routers.new_rag import router as new_rag_router
app.include_router(new_rag_router, prefix="/v2")  # 新版本API


# 根路由
@app.get("/", response_model=HealthCheck)
async def root():
    """根路由"""
    return HealthCheck(
        status="healthy",
        version="1.0.0"
    )


@app.get("/health", response_model=HealthCheck)
async def health():
    """健康检查端点"""
    return HealthCheck(
        status="healthy",
        version="1.0.0"
    )


@app.get("/status", response_model=SystemStatus)
async def system_status():
    """系统状态检查"""
    try:
        # 检查数据库连接
        db_connected = await database.test_connection()
        
        # 检查Ollama服务
        ollama_available = False
        try:
            async with OllamaService() as service:
                ollama_available = True
        except:
            ollama_available = False
        
        # 获取统计信息
        doc_count = 0
        image_count = 0
        relation_count = 0
        
        try:
            doc_count = await vector_store.get_document_count()
            image_count = await vector_store.get_image_count()
            # 关系数量需要额外查询
        except:
            pass
        
        uptime = time.time() - app_start_time
        
        return SystemStatus(
            status="running",
            database_connected=db_connected,
            ollama_available=ollama_available,
            document_count=doc_count,
            image_count=image_count,
            relation_count=relation_count,
            uptime_seconds=uptime,
            version="1.0.0"
        )
        
    except Exception as e:
        logger.error(f"System status check failed: {e}")
        return SystemStatus(
            status="degraded",
            database_connected=False,
            ollama_available=False,
            document_count=0,
            image_count=0,
            relation_count=0,
            uptime_seconds=time.time() - app_start_time,
            version="1.0.0"
        )


@app.get("/info")
async def api_info():
    """API信息"""
    return {
        "name": "多模态RAG系统API",
        "description": "基于文本-图像关联的本地知识库构建与多模态检索增强系统",
        "version": "1.0.0",
        "author": "校园导览项目组",
        "endpoints": {
            "rag": {
                "query": "POST /rag/query - 多模态检索",
                "query_with_rag": "POST /rag/query-with-rag - RAG检索+生成",
                "batch_query": "POST /rag/batch-query - 批量检索",
                "batch_rag": "POST /rag/batch-rag - 批量RAG"
            },
            "system": {
                "health": "GET /health - 健康检查",
                "status": "GET /status - 系统状态",
                "info": "GET /info - API信息"
            }
        },
        "technology_stack": {
            "vector_database": "PostgreSQL + pgvector",
            "text_embedding": "BGE3 (via Ollama)",
            "image_embedding": "Qwen3-Max (via DashScope)",
            "llm_model": "Qwen3-Max (via DashScope)",
            "relation_model": "KNN (scikit-learn)",
            "framework": "FastAPI + 新的多模态架构"
        },
        "api_versions": {
            "v1": "旧版多模态RAG API",
            "v2": "新版多模态RAG API (推荐)"
        }
    }


# 主函数
def main():
    """主函数"""
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # 开发模式启用热重载
        log_level="info"
    )


if __name__ == "__main__":
    main()