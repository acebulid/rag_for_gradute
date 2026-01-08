# RAG API 路由
import logging
import time
from fastapi import APIRouter, HTTPException, Depends
from src.api.schemas import QueryRequest, RAGResponse, BatchQueryRequest, BatchRAGResponse
from src.services.retrieval_service import MultimodalRetriever, RetrievalRequest, BatchRetriever

# 基础配置
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/rag", tags=["RAG"])

# ===================== 依赖注入 =====================
async def get_retriever():
    """获取检索器实例"""
    from src.services.ollama_service import OllamaService
    from src.database.vector_store import PostgreSQLVectorStore
    
    ollama_service = OllamaService()
    vector_store = PostgreSQLVectorStore()
    await vector_store.connect()
    return MultimodalRetriever(ollama_service, vector_store)

# ===================== 核心接口 =====================
@router.post("/query")
async def query_documents(request: QueryRequest, retriever=Depends(get_retriever)):
    """单条检索（仅返回混合结果）"""
    try:
        # 执行检索
        resp = await retriever.retrieve(RetrievalRequest(**request.dict()))
        # 只返回核心字段
        return {
            "query_id": resp.query_id,
            "results": [{"id": r.id, "content": r.content, "score": r.score} for r in resp.hybrid_results],
            "time_ms": resp.response_time_ms
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检索失败: {str(e)}")

@router.post("/query-with-rag", response_model=RAGResponse)
async def query_with_rag(request: QueryRequest, retriever=Depends(get_retriever)):
    """单条RAG查询"""
    try:
        rag_resp = await retriever.retrieve_with_rag(RetrievalRequest(**request.dict()))
        # 响应构建（复用已有模型）
        return RAGResponse(
            query_id=rag_resp.retrieval_response.query_id,
            answer=rag_resp.generated_answer,
            source_documents=[{"id": r.id, "content": r.content, "score": r.score} for r in rag_resp.retrieval_response.hybrid_results],
            retrieval_metrics={"total_time_ms": rag_resp.total_time_ms},
            generation_time_ms=rag_resp.generation_time_ms,
            total_time_ms=rag_resp.total_time_ms
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG查询失败: {str(e)}")

@router.post("/batch-rag", response_model=BatchRAGResponse)
async def batch_rag(request: BatchQueryRequest, retriever=Depends(get_retriever)):
    """批量RAG查询"""
    try:
        start = time.time()
        # 转换请求
        requests = [RetrievalRequest(**q.dict()) for q in request.queries]
        # 批量处理
        batch_retriever = BatchRetriever(retriever)
        rag_resps = await batch_retriever.batch_retrieve_with_rag(requests)
        # 简化结果构建
        results = []
        for resp in rag_resps:
            if isinstance(resp, Exception):
                results.append({"query_id": None, "answer": f"失败: {str(resp)}", "source_documents": [], "retrieval_metrics": {"total_time_ms": 0}})
            else:
                results.append({
                    "query_id": resp.retrieval_response.query_id,
                    "answer": resp.generated_answer,
                    "source_documents": [{"id": r.id, "content": r.content} for r in resp.retrieval_response.hybrid_results],
                    "retrieval_metrics": {"total_time_ms": resp.total_time_ms}
                })
        # 返回批量结果
        return BatchRAGResponse(results=results, total_time_ms=(time.time()-start)*1000)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量RAG失败: {str(e)}")

# ===================== 辅助接口 =====================
@router.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}

@router.get("/models")
async def list_models():
    """模型列表"""
    return {"llm": ["qwen2.5:7b"], "embedding": ["bge-m3"], "vlm": ["qwen2.5-vl"]}