#!/usr/bin/env python3
"""
新的RAG API路由 - 基于新的多模态检索器
"""

import logging
import time
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional

from src.api.schemas import QueryRequest, RAGResponse, BatchQueryRequest, BatchRAGResponse
from src.services.new_retrieval_service import (
    NewMultimodalRetriever, RetrievalRequest, create_new_retriever
)

# 基础配置
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/rag", tags=["RAG"])

# ===================== 依赖注入 =====================
async def get_new_retriever() -> NewMultimodalRetriever:
    """获取新的检索器实例"""
    retriever = await create_new_retriever()
    return retriever

# ===================== 核心接口 =====================
@router.post("/query")
async def query_documents(request: QueryRequest, retriever: NewMultimodalRetriever = Depends(get_new_retriever)):
    """单条检索（仅返回混合结果）"""
    try:
        # 转换请求
        retrieval_request = RetrievalRequest(
            text_query=request.text_query,
            image_path=request.image_path,
            top_k=request.top_k,
            threshold=request.threshold,
            use_relation_model=True
        )
        
        # 执行检索
        resp = await retriever.retrieve(retrieval_request)
        
        # 返回结果
        return {
            "query_id": resp.query_id,
            "text_results": [{"id": r.id, "content": r.content, "score": r.score} for r in resp.text_results],
            "image_results": [{"id": r.id, "image_path": r.image_path, "score": r.score} for r in resp.image_results],
            "query_embeddings": {k: v[:5] for k, v in resp.query_embeddings.items()},  # 只返回前5个维度
            "response_time_ms": resp.response_time_ms,
            "relation_model_used": resp.relation_model_used
        }
    except Exception as e:
        logger.error(f"检索失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"检索失败: {str(e)}")

@router.post("/query-with-rag", response_model=RAGResponse)
async def query_with_rag(
    request: QueryRequest, 
    llm_provider: Optional[str] = "dashscope",
    retriever: NewMultimodalRetriever = Depends(get_new_retriever)
):
    """单条RAG查询"""
    try:
        # 转换请求
        retrieval_request = RetrievalRequest(
            text_query=request.text_query,
            image_path=request.image_path,
            top_k=request.top_k,
            threshold=request.threshold,
            use_relation_model=True
        )
        
        # 执行RAG查询
        rag_resp = await retriever.retrieve_with_rag(retrieval_request, llm_provider=llm_provider)
        
        # 构建响应
        return RAGResponse(
            query_id=rag_resp.retrieval_response.query_id,
            answer=rag_resp.generated_answer,
            source_documents=[{"id": r.id, "content": r.content, "score": r.score} 
                            for r in rag_resp.retrieval_response.text_results],
            retrieval_metrics={
                "total_time_ms": rag_resp.total_time_ms,
                "generation_time_ms": rag_resp.generation_time_ms,
                "response_time_ms": rag_resp.retrieval_response.response_time_ms
            },
            generation_time_ms=rag_resp.generation_time_ms,
            total_time_ms=rag_resp.total_time_ms
        )
    except Exception as e:
        logger.error(f"RAG查询失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"RAG查询失败: {str(e)}")

@router.post("/batch-rag", response_model=BatchRAGResponse)
async def batch_rag(
    request: BatchQueryRequest, 
    llm_provider: Optional[str] = "dashscope",
    retriever: NewMultimodalRetriever = Depends(get_new_retriever)
):
    """批量RAG查询"""
    try:
        start_time = time.time()
        
        # 转换请求
        requests = []
        for query in request.queries:
            requests.append(RetrievalRequest(
                text_query=query.text_query,
                image_path=query.image_path,
                top_k=query.top_k,
                threshold=query.threshold,
                use_relation_model=True
            ))
        
        # 批量处理
        results = []
        for req in requests:
            try:
                rag_resp = await retriever.retrieve_with_rag(req, llm_provider=llm_provider)
                results.append(RAGResponse(
                    query_id=rag_resp.retrieval_response.query_id,
                    answer=rag_resp.generated_answer,
                    source_documents=[{"id": r.id, "content": r.content, "score": r.score} 
                                    for r in rag_resp.retrieval_response.text_results],
                    retrieval_metrics={
                        "total_time_ms": rag_resp.total_time_ms,
                        "generation_time_ms": rag_resp.generation_time_ms
                    },
                    generation_time_ms=rag_resp.generation_time_ms,
                    total_time_ms=rag_resp.total_time_ms
                ))
            except Exception as e:
                logger.error(f"批量处理单个查询失败: {e}")
                results.append(RAGResponse(
                    query_id=None,
                    answer=f"处理失败: {str(e)}",
                    source_documents=[],
                    retrieval_metrics={"total_time_ms": 0},
                    generation_time_ms=0,
                    total_time_ms=0
                ))
        
        total_time_ms = (time.time() - start_time) * 1000
        
        return BatchRAGResponse(
            results=results,
            total_time_ms=total_time_ms
        )
        
    except Exception as e:
        logger.error(f"批量RAG失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"批量RAG失败: {str(e)}")

# ===================== 训练接口 =====================
@router.post("/train-relation-model")
async def train_relation_model(retriever: NewMultimodalRetriever = Depends(get_new_retriever)):
    """训练关联模型"""
    try:
        logger.info("开始训练关联模型...")
        
        success = await retriever.train_relation_model()
        
        if success:
            return {
                "status": "success",
                "message": "关联模型训练完成",
                "details": "模型已保存到 data/models/relation_model.pkl"
            }
        else:
            return {
                "status": "warning",
                "message": "关联模型训练失败",
                "details": "训练数据不足或训练过程中出现错误"
            }
            
    except Exception as e:
        logger.error(f"训练关联模型失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"训练关联模型失败: {str(e)}")

@router.get("/training-data")
async def get_training_data(retriever: NewMultimodalRetriever = Depends(get_new_retriever)):
    """获取训练数据统计"""
    try:
        training_pairs = retriever.training_collector.get_training_pairs()
        
        return {
            "total_pairs": len(training_pairs),
            "data_path": retriever.training_collector.data_path,
            "sample_pairs": [
                {
                    "text_id": pair.text_id,
                    "image_id": pair.image_id,
                    "similarity_score": pair.similarity_score
                }
                for pair in training_pairs[:5]  # 返回前5个样本
            ]
        }
        
    except Exception as e:
        logger.error(f"获取训练数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取训练数据失败: {str(e)}")

# ===================== 辅助接口 =====================
@router.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "service": "new-multimodal-rag"}

@router.get("/models")
async def list_models():
    """模型列表"""
    return {
        "llm": ["qwen3-max (DashScope)", "deepseek", "zhipu"],
        "embedding": ["bge-m3 (Ollama)", "text-embedding-v2 (DashScope)"],
        "vlm": ["qwen3-max (DashScope)"],
        "relation_model": ["knn (scikit-learn)"]
    }

@router.get("/system-info")
async def system_info(retriever: NewMultimodalRetriever = Depends(get_new_retriever)):
    """系统信息"""
    try:
        # 获取数据库统计
        doc_count = await retriever.vector_store.get_document_count()
        image_count = await retriever.vector_store.get_image_count()
        
        # 检查关联模型
        relation_model_loaded = False
        try:
            retriever.relation_model.load()
            relation_model_loaded = True
        except:
            relation_model_loaded = False
        
        return {
            "system": "新的多模态RAG系统",
            "version": "2.0.0",
            "architecture": {
                "text_embedding": "本地Ollama (bge-m3)",
                "image_embedding": "阿里云DashScope (qwen3-max)",
                "llm": "阿里云DashScope (qwen3-max)",
                "relation_model": "KNN关联模型",
                "database": "PostgreSQL + pgvector"
            },
            "statistics": {
                "documents": doc_count,
                "images": image_count,
                "training_pairs": len(retriever.training_collector.get_training_pairs()),
                "relation_model_loaded": relation_model_loaded
            },
            "capabilities": [
                "文本检索",
                "图像检索", 
                "多模态检索",
                "RAG生成",
                "关联模型训练"
            ]
        }
        
    except Exception as e:
        logger.error(f"获取系统信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取系统信息失败: {str(e)}")