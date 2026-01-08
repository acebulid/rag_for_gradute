# API的Pydantic模型
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class QueryRequest(BaseModel):
    """查询请求"""
    text_query: Optional[str] = Field(None, description="文本查询")
    image_path: Optional[str] = Field(None, description="图像文件路径")
    top_k: int = Field(5, ge=1, le=20, description="返回结果数量")
    threshold: float = Field(0.3, ge=0.0, le=1.0, description="相似度阈值")
    text_weight: float = Field(0.5, ge=0.0, le=1.0, description="文本查询权重")
    image_weight: float = Field(0.5, ge=0.0, le=1.0, description="图像查询权重")
    
    class Config:
        schema_extra = {
            "example": {
                "text_query": "图书馆在哪里？",
                "image_path": "/path/to/campus_image.jpg",
                "top_k": 5,
                "threshold": 0.3,
                "text_weight": 0.6,
                "image_weight": 0.4
            }
        }


class SearchResult(BaseModel):
    """搜索结果"""
    id: str = Field(..., description="文档ID")
    content: str = Field(..., description="文档内容")
    score: float = Field(..., ge=0.0, le=1.0, description="相似度分数")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    source: Optional[str] = Field(None, description="数据来源")


class ImageSearchResult(BaseModel):
    """图像搜索结果"""
    id: str = Field(..., description="图像ID")
    image_path: str = Field(..., description="图像文件路径")
    vlm_description: str = Field(..., description="VLM生成的描述")
    score: float = Field(..., ge=0.0, le=1.0, description="相似度分数")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class QueryResponse(BaseModel):
    """查询响应"""
    query_id: Optional[str] = Field(None, description="查询ID")
    text_results: List[SearchResult] = Field(default_factory=list, description="文本检索结果")
    image_results: List[ImageSearchResult] = Field(default_factory=list, description="图像检索结果")
    hybrid_results: List[SearchResult] = Field(default_factory=list, description="混合检索结果")
    query_embeddings: Dict[str, List[float]] = Field(default_factory=dict, description="查询嵌入向量")
    response_time_ms: float = Field(..., description="响应时间（毫秒）")


class RAGResponse(BaseModel):
    """RAG响应"""
    query_id: Optional[str] = Field(None, description="查询ID")
    answer: str = Field(..., description="生成的回答")
    source_documents: List[SearchResult] = Field(default_factory=list, description="源文档")
    retrieval_metrics: Dict[str, float] = Field(..., description="检索指标")
    generation_time_ms: Optional[float] = Field(None, description="生成时间（毫秒）")
    total_time_ms: Optional[float] = Field(None, description="总时间（毫秒）")


class BatchQueryRequest(BaseModel):
    """批量查询请求"""
    queries: List[QueryRequest] = Field(..., description="查询列表")
    
    class Config:
        schema_extra = {
            "example": {
                "queries": [
                    {"text_query": "图书馆在哪里？", "top_k": 3},
                    {"image_path": "/path/to/building.jpg", "top_k": 3},
                    {"text_query": "食堂开放时间", "image_path": "/path/to/cafeteria.jpg", "top_k": 5}
                ]
            }
        }


class BatchQueryResponse(BaseModel):
    """批量查询响应"""
    results: List[QueryResponse] = Field(..., description="查询结果列表")
    total_time_ms: float = Field(..., description="总处理时间（毫秒）")


class BatchRAGResponse(BaseModel):
    """批量RAG响应"""
    results: List[RAGResponse] = Field(..., description="RAG结果列表")
    total_time_ms: float = Field(..., description="总处理时间（毫秒）")


class ProcessingRequest(BaseModel):
    """数据处理请求"""
    text_dir: Optional[str] = Field(None, description="文本目录路径")
    image_dir: Optional[str] = Field(None, description="图像目录路径")
    metadata_file: Optional[str] = Field(None, description="元数据文件路径")
    batch_size: int = Field(10, ge=1, le=50, description="批处理大小")
    
    class Config:
        schema_extra = {
            "example": {
                "text_dir": "/path/to/texts",
                "image_dir": "/path/to/images",
                "batch_size": 10
            }
        }


class ProcessingStats(BaseModel):
    """处理统计"""
    total_texts: int = Field(0, description="总文本文件数")
    processed_texts: int = Field(0, description="已处理文本文件数")
    failed_texts: int = Field(0, description="失败文本文件数")
    total_images: int = Field(0, description="总图像文件数")
    processed_images: int = Field(0, description="已处理图像文件数")
    failed_images: int = Field(0, description="失败图像文件数")
    created_relations: int = Field(0, description="创建的关联数")
    elapsed_time_seconds: float = Field(..., description="经过时间（秒）")
    text_success_rate: float = Field(..., ge=0.0, le=1.0, description="文本处理成功率")
    image_success_rate: float = Field(..., ge=0.0, le=1.0, description="图像处理成功率")


class ProcessingResponse(BaseModel):
    """处理响应"""
    task_id: str = Field(..., description="任务ID")
    status: str = Field(..., description="状态")
    stats: Optional[ProcessingStats] = Field(None, description="处理统计")
    message: Optional[str] = Field(None, description="消息")


class SystemStatus(BaseModel):
    """系统状态"""
    status: str = Field(..., description="系统状态")
    database_connected: bool = Field(..., description="数据库连接状态")
    ollama_available: bool = Field(..., description="Ollama可用状态")
    document_count: int = Field(0, description="文档数量")
    image_count: int = Field(0, description="图像数量")
    relation_count: int = Field(0, description="关联数量")
    uptime_seconds: Optional[float] = Field(None, description="运行时间（秒）")
    version: str = Field("1.0.0", description="系统版本")


class ErrorResponse(BaseModel):
    """错误响应"""
    error: str = Field(..., description="错误信息")
    detail: Optional[str] = Field(None, description="错误详情")
    request_id: Optional[str] = Field(None, description="请求ID")


class HealthCheck(BaseModel):
    """健康检查"""
    status: str = Field(..., description="健康状态")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")
    version: str = Field("1.0.0", description="API版本")


# 工具函数
def query_request_to_retrieval_request(query_request: QueryRequest):
    """将QueryRequest转换为RetrievalRequest"""
    from src.services.retrieval_service import RetrievalRequest
    
    return RetrievalRequest(
        text_query=query_request.text_query,
        image_path=query_request.image_path,
        top_k=query_request.top_k,
        threshold=query_request.threshold,
        text_weight=query_request.text_weight,
        image_weight=query_request.image_weight
    )