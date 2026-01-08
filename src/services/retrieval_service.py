import logging
import asyncio
import time
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import uuid

from src.services.ollama_service import OllamaService, EmbeddingResult, GenerationResult
from src.database.vector_store import PostgreSQLVectorStore, SearchResult, ImageSearchResult
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class RetrievalRequest:
    """检索请求"""
    text_query: Optional[str] = None
    image_path: Optional[str] = None
    top_k: int = 5
    threshold: float = 0.3
    text_weight: float = 0.5
    image_weight: float = 0.5


@dataclass
class RetrievalResponse:
    """检索响应"""
    request: RetrievalRequest
    text_results: List[SearchResult]
    image_results: List[ImageSearchResult]
    hybrid_results: List[SearchResult]
    query_embeddings: Dict[str, List[float]]
    response_time_ms: float
    query_id: Optional[str] = None


@dataclass
class RAGResponse:
    """RAG响应"""
    retrieval_response: RetrievalResponse
    generated_answer: Optional[str] = None
    generation_time_ms: Optional[float] = None
    total_time_ms: Optional[float] = None


class MultimodalRetriever:
    """多模态检索器"""
    
    def __init__(self, ollama_service: OllamaService, vector_store: PostgreSQLVectorStore):
        self.ollama = ollama_service
        self.vector_store = vector_store
    
    async def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        """执行多模态检索"""
        start_time = time.time()
        
        # 生成查询嵌入
        query_embeddings = {}
        tasks = []
        
        if request.text_query:
            tasks.append(self._get_text_embedding(request.text_query))
        
        if request.image_path:
            tasks.append(self._get_image_embedding(request.image_path))
        
        # 并行执行嵌入生成
        embedding_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理嵌入结果
        text_embedding = None
        image_embedding = None
        
        for i, result in enumerate(embedding_results):
            if isinstance(result, Exception):
                logger.error(f"Failed to generate embedding: {result}")
                continue
            
            if i == 0 and request.text_query:
                text_embedding = result.embedding if isinstance(result, EmbeddingResult) else None
                if text_embedding:
                    query_embeddings['text'] = text_embedding
            elif request.image_path:
                if isinstance(result, GenerationResult):
                    # 图像描述生成成功，现在需要将描述转换为嵌入
                    desc_embedding = await self._get_text_embedding(result.text)
                    if desc_embedding:
                        image_embedding = desc_embedding.embedding
                        query_embeddings['image'] = image_embedding
        
        # 执行检索
        text_results = []
        image_results = []
        hybrid_results = []
        
        if text_embedding:
            text_results = await self.vector_store.search_similar_documents(
                text_embedding, request.top_k * 2, request.threshold
            )
        
        if image_embedding:
            # 使用图像描述嵌入搜索文档
            image_desc_results = await self.vector_store.search_similar_documents(
                image_embedding, request.top_k * 2, request.threshold
            )
            image_results = image_desc_results
        
        # 混合检索
        if text_embedding or image_embedding:
            hybrid_results = await self.vector_store.hybrid_search(
                text_query_embedding=text_embedding,
                image_query_embedding=image_embedding,
                top_k=request.top_k,
                text_weight=request.text_weight,
                image_weight=request.image_weight
            )
        
        response_time_ms = (time.time() - start_time) * 1000
        
        # 记录查询历史
        query_id = None
        try:
            query_id = await self.vector_store.log_query_history(
                query_text=request.text_query,
                query_image_path=request.image_path,
                query_type="hybrid" if request.text_query and request.image_path else 
                          "text" if request.text_query else "image",
                retrieved_document_ids=[r.id for r in hybrid_results],
                retrieved_image_ids=[],
                response_time_ms=response_time_ms
            )
        except Exception as e:
            logger.error(f"Failed to log query history: {e}")
        
        return RetrievalResponse(
            request=request,
            text_results=text_results[:request.top_k],
            image_results=image_results[:request.top_k],
            hybrid_results=hybrid_results,
            query_embeddings=query_embeddings,
            response_time_ms=response_time_ms,
            query_id=query_id
        )
    
    async def _get_text_embedding(self, text: str) -> Optional[EmbeddingResult]:
        """获取文本嵌入"""
        try:
            # OllamaService现在是同步的，需要在异步环境中运行
            import asyncio
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.ollama.generate_embedding, text)
        except Exception as e:
            logger.error(f"Failed to generate text embedding: {e}")
            return None
    
    async def _get_image_embedding(self, image_path: str) -> Optional[GenerationResult]:
        """获取图像描述（用于生成嵌入）"""
        try:
            # OllamaService现在是同步的，需要在异步环境中运行
            import asyncio
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.ollama.generate_image_description, image_path)
        except Exception as e:
            logger.error(f"Failed to generate image description: {e}")
            return None
    
    async def retrieve_with_rag(self, request: RetrievalRequest, 
                              system_prompt: Optional[str] = None) -> RAGResponse:
        """执行RAG检索（检索+生成）"""
        rag_start_time = time.time()
        
        # 执行检索
        retrieval_response = await self.retrieve(request)
        
        # 如果没有检索结果，直接返回
        if not retrieval_response.hybrid_results:
            total_time_ms = (time.time() - rag_start_time) * 1000
            return RAGResponse(
                retrieval_response=retrieval_response,
                generated_answer="抱歉，没有找到相关的信息。",
                generation_time_ms=0,
                total_time_ms=total_time_ms
            )
        
        # 构建上下文
        context = self._build_context(retrieval_response.hybrid_results)
        
        # 构建提示
        prompt = self._build_rag_prompt(
            text_query=request.text_query,
            image_query=request.image_path,
            context=context
        )
        
        # 生成回答
        generation_start_time = time.time()
        try:
            # OllamaService现在是同步的，需要在异步环境中运行
            import asyncio
            loop = asyncio.get_event_loop()
            generation_result = await loop.run_in_executor(
                None, 
                self.ollama.generate_text,
                prompt,
                None,  # model使用默认
                system_prompt or self._get_default_system_prompt(),
                {"options": {"temperature": 0.7, "top_p": 0.9}}
            )
            generation_time_ms = (time.time() - generation_start_time) * 1000
            generated_answer = generation_result.text
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            generation_time_ms = (time.time() - generation_start_time) * 1000
            generated_answer = f"生成回答时出错: {str(e)}"
        
        total_time_ms = (time.time() - rag_start_time) * 1000
        
        return RAGResponse(
            retrieval_response=retrieval_response,
            generated_answer=generated_answer,
            generation_time_ms=generation_time_ms,
            total_time_ms=total_time_ms
        )
    
    def _build_context(self, results: List[SearchResult]) -> str:
        """构建上下文"""
        context_parts = []
        
        for i, result in enumerate(results[:3]):  # 使用前3个结果作为上下文
            context_parts.append(f"[文档 {i+1}, 相关性: {result.score:.3f}]:\n{result.content}\n")
        
        return "\n".join(context_parts)
    
    def _build_rag_prompt(self, text_query: Optional[str], 
                         image_query: Optional[str], 
                         context: str) -> str:
        """构建RAG提示"""
        query_part = ""
        if text_query and image_query:
            query_part = f"用户提供了文本查询和图像查询：\n文本查询: {text_query}\n图像查询: {image_path_to_description(image_query)}"
        elif text_query:
            query_part = f"用户查询: {text_query}"
        elif image_query:
            query_part = f"用户提供了图像查询: {image_path_to_description(image_query)}"
        
        prompt = f"""基于以下检索到的上下文信息，回答用户的问题。

{query_part}

检索到的上下文信息：
{context}

请根据上下文信息提供准确、详细的回答。如果上下文信息不足以回答问题，请说明这一点。
回答时请引用相关的上下文信息。

回答："""
        
        return prompt
    
    def _get_default_system_prompt(self) -> str:
        """获取默认系统提示"""
        return """你是一个专业的校园导览助手。你的任务是基于提供的上下文信息，回答用户关于校园的问题。
请确保回答准确、详细，并且基于上下文信息。如果上下文信息不足，请诚实地说明。
对于校园建筑、设施、历史等方面的问题，请提供尽可能详细的信息。
回答请使用中文。"""


def image_path_to_description(image_path: str) -> str:
    """将图像路径转换为描述性文字"""
    # 这里可以添加更复杂的逻辑来从路径提取信息
    return f"图像文件: {image_path}"


class BatchRetriever:
    """批量检索器"""
    
    def __init__(self, retriever: MultimodalRetriever):
        self.retriever = retriever
    
    async def batch_retrieve(self, requests: List[RetrievalRequest]) -> List[RetrievalResponse]:
        """批量检索"""
        tasks = [self.retriever.retrieve(req) for req in requests]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def batch_retrieve_with_rag(self, requests: List[RetrievalRequest],
                                    system_prompt: Optional[str] = None) -> List[RAGResponse]:
        """批量RAG检索"""
        tasks = [self.retriever.retrieve_with_rag(req, system_prompt) for req in requests]
        return await asyncio.gather(*tasks, return_exceptions=True)


# 工具函数
async def create_retriever() -> MultimodalRetriever:
    """创建检索器实例"""
    # OllamaService现在是同步的，直接实例化
    ollama_service = OllamaService()
    vector_store = PostgreSQLVectorStore()
    await vector_store.connect()
    return MultimodalRetriever(ollama_service, vector_store)
