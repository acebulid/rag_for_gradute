#!/usr/bin/env python3
"""
新的检索服务 - 基于阿里云DashScope API和关联模型
"""

import logging
import asyncio
import time
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np

from src.services.ollama_service import OllamaService, get_ollama_service
from src.services.dashscope_service import DashScopeService, get_dashscope_service
from src.services.relation_model import RelationModel, get_relation_model, TrainingDataCollector
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
    use_relation_model: bool = True  # 是否使用关联模型


@dataclass
class RetrievalResponse:
    """检索响应"""
    request: RetrievalRequest
    text_results: List[SearchResult]
    image_results: List[ImageSearchResult]
    query_embeddings: Dict[str, List[float]]
    response_time_ms: float
    query_id: Optional[str] = None
    relation_model_used: bool = False


@dataclass
class RAGResponse:
    """RAG响应"""
    retrieval_response: RetrievalResponse
    generated_answer: Optional[str] = None
    generation_time_ms: Optional[float] = None
    total_time_ms: Optional[float] = None
    llm_provider: str = "deepseek"  # deepseek 或 zhipu


class NewMultimodalRetriever:
    """新的多模态检索器 - 基于阿里云API和关联模型"""
    
    def __init__(self):
        self.ollama = get_ollama_service()  # 用于文本向量化
        self.dashscope = get_dashscope_service()  # 用于图像向量化
        self.relation_model = get_relation_model()  # 关联模型
        self.vector_store = PostgreSQLVectorStore()
        self.training_collector = TrainingDataCollector()
        
        # 尝试加载关联模型
        try:
            self.relation_model.load()
            logger.info("关联模型加载成功")
        except Exception as e:
            logger.warning(f"关联模型加载失败: {e}")
    
    async def initialize(self):
        """初始化"""
        await self.vector_store.connect()
    
    async def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        """执行多模态检索"""
        start_time = time.time()
        
        # 生成查询嵌入
        query_embeddings = {}
        text_embedding = None
        image_embedding = None
        
        # 1. 处理文本查询
        if request.text_query:
            try:
                # 使用本地Ollama生成文本向量
                embedding_result = self.ollama.generate_embedding(request.text_query)
                text_embedding = embedding_result.embedding
                query_embeddings['text'] = text_embedding
                logger.info(f"文本向量生成成功，维度: {len(text_embedding)}")
            except Exception as e:
                logger.error(f"文本向量生成失败: {e}")
        
        # 2. 处理图像查询
        if request.image_path:
            try:
                # 使用阿里云DashScope生成图像向量
                image_embedding_result = self.dashscope.generate_image_embedding(request.image_path)
                image_embedding = image_embedding_result.embedding
                query_embeddings['image'] = image_embedding
                logger.info(f"图像向量生成成功，维度: {len(image_embedding)}")
            except Exception as e:
                logger.error(f"图像向量生成失败: {e}")
        
        # 3. 执行检索
        text_results = []
        image_results = []
        relation_model_used = False
        
        # 情况1: 只有文本查询
        if text_embedding and not image_embedding:
            text_results = await self.vector_store.search_similar_documents(
                text_embedding, request.top_k, request.threshold
            )
            logger.info(f"文本检索结果: {len(text_results)} 个文档")
        
        # 情况2: 只有图像查询
        elif image_embedding and not text_embedding:
            if request.use_relation_model:
                # 使用关联模型找到相关的文字向量
                try:
                    similar_texts = self.relation_model.find_similar_texts(
                        image_embedding, top_k=request.top_k
                    )
                    relation_model_used = True
                    
                    # 根据文字ID获取文档
                    for text_id, similarity in similar_texts:
                        # 这里需要根据text_id获取文档内容
                        # 简化处理：直接使用向量搜索
                        pass
                    
                    logger.info(f"关联模型预测结果: {len(similar_texts)} 个相关文字")
                except Exception as e:
                    logger.warning(f"关联模型预测失败，使用备用方案: {e}")
            
            # 备用方案：使用图像向量直接搜索图像描述
            image_results = await self.vector_store.search_similar_images(
                image_embedding, request.top_k, request.threshold
            )
            logger.info(f"图像检索结果: {len(image_results)} 个图像")
        
        # 情况3: 文本+图像查询
        elif text_embedding and image_embedding:
            # 优先使用图像查询结果
            if request.use_relation_model:
                try:
                    similar_texts = self.relation_model.find_similar_texts(
                        image_embedding, top_k=request.top_k
                    )
                    relation_model_used = True
                    logger.info(f"关联模型预测结果: {len(similar_texts)} 个相关文字")
                except Exception as e:
                    logger.warning(f"关联模型预测失败: {e}")
            
            # 同时进行文本检索
            text_results = await self.vector_store.search_similar_documents(
                text_embedding, request.top_k, request.threshold
            )
            logger.info(f"文本检索结果: {len(text_results)} 个文档")
        
        response_time_ms = (time.time() - start_time) * 1000
        
        # 记录查询历史
        query_id = None
        try:
            query_id = await self.vector_store.log_query_history(
                query_text=request.text_query,
                query_image_path=request.image_path,
                query_type="multimodal",
                retrieved_document_ids=[r.id for r in text_results],
                retrieved_image_ids=[r.id for r in image_results],
                response_time_ms=response_time_ms
            )
        except Exception as e:
            logger.error(f"Failed to log query history: {e}")
        
        return RetrievalResponse(
            request=request,
            text_results=text_results,
            image_results=image_results,
            query_embeddings=query_embeddings,
            response_time_ms=response_time_ms,
            query_id=query_id,
            relation_model_used=relation_model_used
        )
    
    async def retrieve_with_rag(self, request: RetrievalRequest, 
                              llm_provider: str = "deepseek") -> RAGResponse:
        """执行RAG检索（检索+生成）"""
        rag_start_time = time.time()
        
        # 执行检索
        retrieval_response = await self.retrieve(request)
        
        # 如果没有检索结果，直接返回
        if not retrieval_response.text_results:
            total_time_ms = (time.time() - rag_start_time) * 1000
            return RAGResponse(
                retrieval_response=retrieval_response,
                generated_answer="抱歉，没有找到相关的信息。",
                generation_time_ms=0,
                total_time_ms=total_time_ms,
                llm_provider=llm_provider
            )
        
        # 构建上下文
        context = self._build_context(retrieval_response.text_results)
        
        # 构建提示
        prompt = self._build_rag_prompt(
            text_query=request.text_query,
            image_query=request.image_path,
            context=context
        )
        
        # 生成回答
        generation_start_time = time.time()
        generated_answer = ""
        
        try:
            if llm_provider == "dashscope":
                generated_answer = await self._call_dashscope_api(prompt, context)
            elif llm_provider == "deepseek":
                generated_answer = await self._call_deepseek_api(prompt)
            elif llm_provider == "zhipu":
                generated_answer = await self._call_zhipu_api(prompt)
            else:
                generated_answer = "未支持的LLM提供商"
        except Exception as e:
            logger.error(f"LLM API调用失败: {e}")
            generated_answer = f"生成回答时出错: {str(e)}"
        
        generation_time_ms = (time.time() - generation_start_time) * 1000
        total_time_ms = (time.time() - rag_start_time) * 1000
        
        return RAGResponse(
            retrieval_response=retrieval_response,
            generated_answer=generated_answer,
            generation_time_ms=generation_time_ms,
            total_time_ms=total_time_ms,
            llm_provider=llm_provider
        )
    
    async def _call_dashscope_api(self, prompt: str, context: str = "") -> str:
        """调用DashScope API (qwen3-max)"""
        try:
            # 构建消息
            messages = [
                {
                    "role": "system", 
                    "content": "你是一个专业的校园导览助手。请基于提供的上下文信息，准确、详细地回答用户的问题。如果上下文信息不足，请诚实地说明。"
                },
                {
                    "role": "user",
                    "content": f"上下文信息：\n{context}\n\n用户问题：{prompt}\n\n请根据上下文信息回答问题："
                }
            ]
            
            # 使用DashScope生成回答
            response = self.dashscope.generate_chat_response(
                messages=messages,
                model="qwen3-max",
                temperature=0.7,
                top_p=0.9
            )
            
            return response
            
        except Exception as e:
            logger.error(f"DashScope API调用失败: {e}")
            return f"生成回答时出错: {str(e)}"
    
    async def _call_deepseek_api(self, prompt: str) -> str:
        """调用DeepSeek API"""
        # TODO: 实现DeepSeek API调用
        # 这里先返回模拟数据
        return f"DeepSeek回答模拟: 基于您的问题和上下文，我找到了相关信息..."
    
    async def _call_zhipu_api(self, prompt: str) -> str:
        """调用智谱API"""
        # TODO: 实现智谱API调用
        # 这里先返回模拟数据
        return f"智谱AI回答模拟: 根据检索到的信息，我可以为您提供以下回答..."
    
    def _build_context(self, results: List[SearchResult]) -> str:
        """构建上下文"""
        context_parts = []
        
        for i, result in enumerate(results[:3]):  # 使用前3个结果作为上下文
            context_parts.append(f"[文档 {i+1}, 相关性: {result.score:.3f}]:\n{result.content[:500]}...\n")
        
        return "\n".join(context_parts)
    
    def _build_rag_prompt(self, text_query: Optional[str], 
                         image_query: Optional[str], 
                         context: str) -> str:
        """构建RAG提示"""
        query_part = ""
        if text_query and image_query:
            query_part = f"用户提供了文本查询和图像查询：\n文本查询: {text_query}\n图像查询: 用户上传了一张图片"
        elif text_query:
            query_part = f"用户查询: {text_query}"
        elif image_query:
            query_part = f"用户提供了图像查询: 用户上传了一张图片"
        
        prompt = f"""基于以下检索到的上下文信息，回答用户的问题。

{query_part}

检索到的上下文信息：
{context}

请根据上下文信息提供准确、详细的回答。如果上下文信息不足以回答问题，请说明这一点。
回答时请引用相关的上下文信息。

回答："""
        
        return prompt
    
    async def train_relation_model(self, training_data_path: Optional[str] = None):
        """训练关联模型"""
        logger.info("开始训练关联模型...")
        
        # 收集训练数据
        # 1. 从数据库获取所有文档和图像
        doc_count = await self.vector_store.get_document_count()
        image_count = await self.vector_store.get_image_count()
        
        logger.info(f"数据库中有 {doc_count} 个文档和 {image_count} 个图像")
        
        # 2. 为每个文档-图像对生成向量并添加到训练集
        # 这里简化处理：假设所有文档和图像都有对应关系
        
        # 3. 训练模型
        training_pairs = self.training_collector.get_training_pairs()
        
        if len(training_pairs) < 10:
            logger.warning(f"训练数据不足 ({len(training_pairs)} 对)，需要更多数据")
            return False
        
        try:
            self.relation_model.train(training_pairs)
            self.relation_model.save()
            logger.info("关联模型训练完成并保存")
            return True
        except Exception as e:
            logger.error(f"关联模型训练失败: {e}")
            return False
    
    async def add_training_pair(self, text_id: str, image_id: str):
        """添加训练对"""
        try:
            # 获取文档内容
            # 这里需要从数据库获取文档和图像的向量
            # 简化处理：先生成向量
            
            # 1. 获取文档内容
            # 2. 生成文本向量
            # 3. 生成图像向量
            # 4. 添加到训练集
            
            logger.info(f"添加训练对: 文字ID={text_id}, 图片ID={image_id}")
            return True
        except Exception as e:
            logger.error(f"添加训练对失败: {e}")
            return False


# 工具函数
async def create_new_retriever() -> NewMultimodalRetriever:
    """创建新的检索器实例"""
    retriever = NewMultimodalRetriever()
    await retriever.initialize()
    return retriever


# 测试函数
async def test_new_retrieval():
    """测试新的检索系统"""
    print("="*60)
    print("测试新的多模态检索系统")
    print("="*60)
    
    retriever = await create_new_retriever()
    
    # 测试1: 纯文本查询
    print("\n测试1: 纯文本查询")
    request1 = RetrievalRequest(
        text_query="校园正门在哪里？",
        top_k=3
    )
    
    response1 = await retriever.retrieve(request1)
    print(f"  检索到 {len(response1.text_results)} 个文档")
    for i, result in enumerate(response1.text_results):
        print(f"  文档 {i+1}: {result.content[:100]}...")
    
    # 测试2: 纯图像查询
    print("\n测试2: 纯图像查询")
    request2 = RetrievalRequest(
        image_path="data/raw/images/本部_正门.png",
        top_k=3
    )
    
    response2 = await retriever.retrieve(request2)
    print(f"  检索到 {len(response2.image_results)} 个图像")
    print(f"  关联模型使用: {response2.relation_model_used}")
    
    # 测试3: RAG查询
    print("\n测试3: RAG查询 (使用DashScope)")
    rag_response = await retriever.retrieve_with_rag(request1, llm_provider="dashscope")
    print(f"  生成回答: {rag_response.generated_answer[:200]}...")
    print(f"  LLM提供商: {rag_response.llm_provider}")
    
    print("\n" + "="*60)
    print("测试完成")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(test_new_retrieval())