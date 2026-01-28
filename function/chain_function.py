"""
chain功能函数 - 实现chainA（图片检索）和chainB（文本检索）流程
完全使用api_function中的函数，不重复实现已有功能
"""

import os
import asyncio
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import numpy as np

# 复用现有function
from function.database_function import PostgreSQLVectorStore, SearchResult, print_database_vectors
from function.model_function import load_model, predict_single, ModelAConfig
from function.api_function import OllamaService, generate_embedding, extract_keywords, generate_text


# ========== chain配置 ==========
class ChainConfig:
    """chain配置，复用环境变量"""
    
    def __init__(self):
        # Ollama配置（从.env读取）
        self.ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        self.embedding_model = os.getenv('EMBEDDING_MODEL', 'bge-m3:latest')
        self.llm_model = os.getenv('LLM_MODEL', 'qwen3:8b')
        
        # 检索配置
        self.top_k = 2  # 只返回前两个文本段
        self.similarity_threshold = 0.3  # 降低阈值以提高检索率
        
        # 模型A配置（从.env读取，优先使用model/目录下的模型）
        self.model_a_path = os.getenv('MODEL_A_PATH', 'model/model_a.pth')
        self.model_a_config_path = os.getenv('MODEL_A_CONFIG_PATH', 'model/model_a_config.json')
        self.model_a_scaler_path = os.getenv('MODEL_A_SCALER_PATH', 'model/model_a_scaler.pkl')
        
        # 如果model目录下的模型不存在，使用data/models/目录下的模型
        if not Path(self.model_a_path).exists():
            self.model_a_path = "data/models/model_a.pth"
            self.model_a_config_path = "data/models/model_a_config.json"
            self.model_a_scaler_path = "data/models/model_a_scaler.pkl"


# ========== 辅助函数 ==========
async def get_image_description(image_path: str, ollama_service: OllamaService) -> str:
    """
    获取图片描述（使用VLM模型）
    注意：当前Ollama的bge模型只支持文本，这里简化处理
    实际应该使用VLM模型生成图片描述
    """
    # 简化：使用文件名作为描述
    # 实际应该调用VLM模型：image_path → VLM → 图片描述
    return f"校园导览图片: {Path(image_path).name}"


async def get_image_embedding(image_path: str, ollama_service: OllamaService) -> List[float]:
    """
    获取图片向量（chainA第一步）
    图片 → bge模型 → 图片编码
    """
    print(f"[chainA] 获取图片向量: {image_path}")
    
    # 1. 获取图片描述
    image_description = await get_image_description(image_path, ollama_service)
    
    # 2. 使用bge模型生成图片向量
    image_embedding = await ollama_service.generate_embedding(image_description)
    
    if image_embedding:
        print(f"[chainA] 图片向量生成成功，维度: {len(image_embedding)}")
    else:
        print("[chainA] 图片向量生成失败")
    
    return image_embedding


async def get_meaning_vector(image_embedding: List[float], config: ChainConfig) -> List[float]:
    """
    获取含义向量（chainA第二步）
    图片编码 → 模型A → 含义向量
    """
    print("[chainA] 获取含义向量")
    
    try:
        # 1. 加载模型A
        model_config = ModelAConfig()
        model, scaler, device = load_model(model_config, config.model_a_path)
        
        # 2. 预测含义向量
        meaning_vector = predict_single(
            model, scaler, device, 
            np.array(image_embedding), 
            verbose=False
        )
        
        print(f"[chainA] 含义向量生成成功，维度: {len(meaning_vector)}")
        return meaning_vector.tolist()
        
    except Exception as e:
        print(f"[chainA] 获取含义向量失败: {e}")
        return []


async def extract_question_keywords(question: str, ollama_service: OllamaService) -> List[str]:
    """
    提取问题关键词（chainB第一步）
    提问 → qwen3:8b模型 → 关键词
    """
    print(f"[chainB] 提取问题关键词: {question}")
    
    # 使用qwen3:8b模型提取关键词
    keywords = await ollama_service.extract_keywords(question)
    
    if keywords:
        print(f"[chainB] 提取到关键词: {keywords}")
    else:
        print("[chainB] 关键词提取失败，使用原始问题")
        keywords = [question]
    
    return keywords


async def get_keyword_embeddings(keywords: List[str], ollama_service: OllamaService) -> List[List[float]]:
    """
    获取关键词向量（chainB第二步）
    关键词 → bge模型 → 关键词向量
    """
    print(f"[chainB] 获取关键词向量: {keywords}")
    
    keyword_embeddings = []
    for keyword in keywords:
        embedding = await ollama_service.generate_embedding(keyword)
        if embedding:
            keyword_embeddings.append(embedding)
            print(f"[chainB] 关键词 '{keyword}' 向量生成成功，维度: {len(embedding)}")
        else:
            print(f"[chainB] 关键词 '{keyword}' 向量生成失败")
    
    return keyword_embeddings


# ========== chainA: 图片检索链 ==========
async def chainA_process_image(
    image_path: str,
    vector_store: PostgreSQLVectorStore,
    ollama_service: OllamaService,
    config: ChainConfig
) -> List[SearchResult]:
    """
    chainA完整流程：
    图片 → bge模型 → 图片编码 → 模型A → 含义向量 → 向量数据库 → 文本段
    """
    print(f"[chainA] 开始处理图片: {image_path}")
    
    try:
        # 1. 获取图片向量（bge模型）
        print(f"[chainA调试] 步骤1: 获取图片向量")
        image_embedding = await get_image_embedding(image_path, ollama_service)
        if not image_embedding:
            print(f"[chainA调试] 图片向量获取失败")
            return []
        print(f"[chainA调试] 图片向量维度: {len(image_embedding)}")
        print(f"[chainA调试] 图片向量前5个值: {image_embedding[:5]}")
        
        # 2. 获取含义向量（模型A）
        print(f"[chainA调试] 步骤2: 获取含义向量")
        meaning_vector = await get_meaning_vector(image_embedding, config)
        if not meaning_vector:
            print(f"[chainA调试] 含义向量获取失败")
            return []
        print(f"[chainA调试] 含义向量维度: {len(meaning_vector)}")
        print(f"[chainA调试] 含义向量前5个值: {meaning_vector[:5]}")
        
        # 3. 数据库检索
        print(f"[chainA调试] 步骤3: 数据库检索")
        search_results = await vector_store.search_similar_documents(
            query_embedding=meaning_vector,
            top_k=config.top_k,
            threshold=config.similarity_threshold
        )
        
        print(f"[chainA] 检索到 {len(search_results)} 个相关文本段")
        return search_results
        
    except Exception as e:
        print(f"[chainA] 处理失败: {e}")
        return []


# ========== chainB: 文本检索链 ==========
async def chainB_process_question(
    question: str,
    vector_store: PostgreSQLVectorStore,
    ollama_service: OllamaService,
    config: ChainConfig
) -> List[SearchResult]:
    """
    chainB完整流程：
    提问 → qwen3:8b模型 → 关键词 → bge模型 → 关键词向量 → 向量数据库 → 文本段
    """
    print(f"[chainB] 开始处理问题: {question}")
    
    try:
        # 1. 提取关键词（qwen3:8b模型）
        keywords = await extract_question_keywords(question, ollama_service)
        
        # 2. 获取关键词向量（bge模型）
        keyword_embeddings = await get_keyword_embeddings(keywords, ollama_service)
        if not keyword_embeddings:
            return []
        
        # 调试：打印关键词向量（注释掉）
        # print(f"[chainB调试] 关键词向量信息:")
        # for i, (keyword, embedding) in enumerate(zip(keywords, keyword_embeddings)):
        #     if embedding:
        #         print(f"  关键词{i+1}: '{keyword}'")
        #         print(f"    向量维度: {len(embedding)}")
        #         print(f"    向量前5个值: {embedding[:5]}")
        #         print(f"    向量范数: {sum(x*x for x in embedding)**0.5:.4f}")
        
        # 调试：打印数据库向量（注释掉）
        # await print_database_vectors(vector_store, limit=3)
        
        # 3. 数据库检索
        all_results = []
        for i, embedding in enumerate(keyword_embeddings):
            # print(f"[chainB调试] 使用关键词{i+1}向量进行检索...")
            results = await vector_store.search_similar_documents(
                query_embedding=embedding,
                top_k=config.top_k,
                threshold=config.similarity_threshold
            )
            # print(f"[chainB调试] 关键词{i+1}检索到 {len(results)} 个结果")
            all_results.extend(results)
        
        # 4. 去重和排序
        final_results = vector_store._deduplicate_and_sort_results(all_results)[:config.top_k]
        print(f"[chainB] 检索到 {len(final_results)} 个相关文本段")
        return final_results
        
    except Exception as e:
        print(f"[chainB] 处理失败: {e}")
        return []


# ========== chain路由函数 ==========
async def chain_router(
    query_type: str,
    query_content: Union[str, Dict[str, Any]],
    vector_store: PostgreSQLVectorStore,
    ollama_service: OllamaService,
    config: ChainConfig
) -> List[SearchResult]:
    """
    chain路由函数，选择A链还是B链
    复用chainA和chainB的处理函数
    """
    if query_type == 'image':
        # chainA: 图片检索
        if isinstance(query_content, str):
            image_path = query_content
        elif isinstance(query_content, dict):
            image_path = query_content.get('image_path', '')
        else:
            raise ValueError("图片查询内容必须是字符串路径或字典")
        
        return await chainA_process_image(image_path, vector_store, ollama_service, config)
    
    elif query_type == 'text':
        # chainB: 文本检索
        if isinstance(query_content, str):
            question = query_content
        elif isinstance(query_content, dict):
            question = query_content.get('question', '')
        else:
            raise ValueError("文本查询内容必须是字符串或字典")
        
        return await chainB_process_question(question, vector_store, ollama_service, config)
    
    else:
        raise ValueError(f"不支持的查询类型: {query_type}")


# ========== 主处理函数 ==========
async def process_chain_query(
    query_type: str,
    query_content: Union[str, Dict[str, Any]],
    config: ChainConfig = None
) -> Dict[str, Any]:
    """
    处理chain查询的主函数
    使用api_function中的Ollama服务
    """
    # 使用默认配置
    if config is None:
        config = ChainConfig()
    
    result = {
        "success": False,
        "query_type": query_type,
        "search_results": [],
        "error": None
    }
    
    vector_store = None
    ollama_service = None
    
    try:
        # 1. 连接向量数据库（复用database_function）
        vector_store = PostgreSQLVectorStore()
        await vector_store.connect()
        
        # 2. 初始化Ollama服务（使用api_function中的OllamaService）
        ollama_service = OllamaService()
        
        # 3. 使用异步上下文管理器
        async with ollama_service:
            # 4. 路由处理
            search_results = await chain_router(
                query_type, query_content, vector_store, ollama_service, config
            )
            
            # 5. 格式化结果
            result["search_results"] = [
                {
                    "id": r.id,
                    "content_preview": r.content[:200] + "..." if len(r.content) > 200 else r.content,
                    "score": r.score,
                    "source": r.source
                }
                for r in search_results
            ]
            
            result["success"] = True
        
    except Exception as e:
        result["error"] = str(e)
        print(f"[chain] 处理查询失败: {e}")
    
    finally:
        # 清理资源
        if vector_store:
            await vector_store.close()
    
    return result


# ========== 测试函数 ==========
async def test_chain_function():
    """测试chain功能，复用现有测试模式"""
    print("=" * 70)
    print("测试chain功能")
    print("=" * 70)
    
    config = ChainConfig()
    
    # 测试chainB（文本查询）
    print("\n1. 测试chainB（文本查询）...")
    result = await process_chain_query(
        query_type='text',
        query_content='首都师范大学的校门在哪里？',
        config=config
    )
    
    if result["success"]:
        print(f"   查询成功")
        print(f"   检索到 {len(result['search_results'])} 个结果")
        if result['search_results']:
            print(f"   第一个结果相似度: {result['search_results'][0]['score']:.3f}")
    else:
        print(f"   查询失败: {result['error']}")
    
    # 测试chainA（图片查询）
    print("\n2. 测试chainA（图片查询）...")
    test_image = "data/image-1.png" if Path("data/image-1.png").exists() else None
    if test_image:
        result = await process_chain_query(
            query_type='image',
            query_content=test_image,
            config=config
        )
        
        if result["success"]:
            print(f"   查询成功")
            print(f"   检索到 {len(result['search_results'])} 个结果")
        else:
            print(f"   查询失败: {result['error']}")
    else:
        print("   测试图片不存在，跳过chainA测试")
    
    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)


if __name__ == "__main__":
    # 运行测试，复用scripts中的调用模式
    asyncio.run(test_chain_function())