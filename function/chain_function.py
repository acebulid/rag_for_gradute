"""
chain功能函数 - 实现chain路由、中间结果打印、最终结果打印
使用chain文件夹下的chainA、chainB和chainC
"""

import os
import asyncio
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

# 导入chainA、chainB和chainC
from chain.chainA import process_image_query
from chain.chainB import ChainB
from chain.chainC import ChainC

# 复用现有function
from function.database_function import PostgreSQLVectorStore, SearchResult
from function.api_function import OllamaService


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
        
        # chain配置
        self.chainB_config = {}
        self.chainC_config = {
            'top_k': 2,
            'similarity_threshold': 0.3
        }


# ========== 中间结果打印函数 ==========
def print_intermediate_results(chain_type: str, step: str, data: Any):
    """
    打印中间结果
    
    参数:
        chain_type: chain类型 ('A', 'B', 'C')
        step: 步骤描述
        data: 要打印的数据
    """
    print(f"[chain{chain_type}中间结果] {step}")
    
    if isinstance(data, list):
        if data and isinstance(data[0], str):
            print(f"  关键词列表: {data}")
        elif data and isinstance(data[0], dict):
            for i, item in enumerate(data, 1):
                print(f"  结果{i}: {item}")
    elif isinstance(data, dict):
        for key, value in data.items():
            print(f"  {key}: {value}")
    else:
        print(f"  {data}")


def print_final_results(search_results: List[SearchResult]):
    """
    打印最终返回结果
    
    参数:
        search_results: 搜索结果列表
    """
    print(f"[chain最终结果] 检索到 {len(search_results)} 个相关文本段")
    
    for i, result in enumerate(search_results, 1):
        print(f"\n  [{i}] 相似度: {result.score:.3f}")
        print(f"      内容: {result.content[:200]}..." if len(result.content) > 200 else f"      内容: {result.content}")
        if result.source:
            print(f"      来源: {result.source}")


# ========== chain路由函数 ==========
async def chain_router(
    query_type: str,
    query_content: Union[str, Dict[str, Any]],
    config: ChainConfig = None
) -> List[SearchResult]:
    """
    chain路由函数，选择进入A或者B，打印中间结果后，再进入C，打印最终返回结果
    
    参数:
        query_type: 查询类型 ('image', 'text')
        query_content: 查询内容
        config: chain配置
    
    返回:
        List[SearchResult]: 搜索结果
    """
    if config is None:
        config = ChainConfig()
    
    print(f"[chain路由] 开始处理{query_type}查询")
    
    if query_type == 'image':
        # chainA: 图片 → bge模型 → 图片编码 → 模型A → 关键词
        if isinstance(query_content, str):
            image_path = query_content
        elif isinstance(query_content, dict):
            image_path = query_content.get('image_path', '')
        else:
            raise ValueError("图片查询内容必须是字符串或字典")
        
        # 1. chainA: 处理图片查询
        print_intermediate_results('A', '开始处理图片查询', {'image_path': image_path})
        chainA_result = await process_image_query(image_path)
        print_intermediate_results('A', '图片查询完成', {
            'success': chainA_result['success'],
            'keyword': chainA_result.get('keyword'),
            'confidence': chainA_result.get('confidence')
        })
        
        if not chainA_result['success']:
            print(f"[chain路由] chainA处理失败: {chainA_result.get('error')}")
            return []
        
        keyword = chainA_result['keyword']
        if not keyword:
            print("[chain路由] chainA未提取到关键词")
            return []
        
        # 2. chainC: 检索相关文本段
        print_intermediate_results('C', '开始检索相关文本段', {'keyword': keyword})
        chainC = ChainC(config.chainC_config)
        search_results = await chainC.process_keywords([keyword])
        
        return search_results
    
    elif query_type == 'text':
        # chainB + chainC流程
        if isinstance(query_content, str):
            question = query_content
        elif isinstance(query_content, dict):
            question = query_content.get('question', '')
        else:
            raise ValueError("文本查询内容必须是字符串或字典")
        
        # 1. chainB: 提取关键词
        print_intermediate_results('B', '开始提取关键词', {'question': question})
        chainB = ChainB(config.chainB_config)
        keywords = await chainB.process_question(question)
        print_intermediate_results('B', '关键词提取完成', keywords)
        
        if not keywords:
            print("[chain路由] chainB未提取到关键词")
            return []
        
        # 2. chainC: 检索相关文本段
        print_intermediate_results('C', '开始检索相关文本段', {'keywords': keywords})
        chainC = ChainC(config.chainC_config)
        search_results = await chainC.process_keywords(keywords)
        
        return search_results
    
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
    使用chain路由函数，打印中间结果（最终结果由调用者打印）
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
    
    try:
        # 1. 路由处理
        search_results = await chain_router(query_type, query_content, config)
        
        # 2. 格式化结果（不打印最终结果，由调用者打印）
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
        print(f"   测试图片: {test_image}")
        result = await process_chain_query(
            query_type='image',
            query_content=test_image,
            config=config
        )
        
        if result["success"]:
            print(f"   查询成功")
            print(f"   检索到 {len(result['search_results'])} 个结果")
            if result['search_results']:
                print(f"   第一个结果相似度: {result['search_results'][0]['score']:.3f}")
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