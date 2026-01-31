"""
chainC: 关键词检索链
输入关键词，使用qwen3:8b模型生成关键词向量，检索向量数据库，返回最接近的2个文本段
"""

import asyncio
from typing import List, Dict, Any
from function.api_function import OllamaService, generate_embedding
from function.database_function import PostgreSQLVectorStore, SearchResult


class ChainC:
    """chainC: 关键词检索链"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.top_k = self.config.get('top_k', 2)  # 默认返回2个结果
        self.similarity_threshold = self.config.get('similarity_threshold', 0.3)
    
    async def process_keywords(self, keywords: List[str]) -> List[SearchResult]:
        """
        处理关键词，检索相关文本段
        
        参数:
            keywords: 关键词列表
        
        返回:
            List[SearchResult]: 检索到的文本段结果
        """
        print(f"[chainC] 开始处理关键词: {keywords}")
        
        vector_store = None
        try:
            # 1. 连接向量数据库
            vector_store = PostgreSQLVectorStore()
            await vector_store.connect()
            
            # 2. 生成关键词向量
            keyword_embeddings = []
            for keyword in keywords:
                print(f"[chainC] 生成关键词向量: '{keyword}'")
                embedding = await generate_embedding(keyword)
                if embedding:
                    keyword_embeddings.append(embedding)
                    print(f"[chainC] 关键词 '{keyword}' 向量生成成功，维度: {len(embedding)}")
                else:
                    print(f"[chainC] 关键词 '{keyword}' 向量生成失败")
            
            if not keyword_embeddings:
                print("[chainC] 所有关键词向量生成失败")
                return []
            
            # 3. 数据库检索
            all_results = []
            for i, embedding in enumerate(keyword_embeddings):
                print(f"[chainC] 使用关键词{i+1}向量进行检索...")
                results = await vector_store.search_similar_documents(
                    query_embedding=embedding,
                    top_k=self.top_k,
                    threshold=self.similarity_threshold
                )
                print(f"[chainC] 关键词{i+1}检索到 {len(results)} 个结果")
                all_results.extend(results)
            
            # 4. 去重和排序，只返回前top_k个
            final_results = vector_store._deduplicate_and_sort_results(all_results)[:self.top_k]
            print(f"[chainC] 最终检索到 {len(final_results)} 个相关文本段")
            return final_results
            
        except Exception as e:
            print(f"[chainC] 处理失败: {e}")
            return []
        
        finally:
            # 清理资源
            if vector_store:
                await vector_store.close()
    
    async def process_with_debug(self, keywords: List[str]) -> Dict[str, Any]:
        """
        处理关键词并返回调试信息
        
        参数:
            keywords: 关键词列表
        
        返回:
            Dict[str, Any]: 包含检索结果和调试信息
        """
        print(f"[chainC调试] 开始处理关键词: {keywords}")
        
        result = {
            "success": False,
            "keywords": keywords,
            "search_results": [],
            "error": None
        }
        
        vector_store = None
        try:
            # 1. 连接向量数据库
            vector_store = PostgreSQLVectorStore()
            await vector_store.connect()
            
            # 2. 生成关键词向量
            keyword_embeddings = []
            for keyword in keywords:
                embedding = await generate_embedding(keyword)
                if embedding:
                    keyword_embeddings.append(embedding)
                    print(f"[chainC调试] 关键词 '{keyword}' 向量生成成功，维度: {len(embedding)}")
                else:
                    print(f"[chainC调试] 关键词 '{keyword}' 向量生成失败")
            
            if not keyword_embeddings:
                result["error"] = "所有关键词向量生成失败"
                return result
            
            # 3. 数据库检索
            all_results = []
            for i, embedding in enumerate(keyword_embeddings):
                results = await vector_store.search_similar_documents(
                    query_embedding=embedding,
                    top_k=self.top_k,
                    threshold=self.similarity_threshold
                )
                all_results.extend(results)
            
            # 4. 去重和排序
            final_results = vector_store._deduplicate_and_sort_results(all_results)[:self.top_k]
            
            # 格式化结果
            result["search_results"] = [
                {
                    "id": r.id,
                    "content_preview": r.content[:200] + "..." if len(r.content) > 200 else r.content,
                    "score": r.score,
                    "source": r.source
                }
                for r in final_results
            ]
            
            result["success"] = True
            print(f"[chainC调试] 检索到 {len(final_results)} 个相关文本段")
            
        except Exception as e:
            result["error"] = str(e)
            print(f"[chainC调试] 处理失败: {e}")
        
        finally:
            if vector_store:
                await vector_store.close()
        
        return result


if __name__ == "__main__":
    print("chainC模块 - 关键词检索链")
    print("使用方法: 作为模块导入使用，不直接运行")
    print("示例: from chain.chainC import ChainC")
