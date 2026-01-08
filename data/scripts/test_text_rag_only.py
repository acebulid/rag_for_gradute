#!/usr/bin/env python3
"""
只测试文本RAG功能
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncio
import time
from src.services.retrieval_service import create_retriever, RetrievalRequest

async def test_text_retrieval():
    """测试文本检索"""
    print("="*50)
    print("测试文本检索功能")
    print("="*50)
    
    try:
        retriever = await create_retriever()
        
        # 测试查询
        test_queries = [
            "学生活动中心有什么功能？",
            "正门附近有什么建筑？",
            "爬山虎覆盖的建筑在哪里？"
        ]
        
        for query in test_queries:
            print(f"\n查询: '{query}'")
            request = RetrievalRequest(text_query=query, top_k=3)
            
            start_time = time.time()
            response = await retriever.retrieve(request)
            elapsed_ms = (time.time() - start_time) * 1000
            
            print(f"  检索耗时: {elapsed_ms:.2f}ms")
            print(f"  检索到 {len(response.hybrid_results)} 个结果")
            
            if response.hybrid_results:
                for i, result in enumerate(response.hybrid_results[:2]):  # 显示前2个结果
                    print(f"  结果 {i+1}: 相关性 {result.score:.3f}")
                    # 显示前100个字符
                    preview = result.content[:100] + "..." if len(result.content) > 100 else result.content
                    print(f"    内容: {preview}")
            else:
                print("  未找到相关结果")
        
        return True
        
    except Exception as e:
        print(f"❌ 文本检索测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_rag_generation():
    """测试RAG生成"""
    print("\n" + "="*50)
    print("测试RAG生成")
    print("="*50)
    
    try:
        retriever = await create_retriever()
        
        # 测试查询
        test_query = "学生活动中心有什么功能？请详细介绍一下。"
        print(f"查询: '{test_query}'")
        
        request = RetrievalRequest(text_query=test_query, top_k=3)
        
        start_time = time.time()
        response = await retriever.retrieve_with_rag(request)
        elapsed_ms = (time.time() - start_time) * 1000
        
        print(f"\n总耗时: {elapsed_ms:.2f}ms")
        print(f"检索耗时: {response.retrieval_response.response_time_ms:.2f}ms")
        print(f"生成耗时: {response.generation_time_ms:.2f}ms")
        
        print(f"\n检索到 {len(response.retrieval_response.hybrid_results)} 个结果")
        
        print("\n生成的回答:")
        print("-"*50)
        print(response.generated_answer)
        print("-"*50)
        
        return True
        
    except Exception as e:
        print(f"❌ RAG生成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_database_stats():
    """测试数据库统计"""
    print("\n" + "="*50)
    print("测试数据库统计")
    print("="*50)
    
    try:
        from src.database.vector_store import PostgreSQLVectorStore
        
        vector_store = PostgreSQLVectorStore()
        await vector_store.connect()
        
        # 检查数据量
        doc_count = await vector_store.get_document_count()
        image_count = await vector_store.get_image_count()
        
        print(f"文档数量: {doc_count}")
        print(f"图像描述数量: {image_count}")
        
        if doc_count > 0:
            print("\n📝 文档示例:")
            test_embedding = [0.1] * 1024
            results = await vector_store.search_similar_documents(test_embedding, top_k=1)
            if results:
                result = results[0]
                print(f"ID: {result.id}")
                print(f"内容预览: {result.content[:200]}...")
        
        await vector_store.close()
        return True
        
    except Exception as e:
        print(f"❌ 数据库统计测试失败: {e}")
        return False

async def main():
    """主测试函数"""
    print("文本RAG系统功能测试")
    print("="*60)
    
    tests = [
        ("数据库统计", test_database_stats),
        ("文本检索", test_text_retrieval),
        ("RAG生成", test_rag_generation),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            print(f"\n▶️  开始测试: {test_name}")
            success = await test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"{test_name}测试异常: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # 打印总结
    print("\n" + "="*60)
    print("文本RAG系统测试总结")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{test_name}: {status}")
    
    print(f"\n总计: {passed}/{total} 个测试通过")
    
    if passed == total:
