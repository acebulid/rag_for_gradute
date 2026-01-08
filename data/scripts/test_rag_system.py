#!/usr/bin/env python3
"""
测试RAG系统功能
测试文本检索、图像检索和多模态检索
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncio
import time
from src.services.retrieval_service import create_retriever, RetrievalRequest
from src.services.ollama_service import OllamaService

async def test_ollama_connection():
    """测试Ollama连接"""
    print("="*50)
    print("测试Ollama连接")
    print("="*50)
    
    try:
        ollama = OllamaService()
        
        # 检查模型是否可用
        models_to_check = ["bge-m3", "qwen2.5-vl", "qwen3:8b"]
        
        for model in models_to_check:
            available = ollama.check_model_available(model)
            status = "✅ 可用" if available else "❌ 不可用"
            print(f"{model}: {status}")
        
        # 测试文本嵌入生成
        print("\n🧪 测试文本嵌入生成...")
        try:
            embedding_result = ollama.generate_embedding("测试文本")
            print(f"  嵌入维度: {len(embedding_result.embedding)}")
            print(f"  模型: {embedding_result.model}")
            print(f"  耗时: {embedding_result.duration_ms:.2f}ms")
            print("  ✅ 文本嵌入生成成功")
        except Exception as e:
            print(f"  ❌ 文本嵌入生成失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ollama连接测试失败: {e}")
        return False

async def test_database_connection():
    """测试数据库连接"""
    print("\n" + "="*50)
    print("测试数据库连接")
    print("="*50)
    
    try:
        from src.database.vector_store import PostgreSQLVectorStore
        
        vector_store = PostgreSQLVectorStore()
        await vector_store.connect()
        
        # 检查表是否存在
        tables = await vector_store.check_tables_exist()
        print("数据库表状态:")
        for table, exists in tables.items():
            status = "✅ 存在" if exists else "❌ 不存在"
            print(f"  {table}: {status}")
        
        # 检查数据量
        doc_count = await vector_store.count_documents()
        image_count = await vector_store.count_image_descriptions()
        relation_count = await vector_store.count_relations()
        
        print(f"\n数据统计:")
        print(f"  文档数量: {doc_count}")
        print(f"  图像描述数量: {image_count}")
        print(f"  关联数量: {relation_count}")
        
        await vector_store.disconnect()
        return True
        
    except Exception as e:
        print(f"❌ 数据库连接测试失败: {e}")
        return False

async def test_text_retrieval():
    """测试文本检索"""
    print("\n" + "="*50)
    print("测试文本检索")
    print("="*50)
    
    try:
        retriever = await create_retriever()
        
        # 测试查询
        test_queries = [
            "学生活动中心有什么功能？",
            "正门附近有什么建筑？",
            "图书馆在哪里？"
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

async def test_image_retrieval():
    """测试图像检索"""
    print("\n" + "="*50)
    print("测试图像检索")
    print("="*50)
    
    try:
        retriever = await create_retriever()
        
        # 使用现有的图像文件
        image_path = "data/raw/images/本部_正门.png"
        if not Path(image_path).exists():
            print(f"❌ 图像文件不存在: {image_path}")
            return False
        
        print(f"使用图像: {image_path}")
        request = RetrievalRequest(image_path=image_path, top_k=3)
        
        start_time = time.time()
        response = await retriever.retrieve(request)
        elapsed_ms = (time.time() - start_time) * 1000
        
        print(f"检索耗时: {elapsed_ms:.2f}ms")
        print(f"检索到 {len(response.hybrid_results)} 个结果")
        
        if response.hybrid_results:
            for i, result in enumerate(response.hybrid_results[:2]):
                print(f"结果 {i+1}: 相关性 {result.score:.3f}")
                preview = result.content[:100] + "..." if len(result.content) > 100 else result.content
                print(f"  内容: {preview}")
        else:
            print("未找到相关结果")
        
        return True
        
    except Exception as e:
        print(f"❌ 图像检索测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """主测试函数"""
    print("多模态RAG系统功能测试")
    print("="*60)
    
    tests = [
        ("Ollama连接", test_ollama_connection),
        ("数据库连接", test_database_connection),
        ("文本检索", test_text_retrieval),
        ("RAG生成", test_rag_generation),
        ("图像检索", test_image_retrieval),
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
    print("RAG系统测试总结")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{test_name}: {status}")
    
    print(f"\n总计: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("\n🎉 所有RAG系统测试通过！")
        print("\n系统功能正常，可以开始使用。")
        print("\n下一步建议:")
        print("1. 启动API服务器: python -m src.api.main")
        print("2. 访问 http://localhost:8000/docs 查看API文档")
        print("3. 使用API进行多模态检索")
        return 0
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败")
        print("\n需要检查的问题:")
        print("1. Ollama服务是否运行正常")
        print("2. 数据库连接是否正常")
        print("3. 数据是否已导入")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))