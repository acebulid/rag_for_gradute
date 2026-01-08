#!/usr/bin/env python3
"""
服务层测试
测试Ollama服务和检索服务
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_ollama_service_import():
    """测试Ollama服务导入"""
    print("="*50)
    print("测试Ollama服务导入")
    print("="*50)
    
    try:
        from src.services.ollama_service import (
            OllamaService, EmbeddingResult, GenerationResult
        )
        
        print("✅ Ollama服务导入成功")
        print(f"   服务类: {OllamaService}")
        print(f"   嵌入结果: {EmbeddingResult}")
        print(f"   生成结果: {GenerationResult}")
        
        return True
    except Exception as e:
        print(f"❌ Ollama服务导入失败: {e}")
        return False

def test_ollama_service_structure():
    """测试Ollama服务结构"""
    print("\n" + "="*50)
    print("测试Ollama服务结构")
    print("="*50)
    
    try:
        from src.services.ollama_service import OllamaService
        
        # 检查服务方法
        service_methods = [
            method for method in dir(OllamaService) 
            if not method.startswith('_') and callable(getattr(OllamaService, method))
        ]
        
        print("✅ Ollama服务方法:")
        for method in service_methods:
            print(f"   - {method}")
        
        # 检查必需的方法
        required_methods = [
            'generate_embedding',
            'generate_image_description', 
            'generate_text',
            'batch_generate_embeddings'
        ]
        
        missing_methods = []
        for method in required_methods:
            if method not in service_methods:
                missing_methods.append(method)
        
        if missing_methods:
            print(f"❌ 缺少方法: {', '.join(missing_methods)}")
            return False
        
        print("✅ 所有必需方法都存在")
        return True
        
    except Exception as e:
        print(f"❌ 服务结构检查失败: {e}")
        return False

def test_retrieval_service_import():
    """测试检索服务导入"""
    print("\n" + "="*50)
    print("测试检索服务导入")
    print("="*50)
    
    try:
        from src.services.retrieval_service import (
            MultimodalRetriever, RetrievalRequest, RetrievalResponse, RAGResponse
        )
        
        print("✅ 检索服务导入成功")
        print(f"   检索器: {MultimodalRetriever}")
        print(f"   检索请求: {RetrievalRequest}")
        print(f"   检索响应: {RetrievalResponse}")
        print(f"   RAG响应: {RAGResponse}")
        
        return True
    except Exception as e:
        print(f"❌ 检索服务导入失败: {e}")
        return False

def test_retrieval_service_structure():
    """测试检索服务结构"""
    print("\n" + "="*50)
    print("测试检索服务结构")
    print("="*50)
    
    try:
        from src.services.retrieval_service import MultimodalRetriever
        
        # 检查检索器方法
        retriever_methods = [
            method for method in dir(MultimodalRetriever) 
            if not method.startswith('_') and callable(getattr(MultimodalRetriever, method))
        ]
        
        print("✅ 检索器方法:")
        for method in retriever_methods:
            print(f"   - {method}")
        
        # 检查必需的方法
        required_methods = [
            'retrieve',
            'retrieve_with_rag'
        ]
        
        missing_methods = []
        for method in required_methods:
            if method not in retriever_methods:
                missing_methods.append(method)
        
        if missing_methods:
            print(f"❌ 缺少方法: {', '.join(missing_methods)}")
            return False
        
        print("✅ 所有必需方法都存在")
        return True
        
    except Exception as e:
        print(f"❌ 检索服务结构检查失败: {e}")
        return False

def test_vector_store_import():
    """测试向量存储导入"""
    print("\n" + "="*50)
    print("测试向量存储导入")
    print("="*50)
    
    try:
        from src.database.vector_store import (
            PostgreSQLVectorStore, SearchResult, ImageSearchResult
        )
        
        print("✅ 向量存储导入成功")
        print(f"   向量存储: {PostgreSQLVectorStore}")
        print(f"   搜索结果: {SearchResult}")
        print(f"   图像搜索结果: {ImageSearchResult}")
        
        return True
    except Exception as e:
        print(f"❌ 向量存储导入失败: {e}")
        return False

def main():
    """主测试函数"""
    print("服务层测试")
    print("="*60)
    
    tests = [
        ("Ollama服务导入", test_ollama_service_import),
        ("Ollama服务结构", test_ollama_service_structure),
        ("检索服务导入", test_retrieval_service_import),
        ("检索服务结构", test_retrieval_service_structure),
        ("向量存储导入", test_vector_store_import),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"{test_name}测试异常: {e}")
            results.append((test_name, False))
    
    # 打印总结
    print("\n" + "="*60)
    print("服务层测试总结")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{test_name}: {status}")
    
    print(f"\n总计: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("\n🎉 所有服务层测试通过！")
        print("\n注意:")
        print("1. 这些测试只验证了导入和结构")
        print("2. 实际功能测试需要:")
        print("   - PostgreSQL服务运行")
        print("   - Ollama服务运行 (ollama serve)")
        print("   - 相应的模型已下载")
        return 0
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败")
        print("\n可能的原因:")
        print("1. 缺少依赖 (运行: pip install -r requirements.txt)")
        print("2. 代码语法错误")
        print("3. 导入路径问题")
        return 1

if __name__ == "__main__":
    sys.exit(main())