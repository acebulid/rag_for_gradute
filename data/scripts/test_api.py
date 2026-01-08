#!/usr/bin/env python3
"""
API测试
测试FastAPI应用和路由
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_api_import():
    """测试API导入"""
    print("="*50)
    print("测试API模块导入")
    print("="*50)
    
    try:
        from src.api.main import app
        from src.api.schemas import QueryRequest, RAGResponse
        from src.api.routers.rag import router
        
        print("✅ API模块导入成功")
        print(f"   FastAPI应用: {app}")
        print(f"   查询请求模型: {QueryRequest}")
        print(f"   RAG响应模型: {RAGResponse}")
        print(f"   RAG路由: {router}")
        
        return True
    except Exception as e:
        print(f"❌ API模块导入失败: {e}")
        return False

def test_fastapi_app():
    """测试FastAPI应用"""
    print("\n" + "="*50)
    print("测试FastAPI应用")
    print("="*50)
    
    try:
        from src.api.main import app
        
        print("✅ FastAPI应用检查:")
        print(f"   应用标题: {app.title}")
        print(f"   应用描述: {app.description[:100]}...")
        print(f"   应用版本: {app.version}")
        
        # 检查路由
        routes = [route for route in app.routes if hasattr(route, 'path')]
        print(f"   路由数量: {len(routes)}")
        
        # 显示主要路由
        print("\n   主要路由:")
        rag_routes = [route for route in routes if '/rag' in route.path]
        for route in rag_routes[:5]:  # 显示前5个
            methods = getattr(route, 'methods', ['GET'])
            print(f"     {list(methods)[0]} {route.path}")
        
        if len(rag_routes) > 5:
            print(f"     ... 还有 {len(rag_routes) - 5} 个路由")
        
        return True
    except Exception as e:
        print(f"❌ FastAPI应用检查失败: {e}")
        return False

def test_schemas():
    """测试Pydantic模型"""
    print("\n" + "="*50)
    print("测试Pydantic模型")
    print("="*50)
    
    try:
        from src.api.schemas import (
            QueryRequest, QueryResponse, RAGResponse,
            BatchQueryRequest, SystemStatus, HealthCheck
        )
        
        print("✅ Pydantic模型检查:")
        
        # 测试QueryRequest模型
        test_request = QueryRequest(
            text_query="图书馆在哪里？",
            top_k=5,
            threshold=0.3
        )
        print(f"   查询请求模型: {test_request.dict()}")
        
        # 测试HealthCheck模型
        test_health = HealthCheck(
            status="healthy",
            version="1.0.0"
        )
        print(f"   健康检查模型: {test_health.dict()}")
        
        return True
    except Exception as e:
        print(f"❌ Pydantic模型检查失败: {e}")
        return False

def test_routes():
    """测试路由"""
    print("\n" + "="*50)
    print("测试路由")
    print("="*50)
    
    try:
        from src.api.routers.rag import router
        
        print("✅ 路由检查:")
        print(f"   路由前缀: {router.prefix}")
        print(f"   路由标签: {router.tags}")
        
        # 检查路由端点
        endpoints = []
        for route in router.routes:
            if hasattr(route, 'path'):
                path = route.path
                methods = getattr(route, 'methods', ['GET'])
                endpoint = f"{list(methods)[0]} {path}"
                endpoints.append(endpoint)
        
        print(f"   端点数量: {len(endpoints)}")
        print("\n   端点列表:")
        for endpoint in endpoints:
            print(f"     {endpoint}")
        
        return True
    except Exception as e:
        print(f"❌ 路由检查失败: {e}")
        return False

def test_api_lifespan():
    """测试API生命周期"""
    print("\n" + "="*50)
    print("测试API生命周期")
    print("="*50)
    
    try:
        from src.api.main import lifespan
        
        print("✅ 生命周期管理器检查:")
        print(f"   生命周期管理器: {lifespan}")
        print(f"   类型: {type(lifespan).__name__}")
        
        # 检查是否是异步上下文管理器
        import inspect
        if inspect.isasyncgenfunction(lifespan):
            print("   是异步上下文管理器")
        else:
            print("   不是异步上下文管理器")
        
        return True
    except Exception as e:
        print(f"❌ 生命周期检查失败: {e}")
        return False

def main():
    """主测试函数"""
    print("API测试")
    print("="*60)
    
    tests = [
        ("API导入", test_api_import),
        ("FastAPI应用", test_fastapi_app),
        ("Pydantic模型", test_schemas),
        ("路由", test_routes),
        ("生命周期", test_api_lifespan),
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
    print("API测试总结")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{test_name}: {status}")
    
    print(f"\n总计: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("\n🎉 所有API测试通过！")
        print("\n下一步:")
        print("1. 启动API服务器: python -m src.api.main")
        print("2. 访问 http://localhost:8000/docs 查看API文档")
        print("3. 使用curl或Postman测试API端点")
        return 0
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败")
        print("\n可能的原因:")
        print("1. 缺少FastAPI依赖")
        print("2. 代码语法错误")
        print("3. 导入路径问题")
        return 1

if __name__ == "__main__":
    sys.exit(main())