#!/usr/bin/env python3
"""
配置模块测试
测试环境变量加载和配置验证
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_env_file():
    """测试环境文件"""
    print("="*50)
    print("测试环境配置文件")
    print("="*50)
    
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env文件不存在")
        return False
    
    print("✅ .env文件存在")
    
    # 读取环境文件
    with open(env_file, 'r') as f:
        content = f.read()
    
    # 检查关键配置
    required_configs = [
        "POSTGRES_HOST",
        "POSTGRES_PORT", 
        "POSTGRES_DB",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "OLLAMA_BASE_URL",
        "EMBEDDING_MODEL",
        "VLM_MODEL",
        "LLM_MODEL"
    ]
    
    missing_configs = []
    for config in required_configs:
        if f"{config}=" not in content:
            missing_configs.append(config)
    
    if missing_configs:
        print(f"❌ 缺少配置项: {', '.join(missing_configs)}")
        return False
    
    print("✅ 所有必需配置项都存在")
    return True

def test_settings_import():
    """测试配置导入"""
    print("\n" + "="*50)
    print("测试配置模块导入")
    print("="*50)
    
    try:
        from config.settings import settings
        
        print("✅ 配置模块导入成功")
        print(f"   数据库: {settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}")
        print(f"   Ollama: {settings.ollama_base_url}")
        print(f"   嵌入模型: {settings.embedding_model}")
        print(f"   VLM模型: {settings.vlm_model}")
        print(f"   LLM模型: {settings.llm_model}")
        
        return True
    except Exception as e:
        print(f"❌ 配置模块导入失败: {e}")
        return False

def test_database_config():
    """测试数据库配置"""
    print("\n" + "="*50)
    print("测试数据库配置")
    print("="*50)
    
    try:
        from config.database import database
        
        print("✅ 数据库配置模块导入成功")
        
        # 测试数据库连接（可选）
        # 注意：这需要数据库服务正在运行
        print("   数据库连接测试需要PostgreSQL服务正在运行")
        
        return True
    except Exception as e:
        print(f"❌ 数据库配置模块导入失败: {e}")
        return False

def main():
    """主测试函数"""
    print("配置模块测试")
    print("="*60)
    
    tests = [
        ("环境文件", test_env_file),
        ("配置导入", test_settings_import),
        ("数据库配置", test_database_config),
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
    print("配置测试总结")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{test_name}: {status}")
    
    print(f"\n总计: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("\n🎉 所有配置测试通过！")
        return 0
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())