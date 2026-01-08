#!/usr/bin/env python3
"""
测试数据处理流程
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """测试导入"""
    print("="*50)
    print("测试数据处理导入")
    print("="*50)
    
    try:
        from src.processing.pipeline import DataProcessingPipeline
        print('✅ 数据处理管道导入成功')
    except Exception as e:
        print(f'❌ 数据处理管道导入失败: {e}')
        return False
    
    try:
        from src.services.ollama_service import OllamaService
        print('✅ Ollama服务导入成功')
    except Exception as e:
        print(f'❌ Ollama服务导入失败: {e}')
        return False
    
    try:
        from src.database.vector_store import PostgreSQLVectorStore
        print('✅ 向量存储导入成功')
    except Exception as e:
        print(f'❌ 向量存储导入失败: {e}')
        return False
    
    try:
        from config.settings import settings
        print('✅ 配置导入成功')
        print(f'   数据库: {settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}')
        print(f'   Ollama: {settings.ollama_base_url}')
        print(f'   嵌入模型: {settings.embedding_model}')
        print(f'   VLM模型: {settings.vlm_model}')
        print(f'   LLM模型: {settings.llm_model}')
    except Exception as e:
        print(f'❌ 配置导入失败: {e}')
        return False
    
    return True

def test_data_directories():
    """测试数据目录"""
    print("\n" + "="*50)
    print("测试数据目录")
    print("="*50)
    
    text_dir = Path("data/raw/text")
    image_dir = Path("data/raw/images")
    
    if not text_dir.exists():
        print(f'❌ 文本目录不存在: {text_dir}')
        return False
    print(f'✅ 文本目录存在: {text_dir}')
    
    if not image_dir.exists():
        print(f'❌ 图像目录不存在: {image_dir}')
        return False
    print(f'✅ 图像目录存在: {image_dir}')
    
    # 检查文件
    text_files = list(text_dir.glob("*"))
    image_files = list(image_dir.glob("*"))
    
    print(f'📄 文本文件数量: {len(text_files)}')
    for file in text_files:
        print(f'   - {file.name}')
    
    print(f'🖼️  图像文件数量: {len(image_files)}')
    for file in image_files:
        print(f'   - {file.name}')
    
    if len(text_files) == 0 and len(image_files) == 0:
        print('⚠️  警告: 数据目录为空')
    
    return True

def test_file_content():
    """测试文件内容"""
    print("\n" + "="*50)
    print("测试文件内容")
    print("="*50)
    
    # 测试文本文件
    text_file = Path("data/raw/text/1.md")
    if text_file.exists():
        try:
            content = text_file.read_text(encoding='utf-8')
            print(f'✅ 文本文件可读取: {text_file.name}')
            print(f'   文件大小: {len(content)} 字符')
            print(f'   前100字符: {content[:100]}...')
        except Exception as e:
            print(f'❌ 文本文件读取失败: {e}')
            return False
    else:
        print(f'⚠️  文本文件不存在: {text_file}')
    
    # 测试图像文件
    image_file = Path("data/raw/images/本部_正门.png")
    if image_file.exists():
        try:
            file_size = image_file.stat().st_size
            print(f'✅ 图像文件可访问: {image_file.name}')
            print(f'   文件大小: {file_size / 1024:.1f} KB')
        except Exception as e:
            print(f'❌ 图像文件访问失败: {e}')
            return False
    else:
        print(f'⚠️  图像文件不存在: {image_file}')
    
    return True

def test_pipeline_creation():
    """测试管道创建"""
    print("\n" + "="*50)
    print("测试管道创建")
    print("="*50)
    
    try:
        from src.processing.pipeline import create_pipeline
        import asyncio
        
        async def test():
            try:
                pipeline = await create_pipeline()
                print('✅ 数据处理管道创建成功')
                print(f'   管道类型: {type(pipeline).__name__}')
                return True
            except Exception as e:
                print(f'❌ 数据处理管道创建失败: {e}')
                return False
        
        return asyncio.run(test())
        
    except Exception as e:
        print(f'❌ 管道创建测试失败: {e}')
        return False

def main():
    """主测试函数"""
    print("数据处理流程测试")
    print("="*60)
    
    tests = [
        ("导入测试", test_imports),
        ("目录测试", test_data_directories),
        ("文件测试", test_file_content),
        ("管道测试", test_pipeline_creation),
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
    print("数据处理测试总结")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{test_name}: {status}")
    
    print(f"\n总计: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("\n🎉 所有数据处理测试通过！")
        print("\n下一步:")
        print("1. 确保PostgreSQL服务正在运行")
        print("2. 确保Ollama服务正在运行 (ollama serve)")
        print("3. 运行完整的数据处理: python data/scripts/process_data.py")
        return 0
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败")
        print("\n可能的问题:")
        print("1. 数据目录结构不正确")
        print("2. 文件权限问题")
        print("3. 依赖包未安装")
        return 1

if __name__ == "__main__":
    sys.exit(main())