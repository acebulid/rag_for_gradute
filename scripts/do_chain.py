#!/usr/bin/env python3
"""
chain处理主函数 - 调用chain_function处理查询
注意：本文件只包含主函数，不创建新函数
"""

import sys
import asyncio
from pathlib import Path
import json

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入chain_function中的现有函数
from function.chain_function import (
    ChainConfig,
    process_chain_query,
    test_chain_function
)


def main():
    """
    主函数：演示chain功能
    """
    print("=" * 70)
    print("chain处理主函数")
    print("=" * 70)
    
    # 创建配置
    config = ChainConfig()
    
    print("\n1. 测试chainB（文本查询）...")
    print("   示例问题: '首都师范大学的校门在哪里？'")
    
    # 运行异步函数
    result = asyncio.run(process_chain_query(
        query_type='text',
        query_content='首都师范大学的校门在哪里？',
        config=config
    ))
    
    if result["success"]:
        print(f"   查询成功")
        print(f"   检索到 {len(result['search_results'])} 个结果:")
        
        for i, r in enumerate(result['search_results'], 1):
            print(f"\n   [{i}] 相似度: {r['score']:.3f}")
            print(f"       内容: {r['content_preview']}")
            if r.get('source'):
                print(f"       来源: {r['source']}")
    else:
        print(f"   查询失败: {result['error']}")
    
    print("\n2. 测试chainA（图片查询）...")
    # 检查是否有测试图片
    test_images = [
        "data/image-1.png",
        "data/image-2.png",
        "data/image-3.png"
    ]
    
    test_image = None
    for img in test_images:
        if Path(img).exists():
            test_image = img
            break
    
    if test_image:
        print(f"   使用测试图片: {test_image}")
        
        result = asyncio.run(process_chain_query(
            query_type='image',
            query_content=test_image,
            config=config
        ))
        
        if result["success"]:
            print(f"   查询成功")
            print(f"   检索到 {len(result['search_results'])} 个结果:")
            
            for i, r in enumerate(result['search_results'], 1):
                print(f"\n   [{i}] 相似度: {r['score']:.3f}")
                print(f"       内容: {r['content_preview']}")
                if r.get('source'):
                    print(f"       来源: {r['source']}")
        else:
            print(f"   查询失败: {result['error']}")
    else:
        print("   未找到测试图片，跳过chainA测试")
    
    print("\n3. 运行完整测试...")
    print("   运行chain_function中的测试函数...")
    
    try:
        asyncio.run(test_chain_function())
        print("   完整测试完成")
    except Exception as e:
        print(f"   完整测试失败: {e}")
    
    print("\n" + "=" * 70)
    print("chain处理完成")
    print("=" * 70)
    
    # 提供使用说明
    print("\n使用说明:")
    print("1. 文本查询: python scripts/do_chain.py --text '你的问题'")
    print("2. 图片查询: python scripts/do_chain.py --image '图片路径'")
    print("3. 运行测试: python scripts/do_chain.py --test")


def parse_arguments():
    """
    解析命令行参数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='chain处理主函数')
    parser.add_argument('--text', type=str, help='文本查询内容')
    parser.add_argument('--image', type=str, help='图片查询路径')
    parser.add_argument('--test', action='store_true', help='运行测试')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    if args.test:
        # 运行测试
        asyncio.run(test_chain_function())
    elif args.text:
        # 文本查询
        config = ChainConfig()
        result = asyncio.run(process_chain_query(
            query_type='text',
            query_content=args.text,
            config=config
        ))
        
        if result["success"]:
            print(f"查询成功，检索到 {len(result['search_results'])} 个结果:")
            for i, r in enumerate(result['search_results'], 1):
                print(f"\n[{i}] 相似度: {r['score']:.3f}")
                print(f"   内容: {r['content_preview']}")
        else:
            print(f"查询失败: {result['error']}")
    elif args.image:
        # 图片查询
        config = ChainConfig()
        result = asyncio.run(process_chain_query(
            query_type='image',
            query_content=args.image,
            config=config
        ))
        
        if result["success"]:
            print(f"查询成功，检索到 {len(result['search_results'])} 个结果:")
            for i, r in enumerate(result['search_results'], 1):
                print(f"\n[{i}] 相似度: {r['score']:.3f}")
                print(f"   内容: {r['content_preview']}")
        else:
            print(f"查询失败: {result['error']}")
    else:
        # 默认运行演示
        main()