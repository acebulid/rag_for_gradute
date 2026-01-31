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

# ========== 新增：公共结果打印函数 ==========
def print_search_result(result):
    """
    公共函数：统一打印检索结果
    """
    if not isinstance(result, dict) or "success" not in result:
        print("无效的查询结果格式")
        return
    
    if result["success"]:
        if len(result.get('search_results', [])) == 0:
            print(f"查询成功，但未检索到相关结果，请更换查询关键词重试")
        else:
            search_results = result['search_results']
            print(f"查询成功，检索到 {len(search_results)} 个结果:")
            for i, r in enumerate(search_results, 1):
                print(f"\n[{i}] 相似度: {r['score']:.3f}")
                print(f"   内容: {r['content_preview']}")
                if r.get('source'):
                    print(f"   来源: {r['source']}")
    else:
        print(f"查询失败: {result['error']}")

# ========== 其余函数（main）逻辑不变，补充空结果处理 ==========
def main():
    """
    主函数：演示chain功能
    """
    print("=" * 70)
    print("chain处理主函数 - 新架构 (chainB + chainC)")
    print("=" * 70)
    
    # 创建配置
    config = ChainConfig()
    
    print("\n1. 测试chainB + chainC（文本查询）...")
    print("   示例问题: '首都师范大学的校门在哪里？'")
    
    # 运行异步函数
    result = asyncio.run(process_chain_query(
        query_type='text',
        query_content='首都师范大学的校门在哪里？',
        config=config
    ))
    
    # 调用公共打印函数
    print_search_result(result)
    
    # 后续逻辑不变...
    print("\n2. 测试chainA（图片查询）...")
    print("   注意: chainA功能待实现")
    
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
        print(f"   测试图片: {test_image}")
        print("   chainA功能待实现，跳过测试")
    else:
        print("   未找到测试图片")
    
    print("\n3. 运行完整测试...")
    print("   运行chain_function中的测试函数...")
    
    try:
        asyncio.run(test_chain_function())
        print("   完整测试完成")
    except Exception as e:
        print(f"   完整测试失败: {e}")
    


# ========== 修复：parse_arguments添加互斥参数组 + 图片查询友好提示 ==========
def parse_arguments():
    """
    解析命令行参数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='chain处理主函数')
    # 创建互斥参数组，避免多参数同时传入
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--text', type=str, help='文本查询内容')
    group.add_argument('--image', type=str, help='图片查询路径')
    group.add_argument('--test', action='store_true', help='运行测试')
    
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
        print_search_result(result)
    elif args.image:
        # 图片查询（chainA已实现）
        print("=== 图片查询功能（chainA） ===")
        print(f"   传入图片路径: {args.image}")
        
        # 图片文件存在性校验
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"   错误: 图片文件不存在或路径错误: {img_path}")
            result = {"success": False, "error": f"图片文件不存在: {args.image}"}
            print_search_result(result)
        else:
            print(f"   图片文件存在，格式: {img_path.suffix}")
            print("   开始处理图片查询...")
            
            # 调用chainA处理图片查询
            config = ChainConfig()
            result = asyncio.run(process_chain_query(
                query_type='image',
                query_content=args.image,
                config=config
            ))
            print_search_result(result)
    else:
        # 默认运行演示
        main()
