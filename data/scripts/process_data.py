#!/usr/bin/env python3
"""
数据处理脚本
用于批量处理校园导览数据，包括文本和图像
"""

import asyncio
import logging
import sys
import json
from pathlib import Path
from typing import Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.processing.pipeline import DataProcessingPipeline, create_pipeline
from config.settings import settings

# 配置日志
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def process_data(
    text_dir: Optional[str] = None,
    image_dir: Optional[str] = None,
    metadata_file: Optional[str] = None,
    output_dir: str = "data/processed",
    batch_size: int = None
):
    """
    处理数据的主函数
    
    Args:
        text_dir: 文本目录路径
        image_dir: 图像目录路径
        metadata_file: 元数据文件路径
        output_dir: 输出目录路径
        batch_size: 批处理大小
    """
    # 使用配置的批处理大小或默认值
    if batch_size is None:
        batch_size = settings.batch_size
    
    print("="*60)
    print("多模态RAG系统 - 数据处理")
    print("="*60)
    
    # 检查输入目录
    if text_dir:
        text_path = Path(text_dir)
        if not text_path.exists():
            print(f"❌ 文本目录不存在: {text_dir}")
            return False
        print(f"📁 文本目录: {text_dir}")
    
    if image_dir:
        image_path = Path(image_dir)
        if not image_path.exists():
            print(f"❌ 图像目录不存在: {image_dir}")
            return False
        print(f"🖼️  图像目录: {image_dir}")
    
    if metadata_file:
        metadata_path = Path(metadata_file)
        if not metadata_path.exists():
            print(f"⚠️  元数据文件不存在: {metadata_file}")
            metadata_file = None
        else:
            print(f"📄 元数据文件: {metadata_file}")
    
    print(f"⚙️  批处理大小: {batch_size}")
    print(f"📂 输出目录: {output_dir}")
    print("-"*60)
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # 创建数据处理管道
        print("🔄 初始化数据处理管道...")
        pipeline = await create_pipeline()
        
        # 处理数据
        print("🚀 开始处理数据...")
        stats = await pipeline.process_directory(
            text_dir=text_dir,
            image_dir=image_dir,
            metadata_file=metadata_file
        )
        
        # 打印统计信息
        print("\n" + "="*60)
        print("数据处理完成")
        print("="*60)
        pipeline.print_stats()
        
        # 保存处理统计
        stats_file = output_path / "processing_stats.json"
        stats_data = {
            "text_files_processed": stats.processed_texts,
            "text_files_failed": stats.failed_texts,
            "text_files_total": stats.total_texts,
            "image_files_processed": stats.processed_images,
            "image_files_failed": stats.failed_images,
            "image_files_total": stats.total_images,
            "relations_created": stats.created_relations,
            "elapsed_time_seconds": stats.elapsed_time,
            "text_success_rate": stats.text_success_rate,
            "image_success_rate": stats.image_success_rate,
            "config": {
                "batch_size": batch_size,
                "text_dir": text_dir,
                "image_dir": image_dir,
                "metadata_file": metadata_file
            }
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 处理统计已保存到: {stats_file}")
        
        # 保存处理日志
        log_file = output_path / "processing_log.txt"
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"数据处理日志\n")
            f.write(f"时间: {stats_data.get('timestamp', 'N/A')}\n")
            f.write(f"文本文件: {stats.processed_texts}/{stats.total_texts} "
                   f"({stats.text_success_rate*100:.1f}%)\n")
            f.write(f"图像文件: {stats.processed_images}/{stats.total_images} "
                   f"({stats.image_success_rate*100:.1f}%)\n")
            f.write(f"关联创建: {stats.created_relations}\n")
            f.write(f"总耗时: {stats.elapsed_time:.2f}秒\n")
        
        print(f"📝 处理日志已保存到: {log_file}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 数据处理失败: {e}")
        logger.error(f"数据处理失败: {e}", exc_info=True)
        return False


def find_data_directories():
    """查找数据目录"""
    base_dir = Path("data/raw")
    
    text_dir = None
    image_dir = None
    metadata_file = None
    
    if (base_dir / "text").exists():
        text_dir = str(base_dir / "text")
    elif (base_dir / "texts").exists():
        text_dir = str(base_dir / "texts")
    
    if (base_dir / "images").exists():
        image_dir = str(base_dir / "images")
    elif (base_dir / "image").exists():
        image_dir = str(base_dir / "image")
    
    if (base_dir / "metadata.json").exists():
        metadata_file = str(base_dir / "metadata.json")
    
    return text_dir, image_dir, metadata_file


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="多模态RAG系统数据处理脚本")
    parser.add_argument("--text-dir", help="文本目录路径")
    parser.add_argument("--image-dir", help="图像目录路径")
    parser.add_argument("--metadata", help="元数据文件路径")
    parser.add_argument("--output-dir", default="data/processed", help="输出目录路径")
    parser.add_argument("--batch-size", type=int, help="批处理大小")
    
    args = parser.parse_args()
    
    # 如果没有指定目录，尝试自动查找
    text_dir = args.text_dir
    image_dir = args.image_dir
    metadata_file = args.metadata
    
    if not text_dir and not image_dir:
        print("🔍 未指定数据目录，尝试自动查找...")
        found_text, found_image, found_metadata = find_data_directories()
        
        if found_text:
            text_dir = found_text
            print(f"  找到文本目录: {text_dir}")
        
        if found_image:
            image_dir = found_image
            print(f"  找到图像目录: {image_dir}")
        
        if found_metadata:
            metadata_file = found_metadata
            print(f"  找到元数据文件: {metadata_file}")
        
        if not text_dir and not image_dir:
            print("❌ 未找到数据目录，请手动指定")
            print("\n使用方法:")
            print("  python scripts/process_data.py --text-dir /path/to/texts --image-dir /path/to/images")
            print("  或")
            print("  将数据放入 data/raw/texts/ 和 data/raw/images/ 目录")
            return 1
    
    # 运行数据处理
    success = asyncio.run(
        process_data(
            text_dir=text_dir,
            image_dir=image_dir,
            metadata_file=metadata_file,
            output_dir=args.output_dir,
            batch_size=args.batch_size
        )
    )
    
    if success:
        print("\n" + "="*60)
        print("🎉 数据处理完成！")
        print("\n下一步建议：")
        print("1. 启动API服务器: python -m src.api.main")
        print("2. 运行测试: python scripts/test_retrieval.py")
        print("3. 访问 http://localhost:8000/docs 查看API文档")
        print("="*60)
        return 0
    else:
        print("\n数据处理失败，请检查错误信息。")
        return 1


if __name__ == "__main__":
    sys.exit(main())