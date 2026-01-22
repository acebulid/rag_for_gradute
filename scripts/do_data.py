#!/usr/bin/env python3
"""
数据处理脚本 - 调用data_function处理markdown数据
将数据转换为适合数据库存储的格式
注意：本文件只包含主函数，不创建新函数
"""

import sys
import json
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入data_function中的现有函数
from function.data_function import (
    create_parser,
    parse_markdown_file,
    show_chapter_details,
    analyze_markdown_content
)


if __name__ == "__main__":
    """
    主函数：处理markdown数据并准备数据库数据
    """
    print("=" * 70)
    print("开始处理markdown数据")
    print("=" * 70)
    
    # 1. 解析markdown文件
    print("\n[1/4] 解析markdown文件...")
    file_path = "data/首都师范大学本部校园导览知识库.md"
    print(f"   文件路径: {file_path}")
    
    parsed_data, error = parse_markdown_file(file_path)
    
    if error:
        print(f"   解析失败: {error}")
        sys.exit(1)
    
    print("   解析成功!")
    
    # 2. 打印解析摘要
    print("\n[2/4] 打印解析摘要...")
    parser = create_parser()
    parser.print_summary(parsed_data, verbose=True)
    
    # 3. 分析内容
    print("\n[3/4] 分析内容统计信息...")
    analysis = analyze_markdown_content(parsed_data)
    print("   内容分析结果:")
    for key, value in analysis.items():
        print(f"     {key}: {value}")
    
    # 4. 显示章节详情（前3个章节）
    print("\n[4/4] 显示章节详情...")
    for i in range(min(3, len(parsed_data.chapters))):
        show_chapter_details(parsed_data, chapter_index=i)
        print()
    
    # 5. 导出为JSON格式（使用parser的现有方法）
    print("\n[额外] 导出数据为JSON格式...")
    output_path = "data/processed/markdown_parsed.json"
    parser.export_to_json(parsed_data, output_path, full_content=False)
    
    # 6. 准备数据库数据结构
    print("\n[额外] 准备数据库数据结构...")
    database_data = {
        "metadata": parsed_data.metadata,
        "total_chapters": len(parsed_data.chapters),
        "total_images": parsed_data.total_images,
        "chapters": []
    }
    
    # 提取章节数据（使用现有数据结构，不创建新函数）
    for i, chapter in enumerate(parsed_data.chapters):
        chapter_info = {
            "id": f"chapter_{i+1:03d}",
            "title": chapter.title,
            "level": chapter.level,
            "content_preview": chapter.content[:200] + "..." if len(chapter.content) > 200 else chapter.content,
            "content_length": len(chapter.content),
            "image_count": len(chapter.image_refs),
            "image_refs": chapter.image_refs,
            "start_line": chapter.start_line,
            "end_line": chapter.end_line
        }
        database_data["chapters"].append(chapter_info)
    
    # 保存数据库数据结构
    db_output_path = "data/processed/database_data.json"
    Path(db_output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(db_output_path, 'w', encoding='utf-8') as f:
        json.dump(database_data, f, indent=2, ensure_ascii=False)
    
    print(f"   数据库数据结构已保存到: {db_output_path}")
    
    # 7. 统计信息
    print("\n" + "=" * 70)
    print("数据处理完成!")
    print("=" * 70)
    print(f"   总章节数: {len(parsed_data.chapters)}")
    print(f"   总图片数: {parsed_data.total_images}")
    
    # 统计包含图片的章节
    chapters_with_images = [c for c in parsed_data.chapters if c.image_refs]
    print(f"   包含图片的章节数: {len(chapters_with_images)}")
    
    # 统计图片最多的章节
    if parsed_data.chapters:
        max_images_chapter = max(parsed_data.chapters, key=lambda x: len(x.image_refs))
        print(f"   图片最多的章节: {max_images_chapter.title} ({len(max_images_chapter.image_refs)}张图片)")
    
    print(f"   输出文件:")
    print(f"     - {output_path}")
    print(f"     - {db_output_path}")
    print("=" * 70)
