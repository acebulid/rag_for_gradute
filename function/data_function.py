"""
Markdown解析器 - 解析校园导览知识库.md文件
提取章节标题、文本内容和图片引用，建立章节与图片的对应关系
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict


@dataclass
class Chapter:
    """章节数据结构"""
    title: str  # 章节标题，如"校门"
    level: int  # 标题级别，##为2
    content: str  # 章节文本内容
    image_refs: List[str]  # 图片引用列表，如["image-6.png", "image-8.png"]
    start_line: int  # 起始行号
    end_line: int  # 结束行号


@dataclass
class ParsedMarkdown:
    """解析后的Markdown数据结构"""
    chapters: List[Chapter]  # 所有章节
    total_images: int  # 总图片数量
    metadata: Dict[str, Any]  # 元数据


class CampusMarkdownParser:
    """校园导览Markdown解析器"""
    
    def __init__(self):
        self.chapter_pattern = re.compile(r'^(#{1,})\s+(.+)$')  # 匹配#标题
        self.image_pattern = re.compile(r'!\[.*?\]\((.*?)\)')  # 匹配图片引用
    
    def parse_file(self, file_path: str) -> ParsedMarkdown:
        """解析Markdown文件"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        lines = file_path.read_text(encoding='utf-8').splitlines()
        chapters = []
        current_chapter = None
        in_chapter = False
        
        for i, line in enumerate(lines):
            # 检查是否为章节标题
            match = self.chapter_pattern.match(line)
            if match:
                # 如果已有正在处理的章节，先保存
                if current_chapter:
                    current_chapter.end_line = i - 1
                    chapters.append(current_chapter)
                
                # 开始新章节
                level = len(match.group(1))  # #的数量
                title = match.group(2).strip()
                current_chapter = Chapter(
                    title=title,
                    level=level,
                    content="",
                    image_refs=[],
                    start_line=i,
                    end_line=len(lines) - 1  # 临时值
                )
                in_chapter = True
                continue
            
            # 如果在章节中，处理内容
            if in_chapter and current_chapter:
                # 检查图片引用
                image_matches = self.image_pattern.findall(line)
                if image_matches:
                    current_chapter.image_refs.extend(image_matches)
                
                # 添加文本内容（排除图片行）
                if not image_matches:
                    current_chapter.content += line + "\n"
        
        # 添加最后一个章节
        if current_chapter:
            current_chapter.end_line = len(lines) - 1
            chapters.append(current_chapter)
        
        # 计算总图片数量
        total_images = sum(len(chapter.image_refs) for chapter in chapters)
        
        # 提取元数据
        metadata = {
            "file_path": str(file_path),
            "total_lines": len(lines),
            "total_chapters": len(chapters),
            "chapter_titles": [chapter.title for chapter in chapters],
            "image_files": self._extract_all_image_files(chapters)
        }
        
        return ParsedMarkdown(
            chapters=chapters,
            total_images=total_images,
            metadata=metadata
        )
    
    def _extract_all_image_files(self, chapters: List[Chapter]) -> List[str]:
        """提取所有图片文件"""
        image_files = []
        for chapter in chapters:
            for image_ref in chapter.image_refs:
                if image_ref not in image_files:
                    image_files.append(image_ref)
        return image_files
    
    def get_chapter_by_title(self, parsed: ParsedMarkdown, title: str) -> Optional[Chapter]:
        """根据标题获取章节"""
        for chapter in parsed.chapters:
            if chapter.title == title:
                return chapter
        return None
    
    def get_chapters_with_images(self, parsed: ParsedMarkdown) -> List[Chapter]:
        """获取包含图片的章节"""
        return [chapter for chapter in parsed.chapters if chapter.image_refs]
    
    def print_summary(self, parsed: ParsedMarkdown, verbose: bool = True):
        """
        打印解析摘要
        
        参数:
            parsed: 解析后的Markdown数据
            verbose: 是否打印详细的章节列表，默认True
        """
        print("=" * 60)
        print("Markdown解析摘要")
        print("=" * 60)
        print(f"文件路径: {parsed.metadata['file_path']}")
        print(f"总章节数: {parsed.metadata['total_chapters']}")
        print(f"总图片数: {parsed.total_images}")
        print(f"包含图片的章节: {len(self.get_chapters_with_images(parsed))}")
        
        if verbose:
            print("\n章节列表:")
            for i, chapter in enumerate(parsed.chapters, 1):
                print(f"  {i}. {chapter.title} (级别: {chapter.level}, 图片: {len(chapter.image_refs)}个)")
        print("=" * 60)
    
    def export_to_json(self, parsed: ParsedMarkdown, output_path: str, full_content: bool = False):
        """
        导出为JSON格式
        
        参数:
            parsed: 解析后的Markdown数据
            output_path: 输出文件路径
            full_content: 是否导出完整内容，False时只导出前200字符预览，默认False
        """
        # 转换Chapter对象为字典
        def chapter_to_dict(chapter: Chapter) -> Dict[str, Any]:
            chapter_dict = asdict(chapter)
            if not full_content and len(chapter_dict['content']) > 200:
                chapter_dict['content_preview'] = chapter_dict['content'][:200] + "..."
                del chapter_dict['content']
            chapter_dict['image_count'] = len(chapter_dict['image_refs'])
            return chapter_dict
        
        data = {
            "metadata": parsed.metadata,
            "chapters": [chapter_to_dict(chapter) for chapter in parsed.chapters]
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"解析结果已导出到: {output_path}")


# ============================ 功能函数 ============================
def create_parser() -> CampusMarkdownParser:
    """创建解析器实例"""
    return CampusMarkdownParser()


def parse_markdown_file(file_path: str) -> Tuple[Optional[ParsedMarkdown], Optional[str]]:
    """
    解析单个Markdown文件（带错误处理）
    
    参数:
        file_path: Markdown文件路径
    
    返回:
        Tuple: (解析结果/None, 错误信息/None)
    """
    try:
        parser = create_parser()
        parsed_data = parser.parse_file(file_path)
        return parsed_data, None
    except Exception as e:
        error_msg = f"解析文件失败: {str(e)}"
        print(error_msg)
        return None, error_msg


def batch_parse_markdown_files(file_paths: List[str]) -> Dict[str, Any]:
    """
    批量解析多个Markdown文件
    
    参数:
        file_paths: 文件路径列表
    
    返回:
        Dict: 批量解析结果
            - success_count: 成功解析的文件数
            - failed_count: 解析失败的文件数
            - failed_files: 失败的文件列表
            - parsed_results: 解析结果字典 {file_path: ParsedMarkdown}
    """
    result = {
        "success_count": 0,
        "failed_count": 0,
        "failed_files": [],
        "parsed_results": {}
    }
    
    parser = create_parser()
    
    for file_path in file_paths:
        print(f"\n正在解析: {file_path}")
        try:
            parsed_data = parser.parse_file(file_path)
            result["parsed_results"][file_path] = parsed_data
            result["success_count"] += 1
            print(f"✅ 解析成功: {file_path}")
        except Exception as e:
            error_msg = f"❌ 解析失败 {file_path}: {str(e)}"
            print(error_msg)
            result["failed_files"].append({
                "file_path": file_path,
                "error": error_msg
            })
            result["failed_count"] += 1
    
    print(f"\n批量解析完成 - 成功: {result['success_count']}, 失败: {result['failed_count']}")
    return result


def show_chapter_details(parsed: ParsedMarkdown, chapter_title: Optional[str] = None, chapter_index: int = 0):
    """
    显示指定章节的详细信息
    
    参数:
        parsed: 解析后的Markdown数据
        chapter_title: 章节标题（优先使用）
        chapter_index: 章节索引（当标题不存在时使用）
    """
    if chapter_title:
        chapter = create_parser().get_chapter_by_title(parsed, chapter_title)
        if not chapter:
            print(f"未找到章节: {chapter_title}")
            return
    else:
        if len(parsed.chapters) <= chapter_index:
            print(f"章节索引超出范围，总章节数: {len(parsed.chapters)}")
            return
        chapter = parsed.chapters[chapter_index]
    
    print(f"\n章节详细信息: {chapter.title}")
    print("-" * 40)
    print(f"级别: {chapter.level}")
    print(f"行范围: {chapter.start_line}-{chapter.end_line}")
    print(f"内容长度: {len(chapter.content)} 字符")
    print(f"图片数量: {len(chapter.image_refs)}")
    if chapter.image_refs:
        print(f"图片列表: {', '.join(chapter.image_refs)}")
    print(f"\n内容预览:\n{chapter.content[:300]}...")


def analyze_markdown_content(parsed: ParsedMarkdown) -> Dict[str, Any]:
    """
    分析Markdown内容统计信息
    
    参数:
        parsed: 解析后的Markdown数据
    
    返回:
        Dict: 统计分析结果
            - avg_chapter_length: 平均章节内容长度
            - chapters_by_level: 按级别统计章节数
            - chapters_without_images: 无图片的章节数
            - top_image_chapters: 图片最多的前3个章节
    """
    # 计算平均章节内容长度
    content_lengths = [len(chapter.content) for chapter in parsed.chapters]
    avg_length = sum(content_lengths) / len(content_lengths) if content_lengths else 0
    
    # 按级别统计章节
    chapters_by_level = {}
    for chapter in parsed.chapters:
        level_key = f"level_{chapter.level}"
        chapters_by_level[level_key] = chapters_by_level.get(level_key, 0) + 1
    
    # 无图片的章节数
    chapters_without_images = len([c for c in parsed.chapters if not c.image_refs])
    
    # 图片最多的前3个章节
    sorted_chapters = sorted(
        parsed.chapters, 
        key=lambda x: len(x.image_refs), 
        reverse=True
    )
    top_image_chapters = [
        {
            "title": chapter.title,
            "image_count": len(chapter.image_refs)
        } 
        for chapter in sorted_chapters[:3]
    ]
    
    return {
        "avg_chapter_length": round(avg_length, 2),
        "chapters_by_level": chapters_by_level,
        "chapters_without_images": chapters_without_images,
        "top_image_chapters": top_image_chapters,
        "total_unique_images": len(parsed.metadata["image_files"])
    }


# ============================ 测试和使用示例 ============================
def test_single_file_parsing(test_file: str = "data/raw/text/首都师范大学本部校园导览知识库.md"):
    """
    测试单个文件解析功能
    
    参数:
        test_file: 测试文件路径
    """
    print("=== 测试单个文件解析 ===")
    parsed, error = parse_markdown_file(test_file)
    
    if parsed:
        # 打印摘要
        create_parser().print_summary(parsed)
        
        # 显示第一个章节详情
        show_chapter_details(parsed, chapter_index=0)
        
        # 分析内容
        analysis = analyze_markdown_content(parsed)
        print("\n=== 内容分析结果 ===")
        for key, value in analysis.items():
            print(f"{key}: {value}")
        
        # 导出JSON
        create_parser().export_to_json(parsed, "data/processed/markdown_parsed.json", full_content=False)
        return True
    else:
        print(f"测试失败: {error}")
        return False


def test_batch_parsing(test_files: List[str] = None):
    """
    测试批量解析功能
    
    参数:
        test_files: 测试文件列表，默认使用示例文件
    """
    if test_files is None:
        test_files = [
            "data/raw/text/首都师范大学本部校园导览知识库.md",
            "data/raw/text/首都师范大学北校区导览知识库.md",
            "data/raw/text/首都师范大学良乡校区导览知识库.md"
        ]
    
    print("=== 测试批量文件解析 ===")
    batch_result = batch_parse_markdown_files(test_files)
    
    # 打印批量解析结果
    print("\n=== 批量解析统计 ===")
    print(f"成功解析: {batch_result['success_count']} 个文件")
    print(f"解析失败: {batch_result['failed_count']} 个文件")
    
    if batch_result["failed_files"]:
        print("\n失败文件列表:")
        for failed in batch_result["failed_files"]:
            print(f"  - {failed['file_path']}: {failed['error']}")
    
    # 对成功解析的文件显示摘要
    for file_path, parsed_data in batch_result["parsed_results"].items():
        print(f"\n=== {file_path} 摘要 ===")
        create_parser().print_summary(parsed_data, verbose=False)


if __name__ == "__main__":
    # 1. 测试单个文件解析
    test_single_file_parsing()
    
    # 2. 测试批量解析（取消注释启用）
    # test_batch_parsing()
    
    # 3. 自定义使用示例
    # parsed_data, _ = parse_markdown_file("你的文件路径.md")
    # if parsed_data:
    #     # 查找特定章节
    #     chapter = create_parser().get_chapter_by_title(parsed_data, "校门")
    #     if chapter:
    #         print(f"\n找到章节: {chapter.title}, 包含 {len(chapter.image_refs)} 张图片")