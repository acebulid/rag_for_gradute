#!/usr/bin/env python3
"""
导入示例数据到数据库
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncio
import json
from src.services.ollama_service import OllamaService
from src.database.vector_store import PostgreSQLVectorStore

async def import_sample_data():
    """导入示例数据"""
    print("="*50)
    print("导入示例数据到数据库")
    print("="*50)
    
    try:
        # 初始化服务
        ollama = OllamaService()
        vector_store = PostgreSQLVectorStore()
        await vector_store.connect()
        
        # 1. 导入文本数据
        print("\n📄 导入文本数据...")
        text_file = Path("data/raw/text/1.md")
        if text_file.exists():
            content = text_file.read_text(encoding="utf-8")
            print(f"  读取文件: {text_file.name} ({len(content)} 字符)")
            
            # 生成嵌入
            print("  生成文本嵌入...")
            embedding_result = ollama.generate_embedding(content)
            print(f"    嵌入维度: {len(embedding_result.embedding)}")
            print(f"    耗时: {embedding_result.duration_ms:.2f}ms")
            
            # 插入数据库
            doc_id = await vector_store.insert_document(
                content=content,
                embedding=embedding_result.embedding,
                metadata={
                    "source_file": text_file.name,
                    "file_size": len(content),
                    "import_time": "2026-01-07"
                },
                source="校园导览文档"
            )
            print(f"  ✅ 文档导入成功: {doc_id}")
        else:
            print(f"  ❌ 文本文件不存在: {text_file}")
            return False
        
        # 2. 导入图像数据
        print("\n🖼️  导入图像数据...")
        image_dir = Path("data/raw/images")
        image_files = list(image_dir.glob("*.png"))
        
        image_ids = []
        for image_file in image_files:
            print(f"  处理图像: {image_file.name}")
            
            # 生成图像描述
            print("    生成图像描述...")
            try:
                description_result = ollama.generate_image_description(str(image_file))
                print(f"    描述长度: {len(description_result.text)} 字符")
                print(f"    耗时: {description_result.duration_ms:.2f}ms")
                
                # 生成描述的嵌入
                print("    生成描述嵌入...")
                desc_embedding_result = ollama.generate_embedding(description_result.text)
                
                # 插入数据库
                image_id = await vector_store.insert_image_description(
                    image_path=str(image_file),
                    vlm_description=description_result.text,
                    embedding=desc_embedding_result.embedding,
                    metadata={
                        "source_file": image_file.name,
                        "file_size": image_file.stat().st_size,
                        "import_time": "2026-01-07"
                    },
                    image_size=f"{image_file.stat().st_size} bytes",
                    file_format="png"
                )
                image_ids.append(image_id)
                print(f"    ✅ 图像导入成功: {image_id}")
                
            except Exception as e:
                print(f"    ❌ 图像处理失败: {e}")
                continue
        
        if not image_ids:
            print("  ⚠️  没有成功导入的图像")
            return False
        
        # 3. 创建文本-图像关联
        print("\n🔗 创建文本-图像关联...")
        if image_ids:
            for image_id in image_ids:
                # 这里简化处理：假设所有图像都与文档相关
                relation_id = await vector_store.create_text_image_relation(
                    document_id=doc_id,
                    image_id=image_id,
                    similarity_score=0.8,  # 假设相似度
                    relation_type="校园建筑"
                )
                print(f"  ✅ 创建关联: {relation_id}")
        
        # 4. 验证数据
        print("\n📊 验证导入的数据...")
        doc_count = await vector_store.get_document_count()
        image_count = await vector_store.get_image_count()
        
        print(f"  文档数量: {doc_count}")
        print(f"  图像描述数量: {image_count}")
        
        await vector_store.close()
        return True
        
    except Exception as e:
        print(f"❌ 数据导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_imported_data():
    """测试导入的数据"""
    print("\n" + "="*50)
    print("测试导入的数据")
    print("="*50)
    
    try:
        vector_store = PostgreSQLVectorStore()
        await vector_store.connect()
        
        # 获取所有文档
        print("📄 检索文档...")
        test_embedding = [0.1] * 1024  # 测试用嵌入
        results = await vector_store.search_similar_documents(test_embedding, top_k=3)
        
        print(f"  找到 {len(results)} 个文档")
        for i, result in enumerate(results):
            print(f"  文档 {i+1}:")
            print(f"    ID: {result.id}")
            print(f"    相关性: {result.score:.3f}")
            preview = result.content[:100] + "..." if len(result.content) > 100 else result.content
            print(f"    内容预览: {preview}")
        
        # 获取所有图像
        print("\n🖼️  检索图像...")
        image_results = await vector_store.search_similar_images(test_embedding, top_k=3)
        
        print(f"  找到 {len(image_results)} 个图像")
        for i, result in enumerate(image_results):
            print(f"  图像 {i+1}:")
            print(f"    ID: {result.id}")
            print(f"    路径: {result.image_path}")
            print(f"    相关性: {result.score:.3f}")
            preview = result.vlm_description[:100] + "..." if len(result.vlm_description) > 100 else result.vlm_description
            print(f"    描述预览: {preview}")
        
        await vector_store.close()
        return True
        
    except Exception as e:
        print(f"❌ 数据测试失败: {e}")
        return False

async def main():
    """主函数"""
    print("示例数据导入工具")
    print("="*60)
    
    # 导入数据
    if not await import_sample_data():
        return 1
    
    # 测试数据
    if not await test_imported_data():
        return 1
    
    print("\n" + "="*60)
    print("🎉 示例数据导入完成！")
    print("="*60)
    print("\n数据已成功导入数据库。")
    print("现在可以运行RAG系统测试了。")
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))