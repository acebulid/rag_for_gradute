import logging
import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time
from tqdm import tqdm
import hashlib

from src.services.ollama_service import OllamaService, EmbeddingResult, GenerationResult
from src.database.vector_store import PostgreSQLVectorStore
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class ProcessingStats:
    """处理统计"""
    total_texts: int = 0
    processed_texts: int = 0
    failed_texts: int = 0
    total_images: int = 0
    processed_images: int = 0
    failed_images: int = 0
    total_relations: int = 0
    created_relations: int = 0
    start_time: float = 0
    end_time: float = 0
    
    @property
    def elapsed_time(self) -> float:
        """经过时间（秒）"""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time
    
    @property
    def text_success_rate(self) -> float:
        """文本处理成功率"""
        if self.total_texts == 0:
            return 0.0
        return self.processed_texts / self.total_texts
    
    @property
    def image_success_rate(self) -> float:
        """图像处理成功率"""
        if self.total_images == 0:
            return 0.0
        return self.processed_images / self.total_images


class DataProcessingPipeline:
    """数据处理管道"""
    
    def __init__(self, ollama_service: OllamaService, vector_store: PostgreSQLVectorStore):
        self.ollama = ollama_service
        self.vector_store = vector_store
        self.stats = ProcessingStats()
    
    async def process_text_file(self, file_path: str, metadata: Optional[Dict] = None) -> Optional[str]:
        """处理单个文本文件"""
        try:
            # 读取文件内容
            content = Path(file_path).read_text(encoding='utf-8')
            
            # 生成嵌入
            embedding_result = await self.ollama.generate_embedding(content)
            
            # 构建元数据
            file_metadata = {
                "source": file_path,
                "file_size": len(content),
                "hash": hashlib.md5(content.encode()).hexdigest(),
                **(metadata or {})
            }
            
            # 插入数据库
            doc_id = await self.vector_store.insert_document(
                content=content,
                embedding=embedding_result.embedding,
                metadata=file_metadata,
                source=file_path
            )
            
            logger.info(f"Processed text file: {file_path} -> {doc_id}")
            return doc_id
            
        except Exception as e:
            logger.error(f"Failed to process text file {file_path}: {e}")
            return None
    
    async def process_image_file(self, image_path: str, metadata: Optional[Dict] = None) -> Optional[Tuple[str, str]]:
        """处理单个图像文件"""
        try:
            # 生成图像描述
            description_result = await self.ollama.generate_image_description(image_path)
            
            # 生成描述嵌入
            embedding_result = await self.ollama.generate_embedding(description_result.text)
            
            # 获取图像信息
            image_file = Path(image_path)
            image_size = self._get_image_size(image_path)
            
            # 构建元数据
            file_metadata = {
                "source": image_path,
                "file_size": image_file.stat().st_size,
                "hash": self._calculate_file_hash(image_path),
                "description_tokens": description_result.tokens,
                "description_time_ms": description_result.duration_ms,
                **(metadata or {})
            }
            
            # 插入数据库
            image_id = await self.vector_store.insert_image_description(
                image_path=image_path,
                vlm_description=description_result.text,
                embedding=embedding_result.embedding,
                metadata=file_metadata,
                image_size=image_size,
                file_format=image_path.split('.')[-1].lower()
            )
            
            logger.info(f"Processed image file: {image_path} -> {image_id}")
            return image_id, description_result.text
            
        except Exception as e:
            logger.error(f"Failed to process image file {image_path}: {e}")
            return None
    
    async def process_text_batch(self, text_files: List[str], 
                               metadata_list: Optional[List[Dict]] = None,
                               batch_size: int = 10) -> List[Optional[str]]:
        """批量处理文本文件"""
        self.stats.total_texts = len(text_files)
        self.stats.start_time = time.time()
        
        results = []
        
        for i in range(0, len(text_files), batch_size):
            batch = text_files[i:i + batch_size]
            batch_metadata = metadata_list[i:i + batch_size] if metadata_list else [None] * len(batch)
            
            # 创建处理任务
            tasks = []
            for file_path, metadata in zip(batch, batch_metadata):
                tasks.append(self.process_text_file(file_path, metadata))
            
            # 并行处理批次
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 统计结果
            for result in batch_results:
                if isinstance(result, Exception) or result is None:
                    self.stats.failed_texts += 1
                    results.append(None)
                else:
                    self.stats.processed_texts += 1
                    results.append(result)
            
            logger.info(f"Processed text batch {i//batch_size + 1}/{(len(text_files)-1)//batch_size + 1}, "
                       f"success: {self.stats.processed_texts}, failed: {self.stats.failed_texts}")
        
        self.stats.end_time = time.time()
        return results
    
    async def process_image_batch(self, image_files: List[str],
                                metadata_list: Optional[List[Dict]] = None,
                                batch_size: int = 5) -> List[Optional[Tuple[str, str]]]:
        """批量处理图像文件"""
        self.stats.total_images = len(image_files)
        self.stats.start_time = time.time()
        
        results = []
        
        for i in range(0, len(image_files), batch_size):
            batch = image_files[i:i + batch_size]
            batch_metadata = metadata_list[i:i + batch_size] if metadata_list else [None] * len(batch)
            
            # 创建处理任务
            tasks = []
            for image_path, metadata in zip(batch, batch_metadata):
                tasks.append(self.process_image_file(image_path, metadata))
            
            # 并行处理批次
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 统计结果
            for result in batch_results:
                if isinstance(result, Exception) or result is None:
                    self.stats.failed_images += 1
                    results.append(None)
                else:
                    self.stats.processed_images += 1
                    results.append(result)
            
            logger.info(f"Processed image batch {i//batch_size + 1}/{(len(image_files)-1)//batch_size + 1}, "
                       f"success: {self.stats.processed_images}, failed: {self.stats.failed_images}")
            
            # 批次间延迟，避免过载
            if i + batch_size < len(image_files):
                await asyncio.sleep(1)
        
        self.stats.end_time = time.time()
        return results
    
    async def create_text_image_relations(self, document_ids: List[str], 
                                        image_ids: List[str],
                                        similarity_threshold: float = 0.7) -> int:
        """创建文本-图像关联"""
        self.stats.total_relations = len(document_ids) * len(image_ids)
        
        created_count = 0
        
        for doc_id in document_ids:
            if not doc_id:
                continue
                
            for img_id in image_ids:
                if not img_id:
                    continue
                
                try:
                    # 这里可以添加更复杂的相似度计算逻辑
                    # 目前使用固定相似度，实际应用中应该计算实际相似度
                    similarity_score = 0.8  # 示例值
                    
                    if similarity_score >= similarity_threshold:
                        await self.vector_store.create_text_image_relation(
                            document_id=doc_id,
                            image_id=img_id,
                            similarity_score=similarity_score,
                            relation_type="auto_generated"
                        )
                        created_count += 1
                        self.stats.created_relations += 1
                        
                except Exception as e:
                    logger.error(f"Failed to create relation between {doc_id} and {img_id}: {e}")
        
        logger.info(f"Created {created_count} text-image relations")
        return created_count
    
    async def process_directory(self, text_dir: Optional[str] = None,
                              image_dir: Optional[str] = None,
                              metadata_file: Optional[str] = None) -> ProcessingStats:
        """处理整个目录"""
        # 加载元数据
        metadata = {}
        if metadata_file and Path(metadata_file).exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        
        # 处理文本文件
        text_doc_ids = []
        if text_dir:
            text_dir_path = Path(text_dir)
            if text_dir_path.exists():
                text_files = list(text_dir_path.glob("*.txt")) + list(text_dir_path.glob("*.md"))
                text_files = [str(f) for f in text_files]
                
                if text_files:
                    logger.info(f"Found {len(text_files)} text files in {text_dir}")
                    text_doc_ids = await self.process_text_batch(text_files)
        
        # 处理图像文件
        image_results = []
        if image_dir:
            image_dir_path = Path(image_dir)
            if image_dir_path.exists():
                image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp']
                image_files = []
                for ext in image_extensions:
                    image_files.extend(image_dir_path.glob(ext))
                image_files = [str(f) for f in image_files]
                
                if image_files:
                    logger.info(f"Found {len(image_files)} image files in {image_dir}")
                    image_results = await self.process_image_batch(image_files)
        
        # 提取图像ID
        image_ids = [result[0] for result in image_results if result]
        
        # 创建关联
        if text_doc_ids and image_ids:
            await self.create_text_image_relations(text_doc_ids, image_ids)
        
        return self.stats
    
    def _get_image_size(self, image_path: str) -> Optional[str]:
        """获取图像尺寸（简化版本）"""
        try:
            # 这里可以添加实际的图像尺寸获取逻辑
            # 目前返回占位符
            return "unknown"
        except Exception:
            return None
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """计算文件哈希值"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return "unknown"
    
    def print_stats(self):
        """打印统计信息"""
        print("\n" + "="*50)
        print("数据处理统计")
        print("="*50)
        print(f"文本文件: {self.stats.processed_texts}/{self.stats.total_texts} "
              f"({self.stats.text_success_rate*100:.1f}%)")
        print(f"图像文件: {self.stats.processed_images}/{self.stats.total_images} "
              f"({self.stats.image_success_rate*100:.1f}%)")
        print(f"文本-图像关联: {self.stats.created_relations} created")
        print(f"总耗时: {self.stats.elapsed_time:.2f}秒")
        print("="*50)


class MetadataGenerator:
    """元数据生成器"""
    
    @staticmethod
    def generate_campus_metadata() -> Dict[str, Any]:
        """生成校园相关元数据"""
        return {
            "domain": "campus_navigation",
            "data_type": "campus_info",
            "language": "zh-CN",
            "version": "1.0.0",
            "tags": ["campus", "university", "navigation", "education"]
        }
    
    @staticmethod
    def generate_building_metadata(building_name: str, building_type: str) -> Dict[str, Any]:
        """生成建筑相关元数据"""
        return {
            "building_name": building_name,
            "building_type": building_type,
            "category": "campus_building",
            "tags": ["building", "campus", building_type.lower()]
        }
    
    @staticmethod
    def generate_image_metadata(image_type: str, location: Optional[str] = None) -> Dict[str, Any]:
        """生成图像相关元数据"""
        metadata = {
            "image_type": image_type,
            "category": "campus_image",
            "tags": ["image", "campus", image_type.lower()]
        }
        
        if location:
            metadata["location"] = location
        
        return metadata


# 工具函数
async def create_pipeline() -> DataProcessingPipeline:
    """创建数据处理管道实例"""
    async with OllamaService() as ollama_service:
        vector_store = PostgreSQLVectorStore()
        await vector_store.connect()
        return DataProcessingPipeline(ollama_service, vector_store)


async def process_campus_data(text_dir: str, image_dir: str, output_dir: str):
    """处理校园数据的主函数"""
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 创建管道
    pipeline = await create_pipeline()
    
    # 处理数据
    stats = await pipeline.process_directory(
        text_dir=text_dir,
        image_dir=image_dir
    )
    
    # 打印统计信息
    pipeline.print_stats()
    
    # 保存处理统计
    stats_file = output_path / "processing_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump({
            "text_files_processed": stats.processed_texts,
            "image_files_processed": stats.processed_images,
            "relations_created": stats.created_relations,
            "elapsed_time_seconds": stats.elapsed_time,
            "text_success_rate": stats.text_success_rate,
            "image_success_rate": stats.image_success_rate
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Processing completed. Stats saved to {stats_file}")