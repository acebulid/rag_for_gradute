from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # 数据库配置 (Docker PostgreSQL)
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "mydb"
    postgres_user: str = "postgres"
    postgres_password: str = "11111111"
    
    # Ollama配置
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "bge-m3:latest"
    vlm_model: str = "qwen2.5vl:7b"
    llm_model: str = "qwen3:8b"
    
    # 向量维度
    embedding_dimension: int = 1024  # BGE3维度
    
    # 应用配置
    log_level: str = "INFO"
    max_retries: int = 3
    batch_size: int = 10
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    @property
    def database_url(self) -> str:
        """构建数据库连接URL"""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
    
    @property
    def async_database_url(self) -> str:
        """构建异步数据库连接URL"""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# 全局配置实例
settings = Settings()