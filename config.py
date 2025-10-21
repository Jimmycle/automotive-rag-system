import os
from dataclasses import dataclass


@dataclass
class Config:
    # 数据配置
    DATA_DIR: str = "data"
    RAW_DATA_DIR: str = "data/raw"
    PROCESSED_DATA_DIR: str = "data/processed"

    # 向量数据库配置
    VECTOR_DB_PATH: str = "data/vector_db"
    VECTOR_STORAGE_PATH: str = "data/storage"
    COLLECTION_NAME: str = "automotive_knowledge"

    # 文本处理配置
    CHUNK_SIZE: int = 1024
    CHUNK_OVERLAP: int = 200

    # 不同场景的推荐配置
    chunk_configs = {
        "general_text": {
            "chunk_size": 1024,
            "chunk_overlap": 200,
            "separator": "\n"
        },
        "code_documents": {
            "chunk_size": 512,
            "chunk_overlap": 100,
            "separator": "\n\n"
        },
        "long_documents": {
            "chunk_size": 2048,
            "chunk_overlap": 300,
            "separator": "。"
        },
        "short_qa": {
            "chunk_size": 512,
            "chunk_overlap": 50,
            "separator": " "
        }
    }

    # 检索配置
    TOP_K: int = 5
    RERANK_TOP_K: int = 3
    BM25_WEIGHT: float = 0.3
    DENSE_WEIGHT: float = 0.7

    # 模型类型
    MODEL_TYPE_OPENAI: str = "openai"
    MODEL_TYPE_QWEN: str = "qwen"
    MODEL_TYPE_DEEPSEEK: str = "deepseek"
    MODEL_TYPE_OLLAMA: str = "ollama"
    MODEL_TYPE_LOCAL: str = "local"

    # OpenAI配置
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = "gpt-3.5-turbo"

    # Qwen模型配置
    QWEN_API_KEY: str = os.getenv("DASHSCOPE_API_KEY", "")
    QWEN3_MAX_MODEL: str = "qwen3-max"

    # DeepSeek模型配置
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_CHAT_MODEL: str = "deepseek-chat"

    # Embedding模型
    QWEN_EMBEDDING_V2_MODEL = "text-embedding-v2"
    QWEN_EMBEDDING_V3_MODEL = "text-embedding-v3"

    # 评估配置
    EVAL_QUESTIONS_PATH: str = "data/evaluation_questions.json"


config = Config()