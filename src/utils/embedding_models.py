from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding

from config import Config


def get_qwen_embedding():
    return DashScopeEmbedding(
        model=Config.QWEN_EMBEDDING_V2_MODEL,
        dashscope_api_key=Config.QWEN_API_KEY
    )


def get_ollama_embedding():
    embed_model = OllamaEmbedding(
        model_name="bge-m3:latest",
        base_url="http://localhost:11434",
    )
    return embed_model
