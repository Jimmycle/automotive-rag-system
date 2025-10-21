import os
from typing import Optional

from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI

from config import Config


class LLMManager:
    """LLM管理类，支持多种模型"""

    def __init__(self,
                 model_type: str = Config.MODEL_TYPE_OPENAI,
                 model_name: str = Config.MODEL_TYPE_OPENAI,
                 base_url: Optional[str] = None,
                 temperature: float = 0.7,
                 **kwargs):
        """
        初始化LLM管理器

        Args:
            model_type: 模型类型，支持 "openai", "qwen", "deepseek", "ollama", "local"
            model_name: 模型名称
            base_url: API基础URL（对于非OpenAI官方模型）
            temperature: 温度参数
            **kwargs: 其他模型特定参数
        """
        self.model_type = model_type
        self.model_name = model_name
        self.api_key = Config.OPENAI_API_KEY
        self.base_url = base_url
        self.temperature = temperature
        self.kwargs = kwargs
        self.llm = self._initialize_llm()

    def _initialize_llm(self):
        """根据模型类型初始化LLM"""
        if self.model_type == Config.MODEL_TYPE_OPENAI:
            return self._init_openai()
        elif self.model_type == Config.MODEL_TYPE_QWEN:
            return self._init_qwen()
        elif self.model_type == Config.MODEL_TYPE_DEEPSEEK:
            return self._init_deepseek()
        elif self.model_type == Config.MODEL_TYPE_OLLAMA:
            return self._init_ollama()
        elif self.model_type == Config.MODEL_TYPE_LOCAL:
            return self._init_local()
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

    def _init_openai(self):
        """初始化OpenAI模型"""
        if not self.api_key:
            self.api_key = Config.OPENAI_API_KEY
            if not self.api_key:
                raise ValueError("OpenAI API密钥未提供")

        return ChatOpenAI(
            openai_api_key=self.api_key,
            model_name=self.model_name,
            temperature=self.temperature,
            **self.kwargs
        )

    def _init_qwen(self):
        """初始化通义千问模型"""
        if not self.api_key:
            self.api_key = Config.QWEN_API_KEY
            if not self.api_key:
                raise ValueError("Qwen API密钥未提供")

        # 如果没有提供base_url，使用默认的DashScope端点
        if not self.base_url:
            self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

        return ChatOpenAI(
            openai_api_key=self.api_key,
            model_name=self.model_name,
            base_url=self.base_url,
            temperature=self.temperature,
            **self.kwargs
        )

    def _init_deepseek(self):
        """初始化DeepSeek模型"""
        if not self.api_key:
            self.api_key = os.getenv("DEEPSEEK_API_KEY")
            if not self.api_key:
                raise ValueError("DeepSeek API密钥未提供")

        # DeepSeek API端点
        if not self.base_url:
            self.base_url = "https://api.deepseek.com/v1"

        return ChatOpenAI(
            openai_api_key=self.api_key,
            model_name=self.model_name,
            base_url=self.base_url,
            temperature=self.temperature,
            **self.kwargs
        )

    def _init_ollama(self):
        """初始化Ollama本地模型"""
        # 对于Ollama，base_url通常是本地地址
        if not self.base_url:
            self.base_url = "http://localhost:11434"

        return ChatOllama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=self.temperature,
            **self.kwargs
        )

    def _init_local(self):
        """初始化其他本地模型"""
        # 这里可以扩展支持其他本地模型接口
        # 例如使用transformers的本地模型，或者自定义的API接口

        if not self.base_url:
            raise ValueError("本地模型需要提供base_url")

        # 假设本地模型提供OpenAI兼容的API接口
        return ChatOpenAI(
            openai_api_key="sk-not-needed",  # 本地模型可能不需要真实的API密钥
            model_name=self.model_name,
            base_url=self.base_url,
            temperature=self.temperature,
            **self.kwargs
        )

    def get_llm(self):
        """获取LLM实例"""
        return self.llm

    def update_config(self, **kwargs):
        """更新LLM配置"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # 重新初始化LLM
        self.llm = self._initialize_llm()
        return self.llm
