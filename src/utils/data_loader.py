import logging
import os
from typing import List

from llama_index.core import (
    SimpleDirectoryReader
)
from llama_index.core.schema import Document

logger = logging.getLogger(__name__)


class AutomotiveLoader:
    """汽车行业数据加载器"""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load_data(self) -> List[Document]:
        logger.info(f"Loading documents...{self.data_dir}")
        logger.info(os.path.exists(self.data_dir))
        documents = SimpleDirectoryReader(
            self.data_dir,
            recursive=True,
            required_exts=[".pdf", ".docx"]  # 支持的格式
        ).load_data()
        if not documents:
            logger.info("未找到任何文档，请确保在 data/raw/")
        return documents
