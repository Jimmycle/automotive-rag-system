from llama_index.core import StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore

from config import config, Config
from src.utils.data_loader import AutomotiveLoader

if __name__ == '__main__':
    pass
    # vector_store = MilvusVectorStore(
    #     uri="http://localhost:19530",
    #     collection_name="collection_name",
    #     dim=1024,  # 自动获取维度
    #     overwrite=False,  # 不覆盖已有集合
    # )
    # storage_context = StorageContext.from_defaults(vector_store=vector_store)
    # print(storage_context)
    # data_loader = AutomotiveLoader("../data/raw")
    # documents = data_loader.load_data()
    # print(f"documents:{type(documents)}")
    # vector_manager = VectorStoreManager(
    #     collection_name=Config.COLLECTION_NAME,
    #     chunk_size=Config.CHUNK_SIZE,
    #     chunk_overlap=Config.CHUNK_OVERLAP
    # )
    # index = vector_manager.create_index_from_documents(documents)