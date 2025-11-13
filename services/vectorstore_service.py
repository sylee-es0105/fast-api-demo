from langchain_upstage import UpstageEmbeddings
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStore
from core.config import get_settings
from core.enums import VectorDBProvider


class VectorStoreService:
    _instance = None
    _database: VectorStore | None = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorStoreService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._database is None:
            self._database = self._load_vector_db()
    
    def _load_chroma_db(self) -> VectorStore:
        embeddings = UpstageEmbeddings(
            model = get_settings().get_embedding_model_name()
        )
        
        # 임시
        database = Chroma(
            persist_directory = "./vectorstore/chroma",
            embedding_function = embeddings,
            collection_name = "chroma-tax"
        )
        return database
    
    def _load_vector_db(self) -> VectorStore:
        vector_db_provider = get_settings().VECTOR_DB_PROVIDER
        
        # 다른 DB 사용 시, load 메서드를 추가한 후 여기에 매핑하기
        db_loaders = {
            VectorDBProvider.CHROMA: self._load_chroma_db,
        }
        
        loader = db_loaders.get(vector_db_provider)
        if not loader:
            raise ValueError(
                f"Unknown vector DB provider: {vector_db_provider}. "
                f"Available options: {', '.join(db_loaders.keys())}"
            )
        
        print(f"Loading {vector_db_provider} vector database...")
        return loader()
    
    def get_vectorstore(self) -> VectorStore:
        return self._database
    
    def search(self, query: str, k: int = 4):
        return self._database.similarity_search(query, k)
    
vector_store_service = VectorStoreService()