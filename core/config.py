from pydantic_settings import BaseSettings
from core.enums import LLMProvider, VectorDBProvider, UpstageEmbeddingModel, OpenAIEmbeddingModel

class Settings(BaseSettings):
    VECTOR_DB_PROVIDER: VectorDBProvider = VectorDBProvider.CHROMA
    LLM_PROVIDER: LLMProvider = LLMProvider.UPSTAGE

    OPENAI_EMBEDDING_MODEL: OpenAIEmbeddingModel | None = None
    UPSTAGE_EMBEDDING_MODEL: UpstageEmbeddingModel | None = UpstageEmbeddingModel.EMBEDDING_QUERY

    def get_embedding_model(self) -> str:
        if self.LLM_PROVIDER == LLMProvider.UPSTAGE:
            return self.UPSTAGE_EMBEDDING_MODEL.value
        if self.LLM_PROVIDER == LLMProvider.OPENAI:
            return self.OPENAI_EMBEDDING_MODEL.value
        return None
    
def get_settings() -> Settings:
    return Settings()