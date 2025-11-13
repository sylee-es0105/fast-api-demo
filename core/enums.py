from enum import Enum

class VectorDBProvider(str, Enum):
    CHROMA = "chroma"

class LLMProvider(str, Enum):
    UPSTAGE = "upstage"
    OPENAI = "openai"

class UpstageEmbeddingModel(str, Enum):
    EMBEDDING_QUERY = "embedding-query"

class OpenAIEmbeddingModel(str, Enum):
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"
