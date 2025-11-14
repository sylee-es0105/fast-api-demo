from enum import Enum

# TODO enum 대신에 langchain에 기능 있는지 찾아보기
class VectorDBProvider(str, Enum):
    CHROMA = "chroma"

class LLMProvider(str, Enum):
    UPSTAGE = "upstage"
    OPENAI = "openai"

class UpstageEmbeddingModel(str, Enum):
    EMBEDDING_QUERY = "embedding-query"

class OpenAIEmbeddingModel(str, Enum):
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"
