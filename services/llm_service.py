from langchain_upstage import ChatUpstage
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from core.enums import LLMProvider
from core.config import get_settings
from dotenv import load_dotenv
load_dotenv()

class LlmService:
    _instance = None
    _llm: BaseChatModel | None = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LlmService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._llm is None:
            self._llm = self._load_llm()

    def _load_upstage_llm(self) -> BaseChatModel:
        # !!! .env UPSTAGE_API_KEY 설정 필요
        llm = ChatUpstage()

        return llm
    
    def _load_openai_llm(self) -> BaseChatModel:
        # !!! .env OPENAI_API_KEY 설정 필요
        llm = ChatOpenAI()
        return llm

    def _load_llm(self) -> BaseChatModel:
        llm_provider = get_settings().LLM_PROVIDER

        # 다른 LLM 사용 시, load 메서드를 추가한 후 여기에 매핑하기
        llm_loaders = {
            LLMProvider.UPSTAGE: self._load_upstage_llm,
            LLMProvider.OPENAI: self._load_openai_llm,
        }

        loader = llm_loaders.get(llm_provider)
        if not loader:
            raise ValueError(
                f"Unknown LLM provider: {llm_provider}."
            )

        print(f"Loading {llm_provider} LLM model...")
        return loader()
    
    def get_llm(self) -> BaseChatModel:
        return self._llm
    
llm_service = LlmService()