import pytest
from core.config import Settings
from core.enums import LLMProvider, OpenAIEmbeddingModel, UpstageEmbeddingModel


class TestSettingsEmbeddingModel:
    """Test cases for get_embedding_model() method in Settings class"""

    def test_upstage_with_valid_model(self):
        """Test UPSTAGE provider with a valid embedding model returns the model value"""
        settings = Settings(
            LLM_PROVIDER=LLMProvider.UPSTAGE,
            UPSTAGE_EMBEDDING_MODEL=UpstageEmbeddingModel.EMBEDDING_QUERY
        )
        assert settings.get_embedding_model() == "embedding-query"

    def test_upstage_with_none_model_raises_error(self):
        """Test UPSTAGE provider with None embedding model raises ValueError"""
        settings = Settings(
            LLM_PROVIDER=LLMProvider.UPSTAGE,
            UPSTAGE_EMBEDDING_MODEL=None
        )
        with pytest.raises(ValueError) as exc_info:
            settings.get_embedding_model()
        assert "UPSTAGE_EMBEDDING_MODEL is not configured" in str(exc_info.value)

    def test_openai_with_valid_model(self):
        """Test OPENAI provider with a valid embedding model returns the model value"""
        settings = Settings(
            LLM_PROVIDER=LLMProvider.OPENAI,
            OPENAI_EMBEDDING_MODEL=OpenAIEmbeddingModel.TEXT_EMBEDDING_3_LARGE
        )
        assert settings.get_embedding_model() == "text-embedding-3-large"

    def test_openai_with_none_model_raises_error(self):
        """Test OPENAI provider with None embedding model raises ValueError"""
        settings = Settings(
            LLM_PROVIDER=LLMProvider.OPENAI,
            OPENAI_EMBEDDING_MODEL=None
        )
        with pytest.raises(ValueError) as exc_info:
            settings.get_embedding_model()
        assert "OPENAI_EMBEDDING_MODEL is not configured" in str(exc_info.value)

    def test_default_settings(self):
        """Test default settings (UPSTAGE provider with EMBEDDING_QUERY model)"""
        settings = Settings()
        assert settings.LLM_PROVIDER == LLMProvider.UPSTAGE
        assert settings.UPSTAGE_EMBEDDING_MODEL == UpstageEmbeddingModel.EMBEDDING_QUERY
        assert settings.get_embedding_model() == "embedding-query"

    def test_openai_embedding_model_can_be_none(self):
        """Test that OPENAI_EMBEDDING_MODEL field can accept None value"""
        settings = Settings(OPENAI_EMBEDDING_MODEL=None)
        assert settings.OPENAI_EMBEDDING_MODEL is None

    def test_upstage_embedding_model_can_be_none(self):
        """Test that UPSTAGE_EMBEDDING_MODEL field can accept None value"""
        settings = Settings(UPSTAGE_EMBEDDING_MODEL=None)
        assert settings.UPSTAGE_EMBEDDING_MODEL is None
