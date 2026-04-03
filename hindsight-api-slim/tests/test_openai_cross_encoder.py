"""
Tests for OpenAICompatibleCrossEncoder.
"""

import os
from unittest.mock import MagicMock

import pytest

from hindsight_api.engine.cross_encoder import OpenAICompatibleCrossEncoder, create_cross_encoder_from_env


class TestOpenAICompatibleCrossEncoder:
    """Test suite for OpenAI-compatible reranker provider."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test successful initialization."""
        encoder = OpenAICompatibleCrossEncoder(
            api_key="test_key",
            model="BAAI/bge-reranker-v2-m3",
            base_url="https://api.siliconflow.cn/v1",
        )

        assert encoder.provider_name == "openai"
        assert encoder.api_key == "test_key"
        assert encoder.model == "BAAI/bge-reranker-v2-m3"
        assert encoder.base_url == "https://api.siliconflow.cn/v1"

        await encoder.initialize()

        assert encoder._async_client is not None

    @pytest.mark.asyncio
    async def test_predict(self):
        """Test prediction with OpenAI-compatible /rerank endpoint."""
        encoder = OpenAICompatibleCrossEncoder(
            api_key="test_key",
            model="BAAI/bge-reranker-v2-m3",
            base_url="https://api.siliconflow.cn/v1",
        )
        await encoder.initialize()

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"index": 0, "relevance_score": 0.91},
                {"index": 1, "relevance_score": 0.63},
            ]
        }
        encoder._async_client.post = MagicMock(return_value=mock_response)

        pairs = [
            ("What is Python?", "Python is a programming language"),
            ("What is Python?", "Python is a kind of snake"),
        ]
        scores = await encoder.predict(pairs)

        assert scores == [0.91, 0.63]
        encoder._async_client.post.assert_called_once()
        call_args = encoder._async_client.post.call_args
        assert call_args[0][0] == "https://api.siliconflow.cn/v1/rerank"
        assert call_args.kwargs["json"]["model"] == "BAAI/bge-reranker-v2-m3"
        assert call_args.kwargs["json"]["query"] == "What is Python?"
        assert call_args.kwargs["json"]["documents"] == [
            "Python is a programming language",
            "Python is a kind of snake",
        ]


class TestOpenAICrossEncoderFactory:
    """Test suite for create_cross_encoder_from_env factory function."""

    @pytest.mark.asyncio
    async def test_create_openai_from_env(self):
        """Test creating OpenAI-compatible reranker from env vars."""
        env_vars = {
            "HINDSIGHT_API_RERANKER_PROVIDER": "openai",
            "HINDSIGHT_API_RERANKER_OPENAI_API_KEY": "test_key",
            "HINDSIGHT_API_RERANKER_OPENAI_MODEL": "BAAI/bge-reranker-v2-m3",
            "HINDSIGHT_API_RERANKER_OPENAI_BASE_URL": "https://api.siliconflow.cn/v1",
        }

        original_environ = os.environ.copy()
        try:
            os.environ.clear()
            os.environ.update(env_vars)
            from hindsight_api import config

            config._config_cache = None
            encoder = create_cross_encoder_from_env()
            assert isinstance(encoder, OpenAICompatibleCrossEncoder)
            assert encoder.api_key == "test_key"
            assert encoder.model == "BAAI/bge-reranker-v2-m3"
            assert encoder.base_url == "https://api.siliconflow.cn/v1"
        finally:
            os.environ.clear()
            os.environ.update(original_environ)
            from hindsight_api import config

            config._config_cache = None

    @pytest.mark.asyncio
    async def test_create_openai_from_env_missing_api_key(self):
        """Test error when API key is missing."""
        env_vars = {
            "HINDSIGHT_API_RERANKER_PROVIDER": "openai",
            "HINDSIGHT_API_RERANKER_OPENAI_MODEL": "BAAI/bge-reranker-v2-m3",
            "HINDSIGHT_API_RERANKER_OPENAI_BASE_URL": "https://api.siliconflow.cn/v1",
        }

        original_environ = os.environ.copy()
        try:
            os.environ.clear()
            os.environ.update(env_vars)
            from hindsight_api import config

            config._config_cache = None
            with pytest.raises(ValueError, match="HINDSIGHT_API_RERANKER_OPENAI_API_KEY is required"):
                create_cross_encoder_from_env()
        finally:
            os.environ.clear()
            os.environ.update(original_environ)
            from hindsight_api import config

            config._config_cache = None
