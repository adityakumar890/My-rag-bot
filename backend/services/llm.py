"""Create the configured LLM (Gemini, OpenAI, or an offline fake)."""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from backend.config import Settings, get_settings


class LLMConfigurationError(RuntimeError):
    """Raised when the LLM provider is misconfigured (e.g. missing API key)."""


def get_llm(settings: Settings | None = None) -> BaseChatModel:
    settings = settings or get_settings()
    provider = settings.llm_provider.lower().strip()

    if provider == "gemini":
        if not settings.gemini_api_key:
            raise LLMConfigurationError(
                "GEMINI_API_KEY is not set. Add it to your .env file."
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )

    if provider == "openai":
        if not settings.openai_api_key:
            raise LLMConfigurationError(
                "OPENAI_API_KEY is not set. Add it to your .env file."
            )
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )

    if provider == "fake":
        # Offline model for tests and demos: always returns the same reply.
        return FakeListChatModel(
            responses=["(fake LLM) This answer is a placeholder. Set LLM_PROVIDER=gemini for real answers."]
        )

    raise LLMConfigurationError(
        f"Unknown LLM_PROVIDER '{settings.llm_provider}'. Use gemini, openai or fake."
    )
