import logging
import threading
import time

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

from app.config import settings
from app.models.llm import LLMResponse

logger = logging.getLogger("rag.llm")

# Only these are worth retrying. Auth errors / bad requests fail fast.
_TRANSIENT = (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)


class LLMError(RuntimeError):
    """Base class for language-model dependency failures."""


class LLMNotConfiguredError(LLMError):
    """Raised when no LLM credentials are configured."""


class LLMUnavailableError(LLMError):
    """Raised when the provider times out, rejects the request or is down."""


class LLMService:
    """
    LLM service for any OpenAI-compatible endpoint (OpenRouter by default).
    The model (e.g. a DeepSeek or Gemini model) is selected by configuration.

    Retry policy (ERR-003): at most LLM_MAX_RETRIES attempts in total, only for
    transient failures (connection, timeout, 429, 5xx), with 1s, 2s, 4s ...
    back-off. Every attempt is bounded by LLM_TIMEOUT_SECONDS (ERR-002).
    """

    def __init__(
        self,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        max_retries: int | None = None,
        timeout: float | None = None,
    ):
        if not settings.OPENROUTER_API_KEY:
            raise LLMNotConfiguredError("OPENROUTER_API_KEY is not configured.")

        self.client = OpenAI(
            base_url=settings.LLM_BASE_URL,
            api_key=settings.OPENROUTER_API_KEY,
            timeout=timeout if timeout is not None else settings.LLM_TIMEOUT_SECONDS,
            max_retries=0,  # we run our own bounded retry loop below
        )
        self.model = settings.LLM_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens if max_tokens is not None else settings.LLM_MAX_TOKENS
        self.max_retries = max(
            1, max_retries if max_retries is not None else settings.LLM_MAX_RETRIES
        )

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        last_exception: Exception | None = None
        messages = [{"role": "user", "content": prompt}]
        if system:
            messages.insert(0, {"role": "system", "content": system})

        for attempt in range(self.max_retries):
            try:
                start = time.perf_counter()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature if temperature is None else temperature,
                    max_tokens=self.max_tokens if max_tokens is None else max_tokens,
                )
                latency_ms = (time.perf_counter() - start) * 1000

                text = (response.choices[0].message.content or "").strip()
                if not text:
                    raise LLMUnavailableError("The model returned an empty response.")

                return LLMResponse(text=text, latency_ms=latency_ms, model=self.model)

            except _TRANSIENT as e:
                last_exception = e
                logger.warning(
                    "llm_transient_failure attempt=%d/%d error=%s",
                    attempt + 1, self.max_retries, type(e).__name__,
                )
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)

            except LLMUnavailableError:
                raise

            except Exception as e:  # auth, bad request, model not found ...
                # Class name and status only: the full error can echo request content.
                logger.error(
                    "llm_request_rejected error=%s status=%s",
                    type(e).__name__, getattr(e, "status_code", "-"),
                )
                raise LLMUnavailableError(
                    "The language-model provider rejected the request."
                ) from e

        raise LLMUnavailableError(
            f"The language-model provider is unavailable after {self.max_retries} attempts."
        ) from last_exception


_llm_service: LLMService | None = None
_llm_lock = threading.Lock()


def get_llm_service() -> LLMService:
    """Create the shared LLM service on first use (never at import time)."""
    global _llm_service
    if _llm_service is None:
        with _llm_lock:
            if _llm_service is None:
                _llm_service = LLMService()
    return _llm_service
