import contextvars
import json
import logging

from app.config import settings

# Correlates every log line of one request (EVAL-005, NFR-MNT-002).
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line: ts, level, logger, request_id, message."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "message": record.getMessage(),
        }
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)


def configure_logging(level: int = logging.INFO) -> None:
    """
    Logs are `event key=value ...` lines (LOG_FORMAT=text) or JSON objects
    (LOG_FORMAT=json), always carrying the request id. Questions, answers,
    document text, passwords, tokens and API keys are never logged (EVAL-006).
    """
    handler = logging.StreamHandler()
    if settings.LOG_FORMAT == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s request_id=%(request_id)s %(message)s"
            )
        )
    handler.addFilter(RequestIdFilter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    # never log prompts/keys: keep HTTP client and model-library chatter at WARNING
    for noisy in ("httpx", "httpcore", "openai", "chromadb", "urllib3", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
