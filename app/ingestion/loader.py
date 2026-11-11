from pathlib import Path
from typing import List

from bs4 import BeautifulSoup
from langchain_core.documents import Document

from app.config import settings


class FileValidationError(ValueError):
    """The file failed type, size or content validation (IF-002, NFR-SEC-008)."""


class DocumentLoader:
    """
    Loads documents of different formats and converts them into
    LangChain Document objects.
    """

    # The supported formats (IF-001, ING-001). Anything else is rejected.
    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".html", ".htm"}

    @classmethod
    def is_supported(cls, filename: str) -> bool:
        return Path(filename).suffix.lower() in cls.SUPPORTED_EXTENSIONS

    @classmethod
    def validate(cls, path: Path) -> None:
        """Type, size and content checks performed before any processing."""
        suffix = path.suffix.lower()
        if suffix not in cls.SUPPORTED_EXTENSIONS:
            raise FileValidationError(f"Unsupported file type: {suffix or '(none)'}")

        size = path.stat().st_size
        if size == 0:
            raise FileValidationError("The file is empty.")
        if size > settings.MAX_UPLOAD_MB * 1024 * 1024:
            raise FileValidationError(f"File exceeds the {settings.MAX_UPLOAD_MB} MB limit.")

        with path.open("rb") as f:
            head = f.read(8192)

        if suffix == ".pdf":
            if b"%PDF-" not in head[:1024]:
                raise FileValidationError("The file is not a valid PDF.")
        elif not head.startswith((b"\xff\xfe", b"\xfe\xff")) and b"\x00" in head:
            # NUL bytes never occur in text/markdown/html: this is a binary
            # file with a text extension.
            raise FileValidationError("The file content does not match its text file type.")

    @staticmethod
    def _read_text(path: Path) -> str:
        raw = path.read_bytes()
        # UTF-16 decodes almost any byte string, so only trust it with a BOM.
        if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
            return raw.decode("utf-16")
        for encoding in ("utf-8-sig", "cp1252"):
            try:
                return raw.decode(encoding)
            except UnicodeError:
                continue
        return raw.decode("latin-1")  # never fails

    def load(self, file_path: str) -> List[Document]:
        path = Path(file_path)

        if not path.is_file():
            raise FileNotFoundError(f"{file_path} not found.")

        self.validate(path)
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            # Imported lazily: keeps start-up light and tests PyMuPDF-free.
            from langchain_community.document_loaders import PyMuPDFLoader

            try:
                return PyMuPDFLoader(str(path)).load()
            except Exception as e:
                raise FileValidationError("The PDF could not be parsed.") from e

        text = self._read_text(path)

        if suffix in (".html", ".htm"):
            soup = BeautifulSoup(text, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            text = soup.get_text("\n")

        return [Document(page_content=text, metadata={"source": str(path)})]
