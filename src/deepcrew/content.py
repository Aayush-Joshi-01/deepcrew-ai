"""
Multimodal content parts for deepcrew agents.

Build OpenAI/LiteLLM-format content blocks for images and PDF documents so
they can be attached to a user message alongside text::

    from deepcrew import image, pdf, user_message

    msg = user_message(
        "What's in this chart, and does it match the report?",
        image("chart.png"),
        pdf("report.pdf"),
    )
    result = await run_agent(agent, [msg])

Images are sent as ``image_url`` content blocks, which LiteLLM forwards to
every major provider (OpenAI, Anthropic, Gemini, Bedrock, ...). PDF/document
blocks use the ``file`` content-block shape supported by OpenAI, Anthropic,
and Gemini; providers that don't support file input will raise an error from
the underlying API — ``litellm.drop_params`` only strips unsupported
top-level parameters, it does not strip or convert content blocks, so an
unsupported attachment fails loudly rather than being silently dropped.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .exceptions import ContentError

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_PDF_BYTES = 32 * 1024 * 1024

_IMAGE_SIGNATURES: dict[bytes, str] = {
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"\xff\xd8\xff": "image/jpeg",
    b"GIF87a": "image/gif",
    b"GIF89a": "image/gif",
}


def _sniff_image_mime(data: bytes) -> str | None:
    for sig, mime in _IMAGE_SIGNATURES.items():
        if data.startswith(sig):
            return mime
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _is_data_or_url(source: str) -> bool:
    return source.startswith(("http://", "https://", "data:"))


def _read_bytes(source: str | Path | bytes, max_bytes: int, kind: str) -> bytes:
    if isinstance(source, bytes):
        data = source
    else:
        path = Path(source)
        if not path.is_file():
            raise ContentError(f"{kind} file not found: {source!r}")
        data = path.read_bytes()
    if len(data) > max_bytes:
        raise ContentError(f"{kind} of {len(data)} bytes exceeds the {max_bytes}-byte limit")
    return data


def _encode(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


@dataclass(frozen=True)
class TextPart:
    """Plain text content block."""

    text: str

    def to_block(self) -> dict:
        return {"type": "text", "text": self.text}


@dataclass(frozen=True)
class ImagePart:
    """An image attached as an OpenAI-style ``image_url`` content block."""

    url: str
    detail: str | None = None

    def to_block(self) -> dict:
        image_url: dict[str, str] = {"url": self.url}
        if self.detail:
            image_url["detail"] = self.detail
        return {"type": "image_url", "image_url": image_url}


@dataclass(frozen=True)
class DocumentPart:
    """A PDF document attached as a ``file`` content block.

    Supported by OpenAI, Anthropic, and Gemini via LiteLLM. Providers without
    file-input support will reject the request with an API error.
    """

    data_url: str
    filename: str = "document.pdf"

    def to_block(self) -> dict:
        return {"type": "file", "file": {"file_data": self.data_url, "filename": self.filename}}


ContentPart = TextPart | ImagePart | DocumentPart


def image(
    source: str | Path | bytes, *, detail: str | None = None, mime: str | None = None
) -> ImagePart:
    """
    Build an :class:`ImagePart` from a URL, a local file path, or raw bytes.

    A string starting with ``http://``, ``https://``, or ``data:`` is passed
    through untouched. Anything else is read from disk (or used directly, for
    ``bytes``), sniffed for its image type by magic bytes, size-checked
    against :data:`MAX_IMAGE_BYTES`, and base64-encoded as a ``data:`` URI.

    Raises :class:`ContentError` if the file is missing, too large, or not a
    recognizable image type (unless ``mime=`` is given explicitly).
    """
    if isinstance(source, str) and _is_data_or_url(source):
        return ImagePart(url=source, detail=detail)

    data = _read_bytes(source, MAX_IMAGE_BYTES, "image")
    resolved_mime = mime or _sniff_image_mime(data)
    if resolved_mime is None:
        raise ContentError(
            "Could not determine image type from content; pass mime= explicitly "
            "(supported: png, jpeg, gif, webp)"
        )
    return ImagePart(url=_encode(data, resolved_mime), detail=detail)


def pdf(source: str | Path | bytes, *, filename: str | None = None) -> DocumentPart:
    """
    Build a :class:`DocumentPart` from a URL, a local PDF file path, or raw bytes.

    A string starting with ``http://``, ``https://``, or ``data:`` is passed
    through untouched. Anything else is read from disk (or used directly, for
    ``bytes``), validated as a PDF (``%PDF`` header), size-checked against
    :data:`MAX_PDF_BYTES`, and base64-encoded as a ``data:`` URI.

    File/document content blocks are supported by OpenAI, Anthropic, and
    Gemini through LiteLLM. Sending a PDF to a provider without file support
    raises an error from the underlying API — this function does not attempt
    to detect provider support ahead of time.

    Raises :class:`ContentError` if the file is missing, too large, or not a
    valid PDF (unless the source is already a URL/data URI).
    """
    if isinstance(source, str) and _is_data_or_url(source):
        return DocumentPart(data_url=source, filename=filename or "document.pdf")

    data = _read_bytes(source, MAX_PDF_BYTES, "pdf")
    if not data.startswith(b"%PDF"):
        raise ContentError("Content does not look like a PDF (missing %PDF header)")

    resolved_name = filename
    if resolved_name is None and isinstance(source, (str, Path)):
        resolved_name = Path(source).name
    return DocumentPart(
        data_url=_encode(data, "application/pdf"), filename=resolved_name or "document.pdf"
    )


def user_message(*parts: str | ContentPart) -> dict:
    """
    Build an OpenAI-format user message from a mix of plain strings and
    content parts (:func:`image`, :func:`pdf`, or a bare :class:`TextPart`).

    A single plain string produces a plain string ``content`` field; any
    attachment forces the OpenAI multi-part ``content`` list format.
    """
    if len(parts) == 1 and isinstance(parts[0], str):
        return {"role": "user", "content": parts[0]}

    blocks = []
    for p in parts:
        block_part = TextPart(p) if isinstance(p, str) else p
        blocks.append(block_part.to_block())
    return {"role": "user", "content": blocks}


def extract_text(content: object) -> str:
    """
    Return the plain-text portion of a message ``content`` value.

    Handles the three shapes that appear in deepcrew message histories: a
    plain string (returned as-is), a list of content blocks (text blocks are
    joined with newlines), or ``None``/anything else (returns ``""``).
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [
            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
        ]
        return "\n".join(t for t in texts if t)
    return ""


def describe_attachments(content_or_parts: str | list[Any] | None) -> str:
    """
    Return a short human-readable summary of attachments, e.g.
    ``"[attachments: 2 images, 1 document]"``, or ``""`` if there are none.

    Accepts either a message ``content`` value (str/list-of-blocks) or a list
    of :class:`ContentPart` objects.
    """
    if isinstance(content_or_parts, str) or content_or_parts is None:
        return ""

    n_images = 0
    n_docs = 0
    for item in content_or_parts:
        if isinstance(item, ImagePart):
            n_images += 1
        elif isinstance(item, DocumentPart):
            n_docs += 1
        elif isinstance(item, dict):
            if item.get("type") == "image_url":
                n_images += 1
            elif item.get("type") == "file":
                n_docs += 1

    if not n_images and not n_docs:
        return ""

    parts = []
    if n_images:
        parts.append(f"{n_images} image{'s' if n_images != 1 else ''}")
    if n_docs:
        parts.append(f"{n_docs} document{'s' if n_docs != 1 else ''}")
    return f"[attachments: {', '.join(parts)}]"
