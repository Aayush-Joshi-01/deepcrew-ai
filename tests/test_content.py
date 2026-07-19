from __future__ import annotations

import base64

import pytest

from deepcrew.content import (
    MAX_IMAGE_BYTES,
    MAX_PDF_BYTES,
    DocumentPart,
    ImagePart,
    TextPart,
    describe_attachments,
    extract_text,
    image,
    pdf,
    user_message,
)
from deepcrew.exceptions import ContentError

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
JPEG_BYTES = b"\xff\xd8\xff" + b"\x00" * 16
GIF_BYTES = b"GIF89a" + b"\x00" * 16
WEBP_BYTES = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"\x00" * 8
PDF_BYTES = b"%PDF-1.4\n" + b"\x00" * 16


# --- image() -----------------------------------------------------------


def test_image_passthrough_url():
    part = image("https://example.com/cat.png")
    assert part == ImagePart(url="https://example.com/cat.png")


def test_image_passthrough_data_uri():
    part = image("data:image/png;base64,AAAA")
    assert part.url == "data:image/png;base64,AAAA"


def test_image_bytes_round_trip_png():
    part = image(PNG_BYTES)
    assert part.url.startswith("data:image/png;base64,")
    encoded = part.url.split(",", 1)[1]
    assert base64.b64decode(encoded) == PNG_BYTES


@pytest.mark.parametrize(
    "data,expected_mime",
    [
        (PNG_BYTES, "image/png"),
        (JPEG_BYTES, "image/jpeg"),
        (GIF_BYTES, "image/gif"),
        (WEBP_BYTES, "image/webp"),
    ],
)
def test_image_mime_sniffing(data, expected_mime):
    part = image(data)
    assert part.url.startswith(f"data:{expected_mime};base64,")


def test_image_unknown_type_raises():
    with pytest.raises(ContentError, match="Could not determine image type"):
        image(b"not an image")


def test_image_explicit_mime_overrides_sniffing():
    part = image(b"not an image", mime="image/bmp")
    assert part.url.startswith("data:image/bmp;base64,")


def test_image_oversize_raises(monkeypatch):
    monkeypatch.setattr("deepcrew.content.MAX_IMAGE_BYTES", 4)
    with pytest.raises(ContentError, match="exceeds"):
        image(PNG_BYTES)


def test_image_missing_file_raises(tmp_path):
    missing = tmp_path / "nope.png"
    with pytest.raises(ContentError, match="not found"):
        image(missing)


def test_image_from_file(tmp_path):
    f = tmp_path / "pic.png"
    f.write_bytes(PNG_BYTES)
    part = image(f)
    assert part.url.startswith("data:image/png;base64,")


def test_image_detail_passed_through():
    part = image("https://example.com/cat.png", detail="high")
    assert part.detail == "high"
    assert part.to_block()["image_url"]["detail"] == "high"


def test_image_max_bytes_constant():
    assert MAX_IMAGE_BYTES == 20 * 1024 * 1024


# --- pdf() ---------------------------------------------------------------


def test_pdf_passthrough_url():
    part = pdf("https://example.com/report.pdf")
    assert isinstance(part, DocumentPart)
    assert part.data_url == "https://example.com/report.pdf"


def test_pdf_bytes_round_trip():
    part = pdf(PDF_BYTES)
    assert part.data_url.startswith("data:application/pdf;base64,")
    encoded = part.data_url.split(",", 1)[1]
    assert base64.b64decode(encoded) == PDF_BYTES


def test_pdf_non_pdf_bytes_raises():
    with pytest.raises(ContentError, match="does not look like a PDF"):
        pdf(b"not a pdf at all")


def test_pdf_oversize_raises(monkeypatch):
    monkeypatch.setattr("deepcrew.content.MAX_PDF_BYTES", 4)
    with pytest.raises(ContentError, match="exceeds"):
        pdf(PDF_BYTES)


def test_pdf_missing_file_raises(tmp_path):
    missing = tmp_path / "nope.pdf"
    with pytest.raises(ContentError, match="not found"):
        pdf(missing)


def test_pdf_filename_defaults_to_path_name(tmp_path):
    f = tmp_path / "report.pdf"
    f.write_bytes(PDF_BYTES)
    part = pdf(f)
    assert part.filename == "report.pdf"


def test_pdf_explicit_filename():
    part = pdf(PDF_BYTES, filename="custom.pdf")
    assert part.filename == "custom.pdf"


def test_pdf_max_bytes_constant():
    assert MAX_PDF_BYTES == 32 * 1024 * 1024


# --- to_block() shapes -----------------------------------------------------


def test_text_part_block():
    assert TextPart("hi").to_block() == {"type": "text", "text": "hi"}


def test_image_part_block():
    block = ImagePart(url="https://x/y.png").to_block()
    assert block == {"type": "image_url", "image_url": {"url": "https://x/y.png"}}


def test_document_part_block():
    block = DocumentPart(data_url="data:application/pdf;base64,AA", filename="r.pdf").to_block()
    assert block == {
        "type": "file",
        "file": {"file_data": "data:application/pdf;base64,AA", "filename": "r.pdf"},
    }


# --- user_message() -----------------------------------------------------


def test_user_message_plain_string():
    msg = user_message("hello")
    assert msg == {"role": "user", "content": "hello"}


def test_user_message_with_attachments():
    img = ImagePart(url="https://x/y.png")
    msg = user_message("describe this", img)
    assert msg["role"] == "user"
    assert msg["content"] == [
        {"type": "text", "text": "describe this"},
        {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
    ]


def test_user_message_multiple_text_parts_forces_block_list():
    msg = user_message("a", "b")
    assert msg["content"] == [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]


# --- extract_text() -----------------------------------------------------


def test_extract_text_from_string():
    assert extract_text("hello") == "hello"


def test_extract_text_from_block_list():
    content = [
        {"type": "text", "text": "part one"},
        {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
        {"type": "text", "text": "part two"},
    ]
    assert extract_text(content) == "part one\npart two"


def test_extract_text_from_none():
    assert extract_text(None) == ""


def test_extract_text_from_other_type():
    assert extract_text(42) == ""


# --- describe_attachments() -----------------------------------------------


def test_describe_attachments_none():
    assert describe_attachments(None) == ""
    assert describe_attachments("plain string") == ""


def test_describe_attachments_content_parts():
    parts = [ImagePart(url="https://x/a.png"), ImagePart(url="https://x/b.png"), pdf(PDF_BYTES)]
    desc = describe_attachments(parts)
    assert desc == "[attachments: 2 images, 1 document]"


def test_describe_attachments_single_image_singular():
    desc = describe_attachments([ImagePart(url="https://x/a.png")])
    assert desc == "[attachments: 1 image]"


def test_describe_attachments_content_blocks():
    blocks = [
        {"type": "text", "text": "hi"},
        {"type": "image_url", "image_url": {"url": "https://x/a.png"}},
        {"type": "file", "file": {"file_data": "data:application/pdf;base64,AA"}},
    ]
    assert describe_attachments(blocks) == "[attachments: 1 image, 1 document]"


def test_describe_attachments_empty_list():
    assert describe_attachments([]) == ""
