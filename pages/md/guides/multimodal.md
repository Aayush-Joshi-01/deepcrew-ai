# Multimodal Input
`image()`, `pdf()`, and `user_message()` build standard OpenAI-format content blocks, forwarded by LiteLLM to whichever provider you're using. Sources can be a URL, a local file path, or raw bytes.

```python
from deepcrew import Agent, run_agent, image, pdf, user_message

agent = Agent(name="analyst", model="anthropic/claude-opus-4-8")

msg = user_message(
    "Summarize this chart and check it against the report.",
    image("chart.png"),   # local file, PNG/JPEG/GIF/WEBP auto-detected
    pdf("report.pdf"),
)
result = await run_agent(agent, [msg])
print(result.text)
```

> `image()`/`pdf()` also accept an `https://` URL or a `data:` URI directly — no encoding happens for those, they pass straight through. Local files and raw `bytes` are size-checked and base64-encoded automatically, raising `ContentError` on anything invalid or oversized.

### How image()/pdf() dispatch on the source

Both functions accept a `str | Path | bytes` and branch on the source's shape, not on any explicit flag you pass:

    - A string starting with `http://`, `https://`, or `data:` passes straight through, untouched — no size check, no mime sniffing, no encoding. deepcrew trusts you and the receiving provider to handle it.
    - Anything else (a plain string path, a `Path`, or raw `bytes`) is read into memory (files) or used directly (bytes), checked against a size limit, sniffed for its actual type by magic bytes, and base64-encoded into a `data:` URI.

```python
from deepcrew import image, pdf

# 1. URL — passthrough, no local I/O at all
image("https://example.com/chart.png")

# 2. data: URI — passthrough
image("data:image/png;base64,iVBORw0KGgoAAAANSU...")

# 3. Local path (str or pathlib.Path) — read, sniffed, encoded
image("./chart.png")

# 4. Raw bytes — sniffed, encoded (no filesystem access at all)
with open("chart.png", "rb") as f:
    image(f.read())

# Explicit mime override, e.g. for bytes with no recognizable magic-byte header
image(some_bytes, mime="image/bmp")

# pdf() works identically, but validates the %PDF header instead of image magic bytes
pdf("./report.pdf")
pdf(some_pdf_bytes, filename="q3-report.pdf")  # filename defaults to the source's own name
```

Image type detection recognizes PNG, JPEG, GIF, and WEBP by their leading magic bytes — nothing else. An unrecognized byte sequence with no `mime=` override raises `ContentError`. `pdf()` is stricter still: it requires the content to literally start with `%PDF`, regardless of file extension — a mislabeled `.pdf` file that isn't actually a PDF is rejected before it ever reaches the LLM.

### Size limits and ContentError

- **MAX_IMAGE_BYTES** (20 * 1024 * 1024 (20 MB)): Enforced only for local files/bytes — URL and `data:` passthrough sources are never size-checked by deepcrew itself (the provider may still reject an oversized payload).

- **MAX_PDF_BYTES** (32 * 1024 * 1024 (32 MB)): Same passthrough exemption as images.

- **ContentError** (exception): Raised for: a missing local file, content over the size limit, an unrecognized image type with no `mime=` override, or bytes/a file that doesn't start with `%PDF` when calling `pdf()`.

Both limits are plain module-level constants in `deepcrew.content` — override them process-wide if you need to (e.g. in tests) with `import deepcrew.content as content; content.MAX_IMAGE_BYTES = 5 * 1024 * 1024`. There's no per-call override parameter.

### ContentPart types

`image()` and `pdf()` return frozen dataclasses, not raw dicts — `ContentPart` is the union type covering all three:

- **TextPart(text)** (-> {"type": "text", "text": ...}): What a bare string gets coerced into inside `user_message()`. You rarely construct this directly.

- **ImagePart(url, detail=None)** (-> {"type": "image_url", "image_url": {...}}): `detail` is an OpenAI-specific hint (`"low"`/`"high"`/`"auto"`) forwarded only if set; harmless no-op on providers that ignore it.

- **DocumentPart(data_url, filename="document.pdf")** (-> {"type": "file", "file": {...}}): The `"file"` content-block shape. See the provider-support callout below.

### extract_text() and describe_attachments()

These two functions are what let the rest of deepcrew (memory injection, the verifier, the loop, the router) work with multimodal messages without needing to know about content blocks themselves.

```python
from deepcrew import extract_text, describe_attachments, image, pdf

content = [
    {"type": "text", "text": "What's in this?"},
    {"type": "image_url", "image_url": {"url": "..."}},
]
extract_text(content)          # "What's in this?" — joins only the text blocks
extract_text("plain string")   # "plain string" — passthrough
extract_text(None)             # "" — never raises

parts = [image("chart.png"), pdf("report.pdf")]
describe_attachments(parts)    # "[attachments: 1 image, 1 document]"
describe_attachments("text")   # "" — no attachments to describe
```

`extract_text()` is what the memory-injection query, the verifier's grading prompt, and the self-improving loop's convergence checks all run on internally — none of them ever see the raw image/document bytes, only whatever text accompanied them.

### With Orchestrator

```python
result = await orch.run(
    "What's in this photo, and does it match the incident report?",
    attachments=[image("scene.jpg"), pdf("incident_report.pdf")],
)
```

The router stays text-only — it never sees the raw attachments, only a summary like `[attachments: 1 image, 1 document]`, so its JSON-mode routing call is never sent binary content. The agent(s) it routes to receive the real attachments. In parallel routing, every agent gets the full attachment set, since the router can't reliably split images across sub-tasks. Spawned sub-agents (see [Agent Spawning](spawning.html)) never automatically inherit attachments either — describe what's relevant directly in the spawn task text.

### Common pitfalls

    - PDF/file-block support is not universal across providers. `image_url` blocks are forwarded by LiteLLM to essentially every vision-capable provider, but `"file"`/document blocks are only reliably supported on OpenAI, Anthropic, and Gemini. `litellm.drop_params` strips unsupported top-level parameters, not content blocks — sending a PDF to a provider without file support fails loudly with an API error, it does not silently degrade.
    - URL/data: sources bypass every safety check. No size limit, no mime sniffing, no `%PDF` validation — deepcrew trusts the string is well-formed and lets the provider be the final arbiter.
    - Only user messages carry multimodal content. Assistant and tool messages in the conversation history remain plain strings throughout deepcrew — there's no path for an agent's own output to include images.
    - Parallel orchestration duplicates attachments to every agent. A 5-agent parallel fan-out with a 10 MB PDF attached means that PDF's bytes are included in 5 separate LLM requests, not one shared reference.

### See also

    - [APEX Synthesizer](apex.html) — synthesizes text results only; it never re-sees the original attachments.
    - [Agent Spawning](spawning.html) — sub-agents don't inherit attachments automatically.
    - [FastAPI Integration](fastapi.html) — `create_stream_router` accepts attachments as URLs/data-URIs in the request body.
