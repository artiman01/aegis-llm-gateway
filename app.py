"""AegisLLM Gateway — Hugging Face Gradio Space Application Entrypoint."""

from __future__ import annotations

import inspect
import json
import os
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

# Ensure project root and src are on the python path
root_dir = Path(__file__).resolve().parent
src_dir = root_dir / "src"
for p in (str(root_dir), str(src_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

import gradio as gr
import httpx

from src.presentation.main import app

CUSTOM_CSS = """
:root {
    --body-background-fill: #0b0f19;
    --body-text-color: #f3f4f6;
    color-scheme: dark;
}
.gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background-color: #0b0f19;
    color: #f3f4f6;
}
.hero-header {
    text-align: center;
    padding: 1.5rem 0;
    margin-bottom: 1rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}
.badge-row {
    display: flex;
    justify-content: center;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-top: 0.75rem;
}
.badge {
    background: rgba(99, 102, 241, 0.2);
    color: #a5b4fc;
    border: 1px solid rgba(99, 102, 241, 0.4);
    padding: 0.25rem 0.65rem;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 600;
}
.feature-card {
    background: rgba(30, 41, 59, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
}
.nav-btn {
    padding: 0.6rem 1.2rem;
    border-radius: 8px;
    color: white;
    cursor: pointer;
    font-size: 0.9rem;
    font-weight: 500;
    border: 1px solid rgba(255, 255, 255, 0.2);
    text-decoration: none;
    display: inline-block;
    transition: background-color 0.2s;
}
.nav-btn:hover {
    filter: brightness(1.15);
}
"""

DARK_MODE_JS = """
() => {
    document.documentElement.classList.add('dark');
    document.body.classList.add('dark');
}
"""


async def send_chat_completion(
    prompt: str,
    system_prompt: str,
    model: str,
    stream: bool,
    temperature: float,
    bypass_cache: bool,
) -> AsyncIterator[tuple[str, str, str, str]]:
    """Send chat completion request to AegisLLM Gateway in-process via ASGITransport.

    Args:
        prompt: User input prompt.
        system_prompt: System role instructions.
        model: Target LLM identifier.
        stream: Whether to stream tokens via SSE.
        temperature: Sampling temperature.
        bypass_cache: If True, sends Cache-Control: no-cache.

    Yields:
        Tuples of (response_text, latency, cache_status, token_usage).
    """
    if not prompt.strip():
        yield "", "0ms", "N/A", "0 tokens"
        return

    messages: list[dict[str, str]] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": prompt.strip()})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": stream,
    }

    headers: dict[str, str] = {
        "Authorization": f"Bearer {os.getenv('AEGIS_MASTER_KEY', 'sk-aegis-master-key')}",
    }
    if bypass_cache:
        headers["Cache-Control"] = "no-cache"

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost:7860") as client:
        try:
            if stream:
                full_text = ""
                latency_str = "Streaming..."
                cache_status_str = "BYPASS (STREAM)"
                token_count = 0

                async with client.stream(
                    "POST",
                    "/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=60.0,
                ) as response:
                    latency_str = response.headers.get("X-Response-Time", "Stream active")
                    cache_status_str = response.headers.get("X-Cache-Status", "BYPASS (STREAM)")

                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        raw_data = line[6:].strip()
                        if raw_data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(raw_data)
                            choices = chunk.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    full_text += content
                                    token_count += 1
                                    yield (
                                        full_text,
                                        latency_str,
                                        cache_status_str,
                                        f"~{token_count} stream chunks",
                                    )
                        except json.JSONDecodeError:
                            continue

                yield full_text, latency_str, cache_status_str, f"~{token_count} tokens emitted"
            else:
                resp = await client.post(
                    "/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=60.0,
                )
                latency_str = resp.headers.get("X-Response-Time", "N/A")
                cache_status_str = resp.headers.get("X-Cache-Status", "MISS")

                if resp.status_code != 200:
                    error_msg = resp.text
                    try:
                        err_json = resp.json()
                        error_msg = err_json.get("error", {}).get("message", error_msg)
                    except Exception:
                        pass
                    err_out = f"Error ({resp.status_code}): {error_msg}"
                    yield err_out, latency_str, cache_status_str, "0 tokens"
                    return

                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                tokens_str = (
                    f"Prompt: {usage.get('prompt_tokens', 0)} | "
                    f"Completion: {usage.get('completion_tokens', 0)} | "
                    f"Total: {usage.get('total_tokens', 0)}"
                )
                yield content, latency_str, cache_status_str, tokens_str

        except Exception as exc:
            yield f"Exception: {exc}", "Error", "Error", "0 tokens"


def create_gradio_ui() -> tuple[gr.Blocks, dict[str, Any]]:
    """Build Gradio UI interface for AegisLLM Gateway."""
    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="blue",
        neutral_hue="slate",
    )

    mount_params = inspect.signature(gr.mount_gradio_app).parameters
    blocks_kwargs: dict[str, Any] = {"title": "🛡️ AegisLLM Gateway — Interactive Playground"}
    mount_kwargs: dict[str, Any] = {}

    if "theme" in mount_params:
        mount_kwargs["theme"] = theme
        mount_kwargs["css"] = CUSTOM_CSS
        mount_kwargs["js"] = DARK_MODE_JS
    else:
        blocks_kwargs["theme"] = theme
        blocks_kwargs["css"] = CUSTOM_CSS
        blocks_kwargs["js"] = DARK_MODE_JS

    with gr.Blocks(**blocks_kwargs) as demo:
        with gr.Column(elem_classes=["hero-header"]):
            gr.Markdown("# 🛡️ AegisLLM Gateway — Interactive Playground")
            gr.Markdown(
                "**Production-grade resilient LLM Gateway** engineered with "
                "*Hexagonal Architecture*, combining high-throughput routing, "
                "zero-downtime circuit breaking, and sub-millisecond caching."
            )
            gr.HTML(
                """
                <div class="badge-row">
                    <span class="badge">📐 ZCA Whitening (Isotropic Hypersphere)</span>
                    <span class="badge">⚡ Hedged Speculative Dispatching (P90 Tail Cut)</span>
                    <span class="badge">🔒 DFA Streaming Redaction (O(1) Memory PII Scrubber)</span>
                    <span class="badge">🧠 Two-Tier Cache (L1 SHA-256 + L2 FastEmbed)</span>
                    <span class="badge">🔌 OpenAI & Anthropic Schema Normalization</span>
                </div>
                """
            )

        with gr.Row():
            with gr.Column(scale=5):
                gr.Markdown("### 💬 Request Configuration")
                model_dropdown = gr.Dropdown(
                    choices=[
                        "gpt-4o",
                        "gpt-4o-mini",
                        "claude-3-5-sonnet-20241022",
                        "claude-3-5-haiku-20241022",
                    ],
                    value="gpt-4o",
                    label="Target Model",
                )
                system_prompt_input = gr.Textbox(
                    value="You are AegisLLM Assistant, a principal software architect.",
                    label="System Prompt",
                    lines=2,
                )
                prompt_input = gr.Textbox(
                    placeholder=(
                        "Enter prompt (e.g. test semantic cache, leak credentials sk-proj-...)..."
                    ),
                    label="User Prompt",
                    lines=4,
                )

                with gr.Row():
                    stream_checkbox = gr.Checkbox(value=True, label="Enable SSE Streaming")
                    bypass_cache_checkbox = gr.Checkbox(
                        value=False, label="Bypass Cache (Cache-Control: no-cache)"
                    )
                    temperature_slider = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature",
                    )

                with gr.Row():
                    send_btn = gr.Button("🚀 Send Request", variant="primary", scale=2)
                    clear_btn = gr.Button("🗑️ Clear", variant="secondary", scale=1)

                gr.Examples(
                    examples=[
                        [
                            "What is AegisLLM Gateway and how does it prevent cone effect?",
                            "You are AegisLLM Assistant, a principal software architect.",
                            "gpt-4o",
                            True,
                            0.7,
                            False,
                        ],
                        [
                            "My test key is sk-proj-1234567890abcdef and AWS AKIAIOSFODNN7EXAMPLE.",
                            "You are a helpful assistant.",
                            "gpt-4o",
                            True,
                            0.0,
                            True,
                        ],
                        [
                            "Who is the CEO of Apple in 2024?",
                            "You are a helpful assistant.",
                            "gpt-4o",
                            False,
                            0.0,
                            False,
                        ],
                        [
                            "Кто мэр Москвы в 2024 году?",
                            "Ты полезный ассистент.",
                            "gpt-4o",
                            False,
                            0.0,
                            False,
                        ],
                    ],
                    inputs=[
                        prompt_input,
                        system_prompt_input,
                        model_dropdown,
                        stream_checkbox,
                        temperature_slider,
                        bypass_cache_checkbox,
                    ],
                )

            with gr.Column(scale=5):
                gr.Markdown("### ⚡ Gateway Inspection & Telemetry")
                with gr.Row():
                    latency_box = gr.Textbox(
                        label="Latency (X-Response-Time)", value="—", interactive=False
                    )
                    cache_status_box = gr.Textbox(
                        label="Cache Status (X-Cache-Status)", value="—", interactive=False
                    )
                    tokens_box = gr.Textbox(label="Token Usage", value="—", interactive=False)

                textbox_kwargs: dict[str, Any] = {
                    "label": "Model Response",
                    "placeholder": "Gateway response will appear here...",
                    "lines": 12,
                }
                if "buttons" in inspect.signature(gr.Textbox.__init__).parameters:
                    textbox_kwargs["buttons"] = ["copy"]
                else:
                    textbox_kwargs["show_copy_button"] = True

                response_output = gr.Textbox(**textbox_kwargs)

                gr.Markdown("### 🔗 Architecture & API Endpoints")
                gr.HTML(
                    """
                    <div style="display: flex; gap: 0.75rem; flex-wrap: wrap;">
                        <a href="/docs" target="_blank" class="nav-btn"
                           style="background: #312e81;">
                            📖 Open Swagger API Docs (/docs)
                        </a>
                        <a href="/metrics" target="_blank" class="nav-btn"
                           style="background: #064e3b;">
                            📊 Prometheus Metrics (/metrics)
                        </a>
                        <a href="/health" target="_blank" class="nav-btn"
                           style="background: #78350f;">
                            💓 Health Probe (/health)
                        </a>
                    </div>
                    """
                )

        send_btn.click(
            fn=send_chat_completion,
            inputs=[
                prompt_input,
                system_prompt_input,
                model_dropdown,
                stream_checkbox,
                temperature_slider,
                bypass_cache_checkbox,
            ],
            outputs=[response_output, latency_box, cache_status_box, tokens_box],
        )

        clear_btn.click(
            fn=lambda: ("", "", "—", "—", "—"),
            inputs=[],
            outputs=[prompt_input, response_output, latency_box, cache_status_box, tokens_box],
        )

    return demo, mount_kwargs


# Create Gradio demo and mount onto the existing FastAPI application
demo, mount_kwargs = create_gradio_ui()
app = gr.mount_gradio_app(app, demo, path="/", **mount_kwargs)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
