"""Shared VLM helper — PydanticAI + Gemini 2.5 Flash / Flash-Lite.

Lanes B (labels) and E (relations) both call a Gemini model with a structured
output schema. Centralising the call here keeps both lanes thin and makes
swapping models a one-line change.

Auth: ``GEMINI_API_KEY`` env var (PydanticAI's google-gla provider picks it
up automatically). To route through the Pydantic AI Gateway, set
``PYDANTIC_AI_GATEWAY_API_KEY`` and ``PYDANTIC_AI_GATEWAY_BASE_URL`` and pass
``via_gateway=True``.

Default model: ``gemini-2.5-flash``. Flash-Lite is available for cheaper /
faster runs via ``SPATIALITY_VLM_MODEL=gemini-2.5-flash-lite`` or by passing
``model="gemini-2.5-flash-lite"`` to the call.
"""

from __future__ import annotations

import io
import logging
import os
from typing import TypeVar

import numpy as np
from PIL import Image
from pydantic import BaseModel

logger = logging.getLogger(__name__)


T = TypeVar("T", bound=BaseModel)


_DEFAULT_MODEL = "gemini-2.5-flash"


def _resolve_model(model: str | None) -> str:
    """Resolve a Gemini model id, preferring an explicit arg, then env, then default.

    PydanticAI uses ``provider:model`` ids; we add the ``google-gla`` prefix
    so the Google Generative Language API is selected.
    """
    name = model or os.environ.get("SPATIALITY_VLM_MODEL", _DEFAULT_MODEL)
    if ":" in name:
        return name
    return f"google-gla:{name}"


def _png_bytes(image: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    return buf.getvalue()


def call_vlm(
    prompt: str,
    images: list[np.ndarray],
    output_type: type[T],
    model: str | None = None,
    system_prompt: str | None = None,
) -> T:
    """Run a single VLM call with `images` + `prompt`, parsed into `output_type`.

    Returns the parsed Pydantic model. Raises on auth / network errors; callers
    that want to degrade gracefully should wrap in try/except.
    """
    from pydantic_ai import Agent, BinaryContent  # noqa: PLC0415

    agent_kwargs: dict = {"output_type": output_type}
    if system_prompt:
        agent_kwargs["system_prompt"] = system_prompt

    agent = Agent(_resolve_model(model), **agent_kwargs)

    parts: list = [BinaryContent(data=_png_bytes(im), media_type="image/png") for im in images]
    parts.append(prompt)

    result = agent.run_sync(parts)
    return result.output


async def call_vlm_async(
    prompt: str,
    images: list[np.ndarray],
    output_type: type[T],
    model: str | None = None,
    system_prompt: str | None = None,
) -> T:
    """Async variant of :func:`call_vlm` for concurrent fan-out via asyncio.gather.

    Used by the scene-scout pass which fans out N parallel Gemini Flash calls,
    one per temporal slice of the video. Building a fresh Agent per call is
    fine — pydantic-ai Agents are cheap construction objects.
    """
    from pydantic_ai import Agent, BinaryContent  # noqa: PLC0415

    agent_kwargs: dict = {"output_type": output_type}
    if system_prompt:
        agent_kwargs["system_prompt"] = system_prompt

    agent = Agent(_resolve_model(model), **agent_kwargs)

    parts: list = [BinaryContent(data=_png_bytes(im), media_type="image/png") for im in images]
    parts.append(prompt)

    result = await agent.run(parts)
    return result.output
