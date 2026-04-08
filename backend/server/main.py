"""FastAPI server for climate LLM explanations."""

import json
import os
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .rate_limit import create_chat_limiter
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI, APIError
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

from .calculate import ExpressionError, resolve_fields
from .climate_data import FIELD_INFO, OBS_FIELD_INFO, ClimateDataStore, ObsDataStore
from .prompts import SYSTEM_PROMPT
from .tools import TOOLS
from onboarding_stages import get_stage_chat_context

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGIN", "http://localhost:5173").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

chat_limiter = create_chat_limiter()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


# Load climate data once at startup
store = ClimateDataStore()
obs_store = ObsDataStore()

SYSTEM = SYSTEM_PROMPT

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", "sk-dummy"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")

# Reasoning models (gpt-5*, o-series) accept a reasoning_effort knob that
# trades latency for deeper deliberation. For a tool-calling chat app the
# default ("medium") wastes seconds on every turn. We use "low" — "minimal"
# is contraindicated by OpenAI for multi-step or tool-heavy workflows and
# noticeably degrades instruction following, while the latency gap to "low"
# is under a second on gpt-5-mini.
# Non-reasoning models (gpt-4.1*, gpt-4o*) reject this parameter, so we only
# pass it when the model name indicates support.
_REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")
SUPPORTS_REASONING_EFFORT = MODEL.startswith(_REASONING_MODEL_PREFIXES)
REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "low")

MAX_TOOL_ROUNDS = 15
MAX_MESSAGES = 50
MAX_MESSAGE_LENGTH = 2000


def _tool_placeholder_label(tool_name: str) -> str:
    """Friendly placeholder shown the moment a tool call's name appears in
    the OpenAI stream, before its arguments have finished streaming."""
    if tool_name == "sample_climate":
        return "Looking up climate…"
    if tool_name == "sample_observations":
        return "Looking up observations…"
    if tool_name == "calculate":
        return "Calculating…"
    return "Looking up…"


@app.get("/api/obs")
async def obs_data(lat: float, lon: float) -> dict:
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        raise HTTPException(400, "Invalid coordinates")
    temps, precips = [], []
    for m in range(12):
        t_land = obs_store.sample("land_temperature", lat, lon, m)
        t_sst = obs_store.sample("sst", lat, lon, m)
        t = t_land if t_land["value"] is not None else t_sst
        p = obs_store.sample("precipitation", lat, lon, m)
        temps.append(t["value"])
        precips.append(p["value"])
    return {"temps": temps, "precips": precips}



@app.post("/api/chat", dependencies=[Depends(chat_limiter)])
async def chat(request: Request) -> StreamingResponse:
    body = await request.json()
    lat: float = body["lat"]
    lon: float = body["lon"]
    user_messages: list[dict[str, str]] = body["messages"]
    imperial: bool = body.get("imperial", False)
    stage: int | None = body.get("stage")

    # Input validation
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        raise HTTPException(400, "Invalid coordinates")
    if not isinstance(user_messages, list) or not user_messages:
        raise HTTPException(400, "Invalid messages")
    if len(user_messages) > MAX_MESSAGES:
        raise HTTPException(400, "Too many messages")
    for msg in user_messages:
        if msg.get("role") != "user":
            continue
        if len(msg.get("content", "")) > MAX_MESSAGE_LENGTH:
            raise HTTPException(400, "Message too long")

    # Build conversation with location context
    system_content = SYSTEM
    if stage is not None and stage < 5:
        system_content += get_stage_chat_context(stage)
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_content}]

    prev_lat: float | None = body.get("prevLat")
    prev_lon: float | None = body.get("prevLon")

    # Always stamp the current location on the latest user message so the model
    # never has to scroll back through prior turns to remember where the user is.
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    location_str = f"{lat:.1f}°{ns}, {abs(lon):.1f}°{ew}"

    loc_changed = (
        prev_lat is not None
        and (round(lat, 1) != round(prev_lat, 1)
             or round(lon, 1) != round(prev_lon or 0, 1))
    )

    location_label = "Now at" if loc_changed else "Location"
    prefix = f"[{location_label}: {location_str}]\n\n"

    messages.extend(user_messages)
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "user":
            messages[i] = {**messages[i], "content": prefix + messages[i]["content"]}
            break

    # Stream everything: tool-call rounds + final text, all via streaming API
    async def generate():
        nonlocal messages
        # Flush an immediate SSE comment so the browser opens the stream and any
        # intermediate proxies (Railway, nginx) commit headers right away.
        yield ": open\n\n"
        for _ in range(MAX_TOOL_ROUNDS):
            # Accumulate tool calls and text from a single streamed response
            tool_calls_acc: dict[int, dict[str, str]] = {}  # index -> {id, name, arguments}
            announced_tool_idx: set[int] = set()  # which indexes already had their name flushed

            try:
                extra_kwargs: dict[str, Any] = {}
                if SUPPORTS_REASONING_EFFORT:
                    extra_kwargs["reasoning_effort"] = REASONING_EFFORT
                # The OpenAI SDK expects discriminated TypedDicts for messages
                # and tools. We build those dicts dynamically (with assistant
                # tool_calls, tool results, etc.), so a structural cast at the
                # boundary is the right call — typing each literal would force
                # an explosion of TypedDict imports without buying real safety.
                stream = await client.chat.completions.create(
                    model=MODEL,
                    messages=cast(list[ChatCompletionMessageParam], messages),
                    tools=cast(list[ChatCompletionToolParam], TOOLS),
                    tool_choice="auto",
                    stream=True,
                    **extra_kwargs,
                )
                async for chunk in stream:
                    delta = chunk.choices[0].delta

                    # Stream text tokens immediately
                    if delta.content:
                        yield f"data: {json.dumps({'text': delta.content})}\n\n"

                    # Accumulate tool call deltas, and flush a "thinking" event
                    # the first time each tool call's name appears so the user
                    # gets feedback in ~1s instead of waiting for the whole round.
                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index
                            if idx not in tool_calls_acc:
                                tool_calls_acc[idx] = {"id": "", "name": "", "arguments": ""}
                            if tc_delta.id:
                                tool_calls_acc[idx]["id"] = tc_delta.id
                            if tc_delta.function:
                                if tc_delta.function.name:
                                    tool_calls_acc[idx]["name"] = tc_delta.function.name
                                if tc_delta.function.arguments:
                                    tool_calls_acc[idx]["arguments"] += tc_delta.function.arguments
                            # Flush a generic placeholder as soon as we know
                            # which tool is being called. The per-field labels
                            # come later, after the arguments fully stream.
                            if idx not in announced_tool_idx and tool_calls_acc[idx]["name"]:
                                announced_tool_idx.add(idx)
                                placeholder = _tool_placeholder_label(tool_calls_acc[idx]["name"])
                                yield f"data: {json.dumps({'tool': placeholder, 'pending': True})}\n\n"
            except APIError as e:
                yield f"data: {json.dumps({'error': f'LLM service error: {e.message}'})}\n\n"
                break
            except Exception:
                yield f"data: {json.dumps({'error': 'Something went wrong. Please try again.'})}\n\n"
                break

            # If no tool calls, we're done — text was already streamed
            if not tool_calls_acc:
                break

            # Resolve tool calls and notify frontend
            assistant_tool_calls = []
            for idx in sorted(tool_calls_acc):
                tc = tool_calls_acc[idx]
                try:
                    args = json.loads(tc["arguments"])
                    if tc["name"] == "calculate":
                        calc_result = resolve_fields(
                            expression=args["expression"],
                            sample_fn=store.sample_raw,
                            lat=args["lat"],
                            lon=args["lon"],
                            month=args["month"],
                        )
                        if "unit" in args:
                            calc_result["unit"] = args["unit"]
                        result = calc_result
                        yield f"data: {json.dumps({'tool': 'Calculate'})}\n\n"
                    else:
                        fields_list = args.get("fields", [args["field"]] if "field" in args else [])
                        data_store = obs_store if tc["name"] == "sample_observations" else store
                        result = data_store.sample_many(
                            fields=fields_list,
                            lat=args["lat"],
                            lon=args["lon"],
                            month=args["month"],
                            imperial=imperial,
                        )
                        # Send tool progress event
                        field_meta = (
                            OBS_FIELD_INFO if tc["name"] == "sample_observations" else FIELD_INFO
                        )
                        labels = [
                            field_meta.get(f, {}).get("label", f)
                            for f in args.get("fields", [args.get("field", "?")])
                        ]
                        for label in labels:
                            yield f"data: {json.dumps({'tool': label})}\n\n"
                except ExpressionError as exc:
                    result = {"error": str(exc)}
                    yield f"data: {json.dumps({'tool': 'Calculate'})}\n\n"
                except Exception:
                    result = {"error": "Failed to process tool call"}
                    yield f"data: {json.dumps({'tool': '?'})}\n\n"
                assistant_tool_calls.append(
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["arguments"]},
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps(result),
                    }
                )
            # Insert assistant message with tool calls before tool results
            messages.insert(
                len(messages) - len(assistant_tool_calls),
                {"role": "assistant", "tool_calls": assistant_tool_calls},
            )

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Serve built frontend (production) — must be after API routes
_frontend_dist = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
if _frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=_frontend_dist, html=True), name="static")
