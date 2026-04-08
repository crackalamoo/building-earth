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
from openai.types.responses import (
    ResponseInputParam,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseReasoningSummaryPartAddedEvent,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseTextDeltaEvent,
    ToolParam,
)

from .calculate import ExpressionError, resolve_fields
from .climate_data import FIELD_INFO, OBS_FIELD_INFO, ClimateDataStore, ObsDataStore
from .prompts import SYSTEM_PROMPT
from .tools import TOOLS, tool_placeholder_label
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


# Load climate data once at startup. One simulation snapshot per onboarding
# stage so the LLM's tool lookups return numbers consistent with what the
# user is actually looking at on the globe. Stage 5 is the full final model
# (data/main.npz); stages 1-4 are the progressive onboarding snapshots.
def _load_stage_store(stage: int) -> ClimateDataStore:
    if stage == 5:
        return ClimateDataStore()
    return ClimateDataStore(
        npz_path=f"data/stage{stage}.npz",
        bin_path=f"frontend/public/stage{stage}.bin.gz",
        manifest_path=f"frontend/public/stage{stage}.manifest.json",
    )

stage_stores: dict[int, ClimateDataStore] = {
    s: _load_stage_store(s) for s in (1, 2, 3, 4, 5)
}
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

    # Pick the simulation snapshot that matches the user's current stage so
    # the LLM's tool lookups return numbers consistent with what the user is
    # actually looking at on the globe. Stage 5 is the full final model.
    request_store = stage_stores[stage if stage in stage_stores else 5]

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

    # Find the index of the latest user-authored message so we can stamp the
    # location prefix on it as we build the typed input list.
    last_user_idx = next(
        (i for i in range(len(user_messages) - 1, -1, -1)
         if user_messages[i].get("role") == "user"),
        None,
    )

    input_items: list[dict[str, Any]] = [
        {"type": "message", "role": "system", "content": system_content}
    ]
    for i, msg in enumerate(user_messages):
        content = msg.get("content", "")
        if i == last_user_idx:
            content = prefix + content
        input_items.append(
            {"type": "message", "role": msg["role"], "content": content}
        )

    # Stream everything: reasoning summaries, tool-call rounds, and final
    # text — all via the Responses streaming API.
    async def generate():
        nonlocal input_items
        # Flush an immediate SSE comment so the browser opens the stream and any
        # intermediate proxies (Railway, nginx) commit headers right away.
        yield ": open\n\n"
        for _ in range(MAX_TOOL_ROUNDS):
            # Tool calls collected from this round, in arrival order. Each
            # entry has name, arguments, and call_id from the completed
            # function_call output item.
            pending_tool_calls: list[dict[str, str]] = []

            try:
                extra_kwargs: dict[str, Any] = {}
                if SUPPORTS_REASONING_EFFORT:
                    # "auto" lets the model decide summary length. Non-reasoning
                    # models reject the whole reasoning kwarg, so we only attach
                    # it for gpt-5/o-series.
                    extra_kwargs["reasoning"] = {
                        "effort": REASONING_EFFORT,
                        "summary": "auto",
                    }
                stream = await client.responses.create(
                    model=MODEL,
                    input=cast(ResponseInputParam, input_items),
                    tools=cast(list[ToolParam], TOOLS),
                    tool_choice="auto",
                    stream=True,
                    **extra_kwargs,
                )
                async for event in stream:
                    if isinstance(event, ResponseTextDeltaEvent):
                        yield f"data: {json.dumps({'text': event.delta})}\n\n"

                    elif isinstance(event, ResponseReasoningSummaryTextDeltaEvent):
                        # Streamed reasoning summary — gives the user something
                        # concrete to read while the model is thinking.
                        yield f"data: {json.dumps({'thinking': event.delta})}\n\n"

                    elif isinstance(event, ResponseReasoningSummaryPartAddedEvent):
                        # Separate consecutive summary parts with a blank line
                        # so they read as paragraphs in the UI.
                        if event.summary_index > 0:
                            yield f"data: {json.dumps({'thinking': '\\n\\n'})}\n\n"

                    elif isinstance(event, ResponseOutputItemAddedEvent):
                        # Flush a placeholder the instant we know which tool the
                        # model is calling, before its arguments finish streaming.
                        if event.item.type == "function_call":
                            placeholder = tool_placeholder_label(event.item.name)
                            yield f"data: {json.dumps({'tool': placeholder, 'pending': True})}\n\n"

                    elif isinstance(event, ResponseOutputItemDoneEvent):
                        # function_call items carry name, arguments, and call_id
                        # together — collect them here for resolution after the
                        # stream ends.
                        if event.item.type == "function_call":
                            pending_tool_calls.append({
                                "name": event.item.name,
                                "arguments": event.item.arguments,
                                "call_id": event.item.call_id,
                            })
            except APIError as e:
                yield f"data: {json.dumps({'error': f'LLM service error: {e.message}'})}\n\n"
                break
            except Exception:
                yield f"data: {json.dumps({'error': 'Something went wrong. Please try again.'})}\n\n"
                break

            if not pending_tool_calls:
                break

            for tc in pending_tool_calls:
                try:
                    args = json.loads(tc["arguments"])
                    if tc["name"] == "calculate":
                        calc_result = resolve_fields(
                            expression=args["expression"],
                            sample_fn=request_store.sample_raw,
                            lat=args["lat"],
                            lon=args["lon"],
                            month=args["month"],
                        )
                        if "unit" in args:
                            calc_result["unit"] = args["unit"]
                        result = calc_result
                        yield f"data: {json.dumps({'tool': 'Calculate'})}\n\n"
                    else:
                        fields_list = args.get(
                            "fields", [args["field"]] if "field" in args else []
                        )
                        data_store = (
                            obs_store
                            if tc["name"] == "sample_observations"
                            else request_store
                        )
                        result = data_store.sample_many(
                            fields=fields_list,
                            lat=args["lat"],
                            lon=args["lon"],
                            month=args["month"],
                            imperial=imperial,
                        )
                        field_meta = (
                            OBS_FIELD_INFO
                            if tc["name"] == "sample_observations"
                            else FIELD_INFO
                        )
                        labels = [
                            field_meta.get(f, {}).get("label", f)
                            for f in fields_list
                        ]
                        for label in labels:
                            yield f"data: {json.dumps({'tool': label})}\n\n"
                except ExpressionError as exc:
                    result = {"error": str(exc)}
                    yield f"data: {json.dumps({'tool': 'Calculate'})}\n\n"
                except Exception:
                    result = {"error": "Failed to process tool call"}
                    yield f"data: {json.dumps({'tool': '?'})}\n\n"

                # function_call must precede function_call_output in the input
                # so the model sees its own call before the output.
                input_items.append(
                    {
                        "type": "function_call",
                        "call_id": tc["call_id"],
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                    }
                )
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": tc["call_id"],
                        "output": json.dumps(result),
                    }
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
