"""FastAPI server for climate LLM explanations."""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from .rate_limit import create_chat_limiter
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI

from .climate_data import FIELD_INFO, ClimateDataStore
from .prompts import build_system_prompt
from .tools import TOOLS

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

# Build system prompt with available fields
_field_descs = [
    {"name": f, "desc": FIELD_INFO[f]["desc"], "unit": FIELD_INFO[f]["unit"]}
    for f in store.available_fields
]
SYSTEM = build_system_prompt(_field_descs)

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", "sk-dummy"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")

MAX_TOOL_ROUNDS = 15
MAX_MESSAGES = 50
MAX_MESSAGE_LENGTH = 2000


@app.post("/api/chat", dependencies=[Depends(chat_limiter)])
async def chat(request: Request) -> StreamingResponse:
    body = await request.json()
    lat: float = body["lat"]
    lon: float = body["lon"]
    month: int = body["month"]
    user_messages: list[dict[str, str]] = body["messages"]
    imperial: bool = body.get("imperial", False)

    # Input validation
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        raise HTTPException(400, "Invalid coordinates")
    if not (0 <= month <= 11):
        raise HTTPException(400, "Invalid month")
    if not isinstance(user_messages, list) or not user_messages:
        raise HTTPException(400, "Invalid messages")
    if len(user_messages) > MAX_MESSAGES:
        raise HTTPException(400, "Too many messages")
    for msg in user_messages:
        if len(msg.get("content", "")) > MAX_MESSAGE_LENGTH:
            raise HTTPException(400, "Message too long")

    # Build conversation with location context
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM}]

    prev_lat: float | None = body.get("prevLat")
    prev_lon: float | None = body.get("prevLon")
    prev_month: int | None = body.get("prevMonth")

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    context_parts: list[str] = []
    loc_changed = (
        prev_lat is None
        or round(lat, 1) != round(prev_lat, 1)
        or round(lon, 1) != round(prev_lon or 0, 1)
    )
    month_changed = prev_month is None or month != prev_month

    if loc_changed:
        ns = "N" if lat >= 0 else "S"
        ew = "E" if lon >= 0 else "W"
        context_parts.append(f"[Location: {lat:.1f}°{ns}, {abs(lon):.1f}°{ew}]")
    if month_changed:
        context_parts.append(f"[Month: {month_names[month % 12]}]")

    messages.extend(user_messages)

    if context_parts:
        prefix = " ".join(context_parts) + "\n\n"
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "user":
                messages[i] = {**messages[i], "content": prefix + messages[i]["content"]}
                break

    # Stream everything: tool-call rounds + final text, all via streaming API
    async def generate():
        nonlocal messages
        for _ in range(MAX_TOOL_ROUNDS):
            # Accumulate tool calls and text from a single streamed response
            tool_calls_acc: dict[int, dict[str, str]] = {}  # index -> {id, name, arguments}
            text_chunks: list[str] = []

            stream = await client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta

                # Stream text tokens immediately
                if delta.content:
                    text_chunks.append(delta.content)
                    yield f"data: {json.dumps({'text': delta.content})}\n\n"

                # Accumulate tool call deltas
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

            # If no tool calls, we're done — text was already streamed
            if not tool_calls_acc:
                break

            # Resolve tool calls and notify frontend
            assistant_tool_calls = []
            for idx in sorted(tool_calls_acc):
                tc = tool_calls_acc[idx]
                args = json.loads(tc["arguments"])
                # Send tool progress event
                field_name = args.get("field", "?")
                field_desc = FIELD_INFO.get(field_name, {}).get("desc", field_name)
                yield f"data: {json.dumps({'tool': field_desc})}\n\n"
                # Resolve and build message history
                result = store.sample(
                    field=args["field"],
                    lat=args["lat"],
                    lon=args["lon"],
                    month=args["month"],
                    imperial=imperial,
                )
                assistant_tool_calls.append({
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(result),
                })
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
