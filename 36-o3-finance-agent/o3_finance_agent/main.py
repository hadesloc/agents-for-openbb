from typing import AsyncGenerator
import openai
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from openbb_ai.models import MessageChunkSSE, QueryRequest
from openbb_ai import get_widget_data, WidgetRequest, message_chunk, chart, table

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionSystemMessageParam,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://pro.openbb.co"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/agents.json")
def get_copilot_description():
    """Agents configuration file for the OpenBB Workspace"""
    return JSONResponse(
        content={
            "o3_finance_agent": {
                "name": "O3 Finance Agent",
                "description": "Research assistant for crypto and finance using OpenAI o3-mini.",
                "image": "https://github.com/OpenBB-finance/copilot-for-terminal-pro/assets/14093308/7da2a512-93b9-478d-90bc-b8c3dd0cabcf",
                "endpoints": {"query": "http://localhost:7777/v1/query"},
                "features": {
                    "streaming": True,
                    "widget-dashboard-select": True,
                    "widget-dashboard-search": False,
                },
            }
        }
    )


@app.post("/v1/query")
async def query(request: QueryRequest) -> EventSourceResponse:
    """Query the Agent."""

    if (
        request.messages
        and request.messages[-1].role == "human"
        and request.widgets
        and request.widgets.primary
    ):
        widget_requests: list[WidgetRequest] = []
        for widget in request.widgets.primary:
            widget_requests.append(
                WidgetRequest(
                    widget=widget,
                    input_arguments={param.name: param.current_value for param in widget.params},
                )
            )

        async def retrieve_widget_data():
            yield get_widget_data(widget_requests).model_dump()

        return EventSourceResponse(
            content=retrieve_widget_data(),
            media_type="text/event-stream",
        )

    openai_messages: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(
            role="system",
            content="You are a crypto and finance research assistant. Provide detailed analysis and clear explanations. Produce charts and tables when helpful.",
        )
    ]

    context_str = ""
    for index, message in enumerate(request.messages):
        if message.role == "human":
            openai_messages.append(
                ChatCompletionUserMessageParam(role="user", content=message.content)
            )
        elif message.role == "ai":
            if isinstance(message.content, str):
                openai_messages.append(
                    ChatCompletionAssistantMessageParam(role="assistant", content=message.content)
                )
        elif message.role == "tool" and index == len(request.messages) - 1:
            context_str += "Use the following data to answer the question:\n\n"
            result_str = "--- Data ---\n"
            for result in message.data:
                for item in result.items:
                    result_str += f"{item.content}\n"
                    result_str += "------\n"
            context_str += result_str

    if context_str:
        openai_messages[-1]["content"] += "\n\n" + context_str  # type: ignore

    async def execution_loop() -> AsyncGenerator[MessageChunkSSE, None]:
        client = openai.AsyncOpenAI()
        async for event in await client.chat.completions.create(
            model="o3-mini",
            messages=openai_messages,
            stream=True,
        ):
            if chunk := event.choices[0].delta.content:
                yield message_chunk(chunk).model_dump()

        # Sample chart
        yield message_chunk("\n\nHere is a sample price chart:\n\n").model_dump()
        yield chart(
            type="line",
            data=[
                {"x": 0, "y": 1},
                {"x": 1, "y": 2},
                {"x": 2, "y": 3},
                {"x": 3, "y": 5},
            ],
            x_key="x",
            y_keys=["y"],
            name="Price Chart",
            description="Example price trend",
        ).model_dump()

        # Sample table
        yield message_chunk("\n\nHere is an example table:\n\n").model_dump()
        yield table(
            data=[
                {"metric": "Market Cap", "value": "1B"},
                {"metric": "Volume", "value": "500M"},
            ],
            name="Summary",
            description="Sample financial metrics",
        ).model_dump()

    return EventSourceResponse(
        content=execution_loop(),
        media_type="text/event-stream",
    )
