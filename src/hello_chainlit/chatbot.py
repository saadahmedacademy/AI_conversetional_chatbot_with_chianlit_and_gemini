import chainlit as cl
import os
from dotenv import load_dotenv, find_dotenv
from agents import Agent, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, Runner
from openai.types.responses import ResponseTextDeltaEvent

# Load environment variables
load_dotenv(find_dotenv())
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Step 1: Define the provider
externel_client_provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Step 2: Define the model
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=externel_client_provider
)

# Step 3: Define the run config
run_config = RunConfig(
    model=model,
    model_provider=externel_client_provider,
    tracing_disabled=True
)

# Step 4: Define the agent
agent1 = Agent(
    name="Saad Ahmed Academy's Agent",
    instructions="You are a helpful assistant that can answer questions and help with tasks.",
)

# Step 5: Initialize chat history
@cl.on_chat_start
async def handle_chat_history():
    cl.user_session.set("history", [])
    await cl.Message(content="Hello! I'm your AI assistant. How can I help you today?").send()

# Step 6: Handle user messages
@cl.on_message
async def main(message: cl.Message):
    try:
        # Load chat history
        history = cl.user_session.get("history") or []

        # Store user message
        history.append({
            "role": "user",
            "content": message.content
        })

        # Create and send an empty streaming message
        streaming_messages = cl.Message(content="")
        await streaming_messages.send()

        # Run agent with streaming enabled
        result = Runner.run_streamed(
            agent1,
            input=history,
            run_config=run_config,
        )

        # Stream the response
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                token = event.data.delta
                await streaming_messages.stream_token(token)

        # Store assistant response
        history.append({
            "role": "assistant",
            "content": streaming_messages.content
        })

        # Save history back to session
        cl.user_session.set("history", history)


    except Exception as e:
        await cl.Message(content=f"⚠️ Error: {str(e)}").send()
