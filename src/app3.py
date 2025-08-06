import os
from dotenv import load_dotenv
import chainlit as cl
from chainlit.input_widget import Select

from langchain_groq import ChatGroq
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.schema import StrOutputParser
from langchain.schema.runnable.config import RunnableConfig

load_dotenv()

SUPPORTED_MODELS = {
    "openai/gpt-oss-20b": "GPT-OSS 20B (by OpenAI)",
    "openai/gpt-oss-120b": "GPT-OSS 120B (by OpenAI)",
    "meta-llama/llama-4-maverick-17b-128e-instruct": "LLaMA 4 Maverick 17B (by Meta AI)",
    "meta-llama/llama-4-scout-17b-16e-instruct": "LLaMA 4 Scout 17B (by Meta AI)",
    "moonshotai/kimi-k2-instruct": "Kimi K2 Instruct (by Moonshot AI)",
    "qwen/qwen3-32b": "Qwen 3 32B (by Alibaba Group)"
}

@cl.on_chat_start
async def on_chat_start():
    settings = await cl.ChatSettings([
        Select(id="model", label="Model", values=list(SUPPORTED_MODELS.keys()), initial_index=0),
        Select(id="temp", label="Temperature", values=["0.0","0.3","0.7","1.0"], initial_index=2),
        Select(id="reasoning", label="Reasoning Level", values=["low","medium","high"], initial_index=1),
    ]).send()

    selected_model = settings["model"]
    temp = float(settings["temp"])
    reasoning = settings["reasoning"]

    cl.user_session.set("chat_settings", settings)
    cl.user_session.set("chat_messages", [])
    await create_runnable(os.getenv("GROQ_API_KEY"), selected_model, temp, reasoning)

async def create_runnable(api_key, selected_model, temperature, reasoning_level):
    # Build system prompt (as before)
    if "llama" in selected_model:
        base = SUPPORTED_MODELS[selected_model]
        system_prompt = f"You are a multimodal assistant powered by {base}, designed for structured reasoning. Use {reasoning_level} reasoning."
    elif "gpt-oss" in selected_model:
        base = SUPPORTED_MODELS[selected_model]
        system_prompt = f"You are a creative assistant powered by {base}, with open-source weights. Use {reasoning_level} reasoning."
    # ... other model cases ...
    else:
        system_prompt = f"You are an assistant running on {selected_model}. Use {reasoning_level} reasoning."

    # Prompt with MessagesPlaceholder
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("history", optional=True, n_messages=20),
        ("human", "{question}")
    ])

    model = ChatGroq(api_key=api_key, model=selected_model, temperature=temperature, streaming=True)
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)

@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")
    if not runnable:
        await cl.Message("⚠️ Session not initialized. Please refresh.").send()
        return

    history_view = cl.user_session.get("chat_messages", [])
    # Convert history to list of tuples
    history_list = [(m["type"], m["content"]) for m in history_view[-20:]]

    response_msg = cl.Message(content="")
    try:
        async for chunk in runnable.astream(
            {"history": history_list, "question": message.content},
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()])
        ):
            await response_msg.stream_token(chunk)
        await response_msg.send()

        # Save the new turn
        chat_history = cl.user_session.get("chat_messages", [])
        chat_history.append({"type": "human", "content": message.content})
        chat_history.append({"type": "ai", "content": response_msg.content})
        cl.user_session.set("chat_messages", chat_history)

    except Exception as e:
        await cl.Message(content=f"❌ Error: {str(e)}").send()

if __name__ == "__main__":
    cl.run()
