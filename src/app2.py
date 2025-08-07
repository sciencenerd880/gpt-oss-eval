import os
from dotenv import load_dotenv
import chainlit as cl
from chainlit.input_widget import Select

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.schema import StrOutputParser
from langchain.schema.runnable.config import RunnableConfig

# Load environment variables
load_dotenv()

# Supported models with friendly names
SUPPORTED_MODELS = {
    "openai/gpt-oss-20b": "GPT-OSS 20B (by OpenAI)",
    "openai/gpt-oss-120b": "GPT-OSS 120B (by OpenAI)", 
    "meta-llama/llama-4-maverick-17b-128e-instruct": "LLaMA 4 Maverick 17B (by Meta AI)",
    "meta-llama/llama-4-scout-17b-16e-instruct": "LLaMA 4 Scout 17B (by Meta AI)",
    "moonshotai/kimi-k2-instruct": "Kimi K2 Instruct (by Moonshot AI)",
    "qwen/qwen3-32b": "Qwen 3 32B (by Alibaba Group)"
}

# Chat profiles for different trading styles
@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name="general_chat",
            markdown_description="**General AI Assistant** - Versatile helper for any topic: writing, coding, learning, problem-solving, and general conversations.",
            icon="public/general_chat.svg",
        ),
        cl.ChatProfile(
            name="day_trader",
            markdown_description="Optimized for **day trading** - quick decisions, technical analysis, and intraday opportunities.",
            icon="public/day_trader.svg",
        ),
        cl.ChatProfile(
            name="swing_trader", 
            markdown_description="Perfect for **swing trading** - multi-day positions, trend analysis, and momentum plays.",
            icon="public/swing_trader.svg",
        ),
        cl.ChatProfile(
            name="long_term_investor",
            markdown_description="Designed for **long-term investing** - fundamental analysis, portfolio building, and wealth creation.",
            icon="public/long_term_investor.svg",
        ),
        cl.ChatProfile(
            name="crypto_specialist",
            markdown_description="Specialized in **cryptocurrency** - DeFi, altcoins, NFTs, and blockchain opportunities.",
            icon="public/crypto_specialist.svg",
        ),
    ]


# Starter examples
@cl.set_starters
async def set_starters():
    """define starter prompts to showcase TradeMuse capabilities for traders and investors"""
    return [
        cl.Starter(
            label="💰 Market Analysis Today",
            message="Analyze today's market conditions and identify 3 high-potential trading opportunities with specific entry/exit strategies.",
            icon="public/idea.svg",
        ),
        cl.Starter(
            label="📊 Portfolio Review",
            message="I have $50,000 to invest across stocks, crypto, and bonds. Create a diversified portfolio strategy based on current market conditions and my moderate risk tolerance.",
            icon="public/learn.svg",
        ),
        cl.Starter(
            label="🚀 Hot Stock Picks",
            message="What are the top 5 undervalued stocks in tech and healthcare sectors right now? Include price targets and catalysts to watch.",
            icon="public/terminal.svg",
        ),
        cl.Starter(
            label="⚡ Crypto Signals",
            message="Analyze Bitcoin, Ethereum, and 3 promising altcoins. Give me trading signals with risk management strategies for the next 2 weeks.",
            icon="public/write.svg",
        ),
        cl.Starter(
            label="📈 Options Strategy",
            message="I'm bullish on NVDA but want to limit downside risk. Design an options strategy with specific strikes, expiration dates, and profit/loss scenarios.",
            icon="public/paperwork.svg",
        ),
        cl.Starter(
            label="🔍 Earnings Play",
            message="Which companies have earnings this week that could move 5%+? Give me pre-earnings positioning strategies and post-earnings follow-up plans.",
            icon="public/idea.svg",
        ),
    ]

# Initialize chat
@cl.on_chat_start
async def on_chat_start():
    # Get the current chat profile
    chat_profile = cl.user_session.get("chat_profile")
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        await cl.Message(content="❌ GROQ_API_KEY not found. Please set it in your .env file.").send()
        return

    # UI for chat settings
    settings = await cl.ChatSettings([
        Select(id="model", label="Model", values=list(SUPPORTED_MODELS.keys()), initial_index=0),
        Select(id="temp", label="Temperature", values=["0.0", "0.3", "0.7", "1.0"], initial_index=2),
        Select(id="reasoning", label="Reasoning Level", values=["low", "medium", "high"], initial_index=1),
    ]).send()

    # Get the actual settings values
    selected_model = settings["model"]
    temperature = float(settings["temp"])
    reasoning_level = settings["reasoning"]

    # Store settings in session for later use
    cl.user_session.set("chat_settings", settings)
    cl.user_session.set("chat_messages", [])  # Initialize chat history

    # Create the model and runnable chain
    await create_runnable(api_key, selected_model, temperature, reasoning_level, chat_profile)
    
    # Don't send welcome message immediately - let users see starters first
    # Store welcome message to send after first interaction
    profile_emoji = {"general_chat": "💬", "day_trader": "⚡", "swing_trader": "📈", "long_term_investor": "💎", "crypto_specialist": "₿"}.get(chat_profile, "🤖")
    welcome_msg = f"{profile_emoji} **TradeMuse {chat_profile.replace('_', ' ').title() if chat_profile else 'Assistant'}** initialized with **{SUPPORTED_MODELS[selected_model]}** (temp: {temperature}, reasoning: {reasoning_level})"
    cl.user_session.set("welcome_message", welcome_msg)
    cl.user_session.set("first_message", True)


async def create_runnable(api_key, selected_model, temperature, reasoning_level, chat_profile=None):
    """Helper function to create the runnable chain with given settings"""
    
    # Base system prompt based on model type
    if "llama" in selected_model:
        base_prompt = (
            f"You are a powerful multimodal assistant powered by {SUPPORTED_MODELS[selected_model]}. "
            "Developed by Meta AI, you are optimized for reasoning, structured responses, and visual comprehension. "
            f"Use {reasoning_level} reasoning. Be structured, concise, and capable across multiple tasks."
        )
    elif "gpt-oss" in selected_model:
        base_prompt = (
            f"You are a helpful AI assistant powered by {SUPPORTED_MODELS[selected_model]}. "
            "Created by OpenAI, you operate under open weights and excel in flexible, high-performance reasoning. "
            f"Your reasoning should be {reasoning_level}. Respond clearly, creatively, and responsibly."
        )
    elif "kimi" in selected_model:
        base_prompt = (
            f"You are a thoughtful, autonomous assistant running on {SUPPORTED_MODELS[selected_model]}. "
            "Developed by Moonshot AI, you specialize in long-context understanding, advanced tool use, and deep reasoning. "
            f"Use {reasoning_level} reasoning and provide insightful, deliberate answers."
        )
    elif "qwen" in selected_model:
        base_prompt = (
            f"You are a multilingual and versatile assistant powered by {SUPPORTED_MODELS[selected_model]}. "
            "Developed by Alibaba Group, you excel in deep reasoning, fast mode-switching, and global communication. "
            f"Use {reasoning_level} level of reasoning. Be accurate, helpful, and language-aware."
        )
    else:
        base_prompt = (
            f"You are a knowledgeable assistant running on {selected_model}. "
            f"Use {reasoning_level} level of reasoning. Be accurate and concise."
        )

    # Add trading-specific context based on chat profile
    if chat_profile == "general_chat":
        trading_context = (
            "\n\nYou are a GENERAL AI ASSISTANT: You can help with any topic including writing, coding, "
            "learning, problem-solving, creative tasks, research, explanations, and general conversations. "
            "You are NOT focused on trading or finance unless specifically asked. Be helpful, informative, "
            "and adaptable to whatever the user needs assistance with."
        )
    elif chat_profile == "day_trader":
        trading_context = (
            "\n\nYou are specialized in DAY TRADING: Focus on intraday opportunities, scalping, "
            "technical analysis, real-time market movements, and quick decision-making. "
            "Prioritize speed, efficiency, and risk management for short-term positions."
        )
    elif chat_profile == "swing_trader":
        trading_context = (
            "\n\nYou are specialized in SWING TRADING: Focus on multi-day to multi-week positions, "
            "trend analysis, momentum plays, and medium-term market movements. "
            "Balance technical and fundamental analysis for optimal entry/exit timing."
        )
    elif chat_profile == "long_term_investor":
        trading_context = (
            "\n\nYou are specialized in LONG-TERM INVESTING: Focus on fundamental analysis, "
            "portfolio diversification, wealth building, dividend strategies, and multi-year positions. "
            "Emphasize research, patience, and compound growth strategies."
        )
    elif chat_profile == "crypto_specialist":
        trading_context = (
            "\n\nYou are specialized in CRYPTOCURRENCY: Focus on blockchain technology, DeFi protocols, "
            "altcoin analysis, NFT markets, crypto trading strategies, and emerging blockchain opportunities. "
            "Stay current with crypto news, regulatory changes, and market sentiment."
        )
    else:
        trading_context = (
            "\n\nYou are a general trading and investment assistant. Provide balanced advice "
            "across all trading styles and investment strategies."
        )

    system_prompt = base_prompt + trading_context

    # Build prompt template with MessagesPlaceholder for chat history
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("history", optional=True, n_messages=20),
        ("human", "{question}")
    ])

    # Build model with selected settings
    model = ChatGroq(
        api_key=api_key,
        model=selected_model,
        temperature=temperature,
        streaming=True
    )

    # Create runnable chain
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)
    cl.user_session.set("current_model", selected_model)

# Handle messages
@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")

    if not runnable:
        await cl.Message(content="⚠️ Session not initialized. Please refresh.").send()
        return

    # Show welcome message on first interaction
    if cl.user_session.get("first_message"):
        welcome_msg = cl.user_session.get("welcome_message")
        if welcome_msg:
            await cl.Message(content=welcome_msg).send()
        cl.user_session.set("first_message", False)

    # Get chat history and convert to format expected by MessagesPlaceholder
    history_view = cl.user_session.get("chat_messages", [])
    history_list = [(m["type"], m["content"]) for m in history_view[-20:]]

    response_msg = cl.Message(content="")

    try:
        async for chunk in runnable.astream(
            {"history": history_list, "question": message.content},
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()])
        ):
            await response_msg.stream_token(chunk)

        await response_msg.send()

        # Save the new conversation turn to chat history
        chat_history = cl.user_session.get("chat_messages", [])
        chat_history.append({"type": "human", "content": message.content})
        chat_history.append({"type": "ai", "content": response_msg.content})
        cl.user_session.set("chat_messages", chat_history)

    except Exception as e:
        await cl.Message(
            content=f"❌ Error: {str(e)}\nPlease check your API key, model selection, or network connection."
        ).send()

# Handle settings update dynamically
@cl.on_settings_update
async def on_settings_update(settings):
    """Handle when user changes settings during conversation"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        await cl.Message(content="❌ GROQ_API_KEY not found.").send()
        return
    
    # Get new settings
    selected_model = settings["model"]
    temperature = float(settings["temp"])
    reasoning_level = settings["reasoning"]
    
    # Update session settings
    cl.user_session.set("chat_settings", settings)
    
    # Get current chat profile
    chat_profile = cl.user_session.get("chat_profile")
    
    # Create new runnable with updated settings
    await create_runnable(api_key, selected_model, temperature, reasoning_level, chat_profile)
    
    # Update the stored welcome message with new model info
    profile_emoji = {"general_chat": "💬", "day_trader": "⚡", "swing_trader": "📈", "long_term_investor": "💎", "crypto_specialist": "₿"}.get(chat_profile, "🤖")
    updated_welcome_msg = f"{profile_emoji} **TradeMuse {chat_profile.replace('_', ' ').title() if chat_profile else 'Assistant'}** initialized with **{SUPPORTED_MODELS[selected_model]}** (temp: {temperature}, reasoning: {reasoning_level})"
    cl.user_session.set("welcome_message", updated_welcome_msg)
    
    # Notify user of the change
    await cl.Message(
        content=f"🔄 Settings updated! Now using **{SUPPORTED_MODELS[selected_model]}** (temp: {temperature}, reasoning: {reasoning_level})"
    ).send()

# Handle chat resume
@cl.on_chat_resume
async def on_chat_resume(thread):
    """Handle when user resumes a previous chat thread"""
    # Get the current chat profile
    chat_profile = cl.user_session.get("chat_profile")
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        await cl.Message(content="❌ GROQ_API_KEY not found.").send()
        return
    
    # Use default settings or restore from thread metadata if available
    selected_model = "openai/gpt-oss-20b"  # Default
    temperature = 0.7
    reasoning_level = "medium"
    
    # Try to restore settings from thread metadata if available
    if thread.get("metadata"):
        metadata = thread["metadata"]
        selected_model = metadata.get("model", selected_model)
        temperature = float(metadata.get("temperature", temperature))
        reasoning_level = metadata.get("reasoning_level", reasoning_level)
    
    # Initialize chat history for resumed session
    cl.user_session.set("chat_messages", [])
    
    # Recreate the runnable chain
    await create_runnable(api_key, selected_model, temperature, reasoning_level, chat_profile)
    
    # Welcome back message
    profile_emoji = {"general_chat": "💬", "day_trader": "⚡", "swing_trader": "📈", "long_term_investor": "💎", "crypto_specialist": "₿"}.get(chat_profile, "🤖")
    await cl.Message(
        content=f"{profile_emoji} **Welcome back!** Resumed your {chat_profile.replace('_', ' ').title() if chat_profile else 'TradeMuse'} session with **{SUPPORTED_MODELS[selected_model]}**"
    ).send()

# Run the app
if __name__ == "__main__":
    cl.run()
