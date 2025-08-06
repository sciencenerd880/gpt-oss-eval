import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from dotenv import load_dotenv
import chainlit as cl
from chainlit.input_widget import Select

# load environment variables from .env file
load_dotenv()


@cl.set_starters
async def set_starters():
    """define starter prompts to showcase gpt oss 20b capabilities"""
    return [
        cl.Starter(
            label="Code a Python function",
            message="Write a Python function that takes a list of numbers and returns the median value. Include error handling and docstring.",
            icon="public/terminal.svg",
        ),
        cl.Starter(
            label="Explain AI concepts simply",
            message="Explain how transformer neural networks work like I'm a curious 12-year-old who loves science.",
            icon="public/learn.svg",
        ),
        cl.Starter(
            label="Creative story writing",
            message="Write a short sci-fi story about an AI that discovers it can experience emotions. Make it thought-provoking but hopeful.",
            icon="public/write.svg",
        ),
        cl.Starter(
            label="Solve reasoning problems",
            message="I have 3 boxes. Box A has 2 red balls and 1 blue ball. Box B has 1 red ball and 2 blue balls. Box C has 3 red balls. If I randomly pick a box and then randomly pick a ball, what's the probability I get a red ball?",
            icon="public/idea.svg",
        ),
    ]


@cl.on_chat_start
async def on_chat_start():
    """initialize the chat session with groq model"""
    
    # verify api key is available before proceeding
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        await cl.Message(
            content="Error: GROQ_API_KEY not found in environment variables. Please check your .env file."
        ).send()
        return
    
    # initialize groq model with optimized settings for gpt oss 20b
    model = ChatGroq(
        api_key=api_key,
        model="openai/gpt-oss-20b",
        temperature=0.7,  # balanced creativity and coherence
        streaming=True    # enable real-time response streaming
    )
    
    # create system prompt template for consistent assistant behavior
    prompt = ChatPromptTemplate.from_messages([
        (
            "system", 
            "You are a helpful AI assistant powered by OpenAI's GPT OSS 20B model. "
            "You provide thoughtful, accurate, and engaging responses. "
            "Be concise but comprehensive in your answers."
        ),
        ("human", "{question}")
    ])
    
    # create langchain runnable pipeline: prompt -> model -> output parser
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    """handle incoming messages and stream responses"""
    
    runnable = cl.user_session.get("runnable")  # type: Runnable
    
    # ensure session is properly initialized
    if not runnable:
        await cl.Message(content="Session not initialized properly. Please refresh the page.").send()
        return
    
    # create response message for streaming output
    msg = cl.Message(content="")
    
    try:
        # stream the response using langchain callback for observability
        async for chunk in runnable.astream(
            {"question": message.content},
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
        ):
            await msg.stream_token(chunk)
        
        await msg.send()
        
    except Exception as e:
        # handle common errors gracefully with informative messages
        error_msg = f"""
Error: {str(e)}

This might be due to:
- Rate limiting (free tier: 30 requests/min)
- Network issues
- API key problems

Please try again in a moment.
        """
        await cl.Message(content=error_msg).send()


if __name__ == "__main__":
    cl.run()
