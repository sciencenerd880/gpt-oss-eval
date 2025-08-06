import chainlit as cl

@cl.on_message
async def on_message(message: cl.Message):
    """basic message handler that echoes user input"""
    await cl.Message(content=f"You said: {message.content}").send()

if __name__ == "__main__":
    cl.run()
