import chainlit as cl
from chainlit.input_widget import Select, Switch, Slider

@cl.set_chat_profiles
async def chat_profile():
    """define different chat personalities using existing svg icons"""
    return [
        cl.ChatProfile(
            name="The Coder",
            markdown_description="**Programming Assistant**\n\nSpecialized in coding, debugging, and technical solutions.",
            icon="public/terminal.svg"  # pink terminal svg for coding tasks
        ),
        cl.ChatProfile(
            name="The Teacher", 
            markdown_description="**Educational Guide**\n\nExcellent at explaining complex topics in simple terms.",
            icon="public/learn.svg"  # pink learn svg for educational content
        ),
        cl.ChatProfile(
            name="The Writer",
            markdown_description="**Creative Assistant**\n\nHelps with writing, storytelling, and creative content.",
            icon="public/write.svg"  # pink write svg for creative tasks
        ),
        cl.ChatProfile(
            name="The Thinker",
            markdown_description="**Problem Solver**\n\nFocused on reasoning, analysis, and strategic thinking.",
            icon="public/idea.svg"  # pink idea svg for analytical tasks
        )
    ]

@cl.set_starters
async def set_starters():
    """define starter prompts that adapt based on selected chat profile"""
    chat_profile = cl.user_session.get("chat_profile", "The Coder")
    
    # return profile-specific starters using existing pink svg icons
    if chat_profile == "The Coder":
        return [
            cl.Starter(
                label="Debug Python code",
                message="Help me debug this Python function that's not working as expected.",
                icon="images/terminal.svg",
            ),
            cl.Starter(
                label="Code review",
                message="Can you review my code and suggest improvements for better performance?",
                icon="images/terminal.svg",
            )
        ]
    elif chat_profile == "The Teacher":
        return [
            cl.Starter(
                label="Explain superconductors",
                message="Explain superconductors like I'm five years old.",
                icon="images/learn.svg",
            ),
            cl.Starter(
                label="Learn quantum computing",
                message="Teach me about quantum computing in simple terms with practical examples.",
                icon="images/learn.svg",
            )
        ]
    elif chat_profile == "The Writer":
        return [
            cl.Starter(
                label="Creative story writing",
                message="Write a short story about time travel with an unexpected twist.",
                icon="images/write.svg",
            ),
            cl.Starter(
                label="Wedding invitation text",
                message="Write a text asking a friend to be my plus-one at a wedding next month. Keep it casual and offer an out.",
                icon="images/write.svg",
            )
        ]
    else:  # the thinker profile
        return [
            cl.Starter(
                label="Morning routine ideation",
                message="Help me create a personalized morning routine for maximum productivity. Start by asking about my current habits.",
                icon="images/idea.svg",
            ),
            cl.Starter(
                label="Strategic problem solving",
                message="I have a complex decision to make. Help me think through it systematically using structured analysis.",
                icon="images/idea.svg",
            )
        ]


@cl.on_chat_start
async def on_chat_start():
    """initialize chat session with profile-aware welcome and settings panel"""
    chat_profile = cl.user_session.get("chat_profile", "The Coder")
    
    # send profile-specific welcome message
    profile_messages = {
        "The Coder": "Ready to tackle programming challenges. What coding problem can I help you solve?",
        "The Teacher": "Hello! I'm here to help you learn and understand complex topics. What would you like to explore?",
        "The Writer": "Welcome! I'm your creative writing companion. Let's craft something amazing together.",
        "The Thinker": "Greetings! I'm here to help you think through problems and generate innovative solutions."
    }
    
    await cl.Message(content=profile_messages.get(chat_profile, "Hello! How can I help you today?")).send()
    
    # create dynamic settings panel for user customization
    settings = await cl.ChatSettings([
        Select(
            id="ResponseStyle",
            label="Response Style",
            values=["Detailed", "Concise", "Step-by-step", "Creative"],
            initial_index=0,
        ),
        Switch(
            id="ShowExamples",
            label="Include examples in responses",
            initial=True,
        ),
        Slider(
            id="Creativity",
            label="Creativity Level",
            initial=0.5,
            min=0,
            max=1,
            step=0.1
        ),
        Switch(
            id="DebugMode",
            label="Show debug information",
            initial=False,
        )
    ]).send()


@cl.on_settings_update
async def setup_settings(settings):
    """handle settings updates and store preferences in user session"""
    cl.user_session.set("response_style", settings["ResponseStyle"])
    cl.user_session.set("show_examples", settings["ShowExamples"])
    cl.user_session.set("creativity", settings["Creativity"])
    cl.user_session.set("debug_mode", settings["DebugMode"])
    
    # confirm settings update without verbose output
    await cl.Message(
        content=f"Settings updated: {settings['ResponseStyle']} style, creativity level {settings['Creativity']}, examples {'enabled' if settings['ShowExamples'] else 'disabled'}"
    ).send()



@cl.on_message
async def on_message(message: cl.Message):
    """handle incoming messages with profile-aware and settings-aware responses"""
    chat_profile = cl.user_session.get("chat_profile", "The Coder")
    response_style = cl.user_session.get("response_style", "Detailed")
    show_examples = cl.user_session.get("show_examples", True)
    creativity = cl.user_session.get("creativity", 0.5)
    debug_mode = cl.user_session.get("debug_mode", False)
    
    # generate profile-specific response headers
    profile_responses = {
        "The Coder": f"Code Analysis: {message.content}\n\nAs your programming assistant, I'd approach this systematically...",
        "The Teacher": f"Learning Topic: {message.content}\n\nLet me break this down step by step...",
        "The Writer": f"Creative Challenge: {message.content}\n\nWhat an interesting writing prompt! Here's how I'd approach it...",
        "The Thinker": f"Problem Analysis: {message.content}\n\nLet me think through this strategically..."
    }
    
    base_response = profile_responses.get(chat_profile, f"You said: {message.content}")
    
    # modify response based on user settings preferences
    if response_style == "Concise":
        base_response += "\n\n[Concise mode: Keeping response brief and focused]"
    elif response_style == "Step-by-step":
        base_response += "\n\nStep 1: First, let's analyze the core problem\nStep 2: Then we'll explore potential solutions\nStep 3: Finally, we'll implement the best approach"
    elif response_style == "Creative":
        base_response += f"\n\n[Creative mode at {creativity:.1f}: Adding innovative perspective to the response]"
    
    # conditionally add examples based on user preference
    if show_examples:
        base_response += "\n\nExample: Here's a practical demonstration of the concept discussed above..."
    
    # include debug information when enabled for development purposes
    if debug_mode:
        base_response += f"\n\nDebug Info: Profile={chat_profile}, Style={response_style}, Creativity={creativity}"
    
    base_response += "\n\nNote: This is a test app demonstrating profile and settings functionality. No AI model is currently connected."
    
    await cl.Message(content=base_response).send()


if __name__ == "__main__":
    cl.run()
