import os
import random
from dotenv import load_dotenv
from typing import Dict, Any
from datetime import datetime

from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import tool

# Load environment variables
load_dotenv()


@tool
def get_weather(city: str) -> str:
    """Get current weather information for a given city.
    
    Args:
        city: The name of the city to get weather for
        
    Returns:
        A string with current weather conditions including temperature, humidity, and description
    """
    # log tool invocation for visibility
    print(f"[TOOL] Getting current weather for {city}")
    
    # simulate realistic weather data for demonstration
    weather_conditions = [
        "sunny", "partly cloudy", "cloudy", "light rain", 
        "heavy rain", "snow", "foggy", "windy"
    ]
    
    # generate baseline weather values
    temperature = random.randint(-10, 35)  # celsius
    humidity = random.randint(30, 90)
    condition = random.choice(weather_conditions)
    wind_speed = random.randint(5, 25)
    
    # adjust weather based on geographic patterns for realism
    city_lower = city.lower()
    if "desert" in city_lower or city_lower in ["phoenix", "las vegas", "dubai"]:
        temperature = random.randint(25, 45)
        humidity = random.randint(10, 30)
        condition = random.choice(["sunny", "partly cloudy", "windy"])
    elif city_lower in ["seattle", "london", "vancouver"]:
        temperature = random.randint(5, 20)
        humidity = random.randint(60, 90)
        condition = random.choice(["cloudy", "light rain", "heavy rain", "foggy"])
    elif city_lower in ["moscow", "helsinki", "anchorage"]:
        temperature = random.randint(-20, 10)
        condition = random.choice(["snow", "cloudy", "foggy"])
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    return f"""Current weather in {city} (as of {current_time}):
Temperature: {temperature}°C
Humidity: {humidity}%
Conditions: {condition}
Wind Speed: {wind_speed} km/h
Location: {city}

Weather data simulated for demonstration purposes."""


@tool  
def get_weather_forecast(city: str, days: int = 3) -> str:
    """Get weather forecast for a given city for the next few days.
    
    Args:
        city: The name of the city to get forecast for
        days: Number of days to forecast (1-7, default 3)
        
    Returns:
        A string with weather forecast for the specified number of days
    """
    # log tool invocation with parameters
    print(f"[TOOL] Getting {days}-day forecast for {city}")
    
    # constrain forecast days to reasonable range
    days = max(1, min(days, 7))
    
    forecast = f"Weather forecast for {city} - Next {days} days:\n\n"
    
    weather_conditions = ["sunny", "partly cloudy", "cloudy", "light rain", "heavy rain", "snow", "foggy"]
    
    for day in range(days):
        day_name = ["Today", "Tomorrow", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7"][day]
        temp_high = random.randint(15, 30)
        temp_low = random.randint(5, temp_high - 5)  # ensure low is below high
        condition = random.choice(weather_conditions)
        rain_chance = random.randint(0, 100)
        
        forecast += f"{day_name}: {condition.title()}\n"
        forecast += f"   High: {temp_high}°C, Low: {temp_low}°C\n"
        forecast += f"   Rain chance: {rain_chance}%\n\n"
    
    forecast += "Forecast data simulated for demonstration purposes."
    return forecast


def create_weather_agent():
    """Create a React agent with weather tools using ChatGroq"""
    
    # validate environment setup
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    
    # initialize model with conservative temperature for consistent tool use
    model = ChatGroq(
        api_key=api_key,
        model="openai/gpt-oss-20b",  # using same model as app2.py
        temperature=0.1,
        streaming=True
    )
    
    # enable conversation memory across interactions
    checkpointer = InMemorySaver()
    
    # define available tools for weather operations
    tools = [get_weather, get_weather_forecast]
    
    # create react agent with focused weather assistant prompt
    agent = create_react_agent(
        model=model,
        tools=tools,
        checkpointer=checkpointer,
        prompt="""You are a helpful weather assistant powered by GPT-OSS via Groq. 

You have access to weather tools that can provide current weather and forecasts for any city worldwide. 
When users ask about weather, use the appropriate tools to get the information.

Key capabilities:
- Get current weather conditions for any city
- Provide weather forecasts for up to 7 days
- Answer general questions about weather patterns
- Give weather-related advice

Always be friendly, informative, and mention that the weather data is simulated for demonstration purposes.
If users ask about non-weather topics, politely redirect them back to weather-related questions or provide general assistance."""
    )
    
    return agent


def run_weather_agent():
    """Interactive function to run the weather agent"""
    print("Weather Agent powered by GPT-OSS via Groq")
    print("Ask me about weather in any city!")
    print("Type 'quit' or 'exit' to stop\n")
    
    agent = create_weather_agent()
    config = {"configurable": {"thread_id": "weather_session_1"}}
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye! Stay weather-aware!")
                break
                
            if not user_input:
                continue
            
            print("\n[AGENT] Processing your request...")
            
            # invoke agent and get response
            response = agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config
            )
            
            # extract final message content
            final_message = response["messages"][-1].content
            print(f"\nAgent: {final_message}")
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye! Stay weather-aware!")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.\n")


def demo_weather_agent():
    """Demo function showing the agent capabilities"""
    print("Weather Agent Demo")
    print("=" * 50)
    
    agent = create_weather_agent()
    config = {"configurable": {"thread_id": "demo_session"}}
    
    # sample queries to demonstrate functionality
    demo_queries = [
        "What's the weather like in San Francisco?",
        "Can you give me a 5-day forecast for Tokyo?",
        "How's the weather in London today?",
        "What about the forecast for New York this week?"
    ]
    
    for query in demo_queries:
        print(f"\nUser: {query}")
        print("[AGENT] Processing request...")
        
        try:
            response = agent.invoke(
                {"messages": [{"role": "user", "content": query}]},
                config
            )
            
            final_message = response["messages"][-1].content
            print(f"\nAgent: {final_message}")
            
        except Exception as e:
            print(f"Error: {e}")


def run_verbose_agent():
    """Run agent with detailed step-by-step visibility"""
    print("Weather Agent (Verbose Mode) - powered by GPT-OSS via Groq")
    print("You can see each step of the agent's reasoning process")
    print("Type 'quit' or 'exit' to stop\n")
    
    agent = create_weather_agent()
    config = {"configurable": {"thread_id": "verbose_session"}}
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
                
            if not user_input:
                continue
            
            print("\n" + "="*60)
            print("[AGENT TRACE] Starting reasoning process...")
            print("="*60)
            
            # stream the agent's execution to see intermediate steps
            for step in agent.stream(
                {"messages": [{"role": "user", "content": user_input}]},
                config
            ):
                # show each step of the agent's reasoning
                for node, output in step.items():
                    if node == "agent":
                        if "messages" in output and output["messages"]:
                            last_msg = output["messages"][-1]
                            if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                                for tool_call in last_msg.tool_calls:
                                    print(f"[REASONING] Agent decided to call: {tool_call['name']}")
                                    print(f"[REASONING] With arguments: {tool_call['args']}")
                    elif node == "tools":
                        print(f"[EXECUTION] Tool execution completed")
            
            print("="*60)
            print("[AGENT TRACE] Reasoning complete\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "demo":
            demo_weather_agent()
        elif sys.argv[1] == "verbose":
            run_verbose_agent()
        else:
            print("Usage: python react_agent.py [demo|verbose]")
            print("  demo    - Run predefined demo queries")
            print("  verbose - Interactive mode with detailed agent tracing")
            print("  (no args) - Standard interactive mode")
    else:
        run_weather_agent()
