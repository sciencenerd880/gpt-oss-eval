import os
from dotenv import load_dotenv
from datetime import datetime

from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

# load environment variables
load_dotenv()


def create_search_agent():
    """Create a React agent with search capabilities using ChatGroq and TavilySearch"""
    
    # validate environment setup
    groq_api_key = os.getenv("GROQ_API_KEY")
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY not found in environment variables")
    
    # initialize chatgroq model - using llama3-70b for better stability
    model = ChatGroq(
        api_key=groq_api_key,
        model="llama3-70b-8192",  # more stable than gpt-oss for reasoning tasks
        temperature=0.1,  # low temperature for more focused search reasoning
        streaming=True
    )
    
    # create tavily search tool with result limit for manageable responses
    search = TavilySearch(
        api_key=tavily_api_key,
        max_results=3  # reasonable number of results to analyze
    )
    
    # enable conversation memory for multi-turn interactions
    checkpointer = InMemorySaver()
    
    # define tools available to the agent
    tools = [search]
    
    # create react agent with search-focused system prompt
    agent = create_react_agent(
        model=model,
        tools=tools,
        checkpointer=checkpointer,
        prompt="""You are a helpful research assistant that can search the web for information.

Your task:
1. When users ask questions, search for relevant information using the Tavily search tool
2. After getting search results, analyze and synthesize the information  
3. Provide a clear, comprehensive answer based on the search results
4. Always provide a final response - do not keep searching endlessly

Important guidelines:
- Use the search tool when you need current information
- After 1-2 searches, provide your answer based on the results you have
- If search results are poor quality, acknowledge this but still provide a response
- Be concise and helpful in your final answers
- Stop searching once you have enough information to answer the question

Remember: Your goal is to provide helpful answers, not to perform perfect searches."""
    )
    
    return agent


def run_search_agent():
    """Interactive function to run the search agent with integrated verbose logging"""
    print("Search Agent powered by Llama3-70B via Groq + Tavily")
    print("Ask me anything - I can search the web for current information!")
    print("Type 'quit' or 'exit' to stop\n")
    
    agent = create_search_agent()
    config = {
        "configurable": {"thread_id": "search_session_1"},
        "recursion_limit": 15  # prevent infinite loops
    }
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye! Happy researching!")
                break
                
            if not user_input:
                continue
            
            print("\n[AGENT] Processing your search request...")
            
            try:
                # stream execution to show tool calls and reasoning in real-time
                all_messages = []
                final_response_captured = False
                
                for step in agent.stream(
                    {"messages": [{"role": "user", "content": user_input}]},
                    config
                ):
                    # capture messages from steps that have them
                    if "messages" in step:
                        all_messages = step["messages"]
                    
                    # show reasoning and tool execution as they happen
                    for node, output in step.items():
                        if node == "agent" and "messages" in output and output["messages"]:
                            last_msg = output["messages"][-1]
                            
                            # check for tool calls first
                            if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                                for tool_call in last_msg.tool_calls:
                                    if tool_call['name'] == 'tavily_search':
                                        query = tool_call['args'].get('query', 'unknown')
                                        print(f"[REASONING] Agent decided to search for: '{query}'")
                            
                            # check for final response content
                            elif hasattr(last_msg, 'content') and last_msg.content:
                                content = str(last_msg.content).strip()
                                if content:
                                    print(f"\nAgent: {content}")
                                    final_response_captured = True
                                    
                        elif node == "tools":
                            print(f"[EXECUTION] Search completed")
                
                # fallback: if we didn't capture response during streaming, try from final messages
                if not final_response_captured:
                    if all_messages:
                        final_response = None
                        for msg in reversed(all_messages):
                            if (hasattr(msg, 'type') and msg.type == 'ai' and 
                                hasattr(msg, 'content') and msg.content):
                                content = str(msg.content).strip()
                                if content:
                                    final_response = content
                                    break
                        
                        if final_response:
                            print(f"\nAgent: {final_response}")
                        else:
                            print("\nAgent: [No AI response found]")
                    else:
                        print("\nAgent: [No messages received]")
                        
            except Exception as api_error:
                if "503" in str(api_error) or "Service unavailable" in str(api_error):
                    print("\n[ERROR] Groq service is currently unavailable. Please try again later.")
                    print("Check https://groqstatus.com/ for service status.")
                elif "Failed to call a function" in str(api_error):
                    print("\n[ERROR] Function calling issue with Groq API.")
                    print("This might be a temporary API issue. Try again in a moment.")
                else:
                    print(f"\n[ERROR] API Error: {api_error}")
                    print("Please try again or check your API keys.")
            
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye! Happy researching!")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.\n")


if __name__ == "__main__":
    run_search_agent()
