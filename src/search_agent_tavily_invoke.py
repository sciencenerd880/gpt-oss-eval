import os
from dotenv import load_dotenv

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
        streaming=False  # use invoke instead of streaming for reliability
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
    
    # create react agent with improved prompt for better stopping behavior
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
    """Interactive function to run the search agent using invoke method for reliability"""
    print("Search Agent powered by Llama3-70B via Groq + Tavily (Invoke Mode)")
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
            
            # use invoke method for more reliable execution
            try:
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": user_input}]},
                    config
                )
                
                # extract final response from result
                if result and "messages" in result and result["messages"]:
                    final_message = result["messages"][-1]
                    if hasattr(final_message, 'content') and final_message.content:
                        print(f"\nAgent: {final_message.content}")
                    else:
                        print("\nAgent: [No content in final message]")
                        # debug info
                        print(f"[DEBUG] Message type: {type(final_message)}")
                        print(f"[DEBUG] Message attributes: {[attr for attr in dir(final_message) if not attr.startswith('_')]}")
                else:
                    print("\nAgent: [No messages in result]")
                    
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
