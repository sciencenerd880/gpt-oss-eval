import os
from dotenv import load_dotenv
from typing import List, Dict, Any

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

# load environment variables
load_dotenv()


@tool
# def duckduckgo_search(query: str, max_results: int = 3, REQD_DEBUG: bool = True) -> str:
def duckduckgo_search(query: str, max_results: int = 3, REQD_DEBUG: bool = False) -> str:
    """Search the web using DuckDuckGo search engine.
    
    Args:
        query: The search query string
        max_results: Maximum number of results to return (default 3)
    
    Returns:
        A formatted string containing search results with titles, URLs, and snippets
    """
    try:
        from ddgs import DDGS
        
        print(f"[TOOL] Searching DuckDuckGo for: '{query}'")
        
        # initialize duckduckgo search - no api key required
        with DDGS() as ddgs:
            # perform web search with specified limit - use us-en region for better results
            results = list(ddgs.text(query, max_results=max_results, region='us-en'))
        
        if not results:
            return f"No search results found for query: {query}"
        
        # print the results for traceability
        if REQD_DEBUG == True:
            print(f"\nDEBUG: The results from DDGS - \n{results}\n")
        # format results for the agent
        formatted_results = f"Search results for '{query}':\n\n"
        
        for i, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            url = result.get('href', 'No URL')
            snippet = result.get('body', 'No description')
            
            # truncate snippet for readability
            if len(snippet) > 200:
                snippet = snippet[:200] + "..."
            
            formatted_results += f"{i}. **{title}**\n"
            formatted_results += f"   URL: {url}\n"
            formatted_results += f"   Description: {snippet}\n\n"
        
        print(f"[TOOL] Found {len(results)} search results")
        return formatted_results
        
    except ImportError:
        return "Error: ddgs package not installed. Run: uv add ddgs"
    except Exception as e:
        return f"Search error: {str(e)}"


def create_search_agent():
    """Create a React agent with DuckDuckGo search capabilities using ChatGroq"""
    
    # validate environment setup - only groq api key needed
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    
    # initialize chatgroq model - using llama3-70b for better stability
    model = ChatGroq(
        api_key=groq_api_key,
        model="llama3-70b-8192",  # more stable than gpt-oss for reasoning tasks
        temperature=0.1,  # low temperature for more focused search reasoning
        streaming=True
    )
    
    # enable conversation memory for multi-turn interactions
    checkpointer = InMemorySaver()
    
    # define tools available to the agent - using duckduckgo search
    tools = [duckduckgo_search]
    
    # create react agent with search-focused system prompt
    agent = create_react_agent(
        model=model,
        tools=tools,
        checkpointer=checkpointer,
        prompt="""You are a helpful research assistant that can search the web for information.

Your task:
1. When users ask questions, search for relevant information using the DuckDuckGo search tool
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
    print("Search Agent powered by Llama3-70B via Groq + DuckDuckGo")
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
                                if tool_call['name'] == 'duckduckgo_search':
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
            
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye! Happy researching!")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.\n")


if __name__ == "__main__":
    run_search_agent()
