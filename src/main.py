import os
from groq import Groq
from dotenv import load_dotenv


def test_gpt_oss_basic():
    """basic test of gpt oss model via groq api"""
    load_dotenv()
    
    # initialize groq client with api key from environment
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    print("testing openai gpt oss 20b model via groq api")
    
    try:
        # test basic completion with standard parameters for consistent evaluation
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "user",
                    "content": "Hi, who are you? What is the difference between you and GPT-4o and the o1 reasoning models? Explain to me in a few sentences."
                }
            ],
            temperature=0.7,
            max_completion_tokens=1024,
            top_p=1,
            reasoning_effort="medium",  # medium reasoning effort for balanced response
            stream=False,
            stop=None
        )
        
        response = completion.choices[0].message.content
        print("response:")
        print(response)
        
        return response
        
    except Exception as e:
        print(f"error: {e}")
        return None


def test_gpt_oss_streaming():
    """test streaming response from gpt oss model"""
    load_dotenv()
    
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    print("testing streaming response")
    
    try:
        # test streaming with creative prompt and higher temperature for variety
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "user",
                    "content": "Tell me a short story about a robot learning to paint. Make it creative and engaging."
                }
            ],
            temperature=0.8,  # higher temperature for more creativity
            max_completion_tokens=512,
            top_p=1,
            reasoning_effort="medium",
            stream=True,
            stop=None
        )
        
        print("streaming response:")
        
        full_response = ""
        for chunk in completion:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content
                
        print("\n")
        return full_response
        
    except Exception as e:
        print(f"error: {e}")
        return None


def test_gpt_oss_reasoning():
    """test reasoning capabilities with different reasoning effort levels"""
    load_dotenv()
    
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    print("testing reasoning capabilities")
    
    # test different reasoning levels to compare output quality
    reasoning_levels = ["low", "medium", "high"]
    
    for level in reasoning_levels:
        print(f"testing reasoning_effort='{level}':")
        
        try:
            completion = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {
                        "role": "user",
                        "content": "If I have 3 apples and I buy 2 more, then give away 1, how many apples do I have? Show your reasoning step by step."
                    }
                ],
                temperature=0.1,  # low temperature for consistent reasoning
                max_completion_tokens=256,
                top_p=1,
                reasoning_effort=level,
                stream=False,
                stop=None
            )
            
            response = completion.choices[0].message.content
            print(f"response: {response}")
            
        except Exception as e:
            print(f"error with {level} reasoning: {e}")


def test_conversation():
    """test multi-turn conversation with context"""
    load_dotenv()
    
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    print("testing multi-turn conversation")
    
    # multi-turn conversation to test context handling and coherence
    messages = [
        {
            "role": "user",
            "content": "hi, who are you? what is the difference between you and GPT 4o and the o1 Reasoning models. Explain to me in a few sentences"
        },
        {
            "role": "assistant", 
            "content": "I'm ChatGPT, which runs on the GPT-4.5 architecture (the latest evolution of GPT-4). I'm designed for conversational use, with a focus on natural language dialogue, personalization, extended context, and integrated tools.\n\n**GPT-4o** (\"GPT-4o\") is the earlier GPT-4 \"omni\" model, more focused on multimodal and \"all-purpose\" tasks but less optimized for fine-tuned chat subtleties.\n\n**o1 Reasoning Models** are specialized research models built to excel at formal reasoning tasks (logic, math, deduction). They're narrow-scope engines trained to answer puzzle-style or proof-style questions, not meant for general chat or open-domain conversation.\n\nIn short: I'm a general-purpose, conversational GPT-4.5 model; GPT-4o is an earlier, more broadly tuned sibling; o1 is a reasoning-specialized research engine, not a conversation chatbot."
        },
        {
            "role": "user",
            "content": "its related to home team"
        }
    ]
    
    try:
        # test conversation with streaming to see real-time response development
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            temperature=1,  # high temperature for more varied responses
            max_completion_tokens=1024,
            top_p=1,
            reasoning_effort="medium",
            stream=True,
            stop=None
        )
        
        print("conversation response:")
        
        for chunk in completion:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
                
        print("\n")
        
    except Exception as e:
        print(f"error: {e}")


def main():
    """run all gpt oss model tests"""
    print("gpt oss model testing suite")
    
    # run comprehensive tests to evaluate model capabilities across different scenarios
    # test_gpt_oss_basic()          # basic response test
    test_gpt_oss_streaming()      # streaming functionality
    # test_gpt_oss_reasoning()      # reasoning at different levels
    # test_conversation()           # multi-turn conversation
    
    print("all tests completed")


if __name__ == "__main__":
    main()
