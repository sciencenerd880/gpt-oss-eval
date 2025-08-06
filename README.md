# GPT OSS Evaluation

Testing OpenAI's GPT OSS models via Groq API. This project provides a comprehensive test suite to evaluate the capabilities of OpenAI's open-source GPT models including basic responses, streaming, reasoning at different levels, and multi-turn conversations.

## Features

- **Basic Response Testing** - Standard completion with configurable parameters
- **Streaming Response** - Real-time response generation for creative tasks
- **Reasoning Capabilities** - Testing different reasoning effort levels (low, medium, high)
- **Multi-turn Conversations** - Context-aware conversation handling
- **Rate Limit Handling** - Built-in error handling for API rate limits

## Getting Started

### Prerequisites

- Python 3.11 or higher
- [uv package manager](https://docs.astral.sh/uv/) (recommended) or pip
- Groq API key (free tier available)

### 1. Get Your Groq API Key

1. Visit [Groq Console](https://console.groq.com/)
2. Sign up for a free account
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key (starts with `gsk_`)

### 2. Clone and Setup

```bash
git clone <your-repo-url>
cd gpt-oss-eval
```

### 3. Environment Setup

Create a `.env` file in the project root:

```bash
# Create .env file
touch .env
```

Add your Groq API key to `.env`:

```env
# Environment variables
GROQ_API_KEY=your_groq_api_key_here
```

**Important**: Never commit your `.env` file to version control. It's already included in `.gitignore`.

### 4. Install Dependencies

#### Using uv (recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

#### Using pip with virtual environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install dependencies
pip install groq python-dotenv
```

### 5. Run the Application

#### Interactive Web UI (Recommended)

```bash
uv run chainlit run src/app.py
```

This starts a **clean, simple web interface** at `http://localhost:8000` powered by LangChain and Groq:

- **Simple Chat Interface**: Clean, distraction-free conversation experience
- **Real-time Streaming**: See responses as they're generated
- **Automatic Context**: Maintains conversation history
- **Error Handling**: Graceful handling of rate limits and API issues
- **OpenAI GPT OSS 20B**: Pre-configured with optimal settings

#### Command Line Testing

```bash
uv run src/main.py
```

This runs comprehensive tests via command line to evaluate all model capabilities.

## Web Interface Features

The LangChain-powered Chainlit interface provides:

### � Simple & Clean Design
- **Focus on Conversation**: Minimal, distraction-free chat interface
- **Real-time Streaming**: Watch responses generate in real-time
- **Context Awareness**: Automatically maintains conversation history
- **Welcome Screen**: Informative introduction to the GPT OSS model

### � Optimized Configuration
- **Pre-tuned Settings**: GPT OSS 20B with optimal parameters
- **Temperature**: 0.7 for balanced creativity and accuracy
- **Streaming Enabled**: Real-time response generation
- **Error Resilience**: Automatic handling of rate limits and API issues

### 🤖 LangChain Integration
- **Reliable Architecture**: Built on battle-tested LangChain framework
- **Groq Provider**: Lightning-fast inference via specialized hardware
- **Prompt Templates**: Professionally crafted system prompts
- **Callback Handling**: Full observability of the AI pipeline

## Model Testing Suite

The test suite includes four main tests:

1. **Basic Response Test** (`test_gpt_oss_basic`)
   - Simple completion with standard parameters
   - Tests basic model functionality

2. **Streaming Response Test** (`test_gpt_oss_streaming`)
   - Real-time response generation
   - Creative storytelling prompt

3. **Reasoning Capabilities Test** (`test_gpt_oss_reasoning`)
   - Tests different reasoning effort levels
   - Mathematical problem solving

4. **Multi-turn Conversation Test** (`test_conversation`)
   - Context-aware conversation
   - Tests conversation memory

## Groq Free Tier Rate Limits

When using the free tier, be aware of the following rate limits:

| Model | RPM | RPD | TPM | TPD | Notes |
|-------|-----|-----|-----|-----|-------|
| openai/gpt-oss-20b | 30 | 1,000 | 8,000 | 200,000 | Available on free tier |
| openai/gpt-oss-120b | 30 | 1,000 | 8,000 | 200,000 | Available on free tier |

**Legend:**
- **RPM**: Requests per minute
- **RPD**: Requests per day  
- **TPM**: Tokens per minute
- **TPD**: Tokens per day

### Rate Limit Handling

When you exceed rate limits, the API returns:
- **HTTP Status**: 429 Too Many Requests
- **Retry-After Header**: Time to wait before retrying (when applicable)

The test suite includes basic error handling for rate limits. If you encounter rate limiting:

1. Wait for the specified time in the `retry-after` header
2. Reduce the frequency of requests
3. Consider upgrading to a paid plan for higher limits

## Configuration

### Model Parameters

You can modify the following parameters in the test functions:

- `temperature`: Controls randomness (0.1 = conservative, 1.0 = creative)
- `max_completion_tokens`: Maximum tokens in response
- `reasoning_effort`: "low", "medium", or "high"
- `stream`: Boolean for streaming vs complete responses

### Example Configuration

```python
completion = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[...],
    temperature=0.7,  # balanced creativity
    max_completion_tokens=1024,
    reasoning_effort="medium",  # balanced reasoning
    stream=False  # get complete response
)
```

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'groq'**
   - Make sure you've installed dependencies with `uv sync` or `pip install`
   - Verify virtual environment is activated if using pip

2. **API Key Error**
   - Check that `.env` file exists and contains `GROQ_API_KEY`
   - Verify API key is valid and not expired
   - Ensure no extra spaces in the `.env` file

3. **Rate Limit Errors (429)**
   - Wait before making more requests
   - Check your usage in Groq Console
   - Consider spacing out test runs

4. **Build Errors with uv**
   - Make sure `pyproject.toml` is properly configured
   - Try `uv sync --refresh` to refresh dependencies

### Getting Help

- Check [Groq Documentation](https://console.groq.com/docs)
- Review [OpenAI GPT OSS Model Info](https://huggingface.co/openai/gpt-oss-20b)
- Examine error messages for specific HTTP status codes

## Project Structure

```
gpt-oss-eval/
├── .env                    # Environment variables (create this)
├── .gitignore             # Git ignore file
├── pyproject.toml         # Project configuration
├── README.md              # This file
├── uv.lock               # uv dependency lock file
└── src/
    └── main.py           # Main test suite
```
