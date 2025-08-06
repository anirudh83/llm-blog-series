# Local LLM Chat Interface

A simple, privacy-focused ChatGPT alternative that runs entirely on your local machine. Built with Python and Ollama.

## Features

- 🔒 **100% Private** - Your data never leaves your machine
- 💬 **Interactive Chat** - Stream responses in real-time like ChatGPT
- 📝 **Conversation History** - Maintains context across messages
- 💾 **Save Conversations** - Export your chats to text files
- 🔄 **Model Switching** - Easily switch between different models
- 🎯 **Simple Setup** - Get running in under 5 minutes

## Prerequisites

- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended)
- 5-10GB free disk space for models

## Installation

### Step 1: Install Ollama

**macOS/Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**
Download the installer from [ollama.com/download](https://ollama.com/download)

Verify installation:
```bash
ollama --version
```

### Step 2: Clone this Repository

```bash
git clone git@github.com:anirudh83/llm-blog-series.git
cd blog1
```

### Step 3: Install Python Dependencies

##### Optional : use python virtual env for installing python dependencies.
```bash
python3 -m venv .venv
source .venv/bin/activate
```

```bash
pip install ollama
```

Or if you prefer using requirements.txt:
```bash
pip install -r requirements.txt
```


### Step 4: Download a Model

```bash
# Recommended: Llama 3.2 (3.9GB)
ollama pull llama3.2

# Alternative smaller model: Phi-3 (2.2GB)
ollama pull phi3:mini

# Alternative smaller : Llama 3.2 (1GB)
ollama pull llama3.2:1b

# For code generation: CodeLlama (3.8GB)
ollama pull codellama
```

## Usage

### Basic Usage

Run the chat interface:
```bash
python3 local_chat.py
```

### Available Commands

While chatting, you can use these commands:
- `quit` or `exit` - End the conversation
- `clear` - Clear conversation history
- `save` - Save conversation to a timestamped file
- `model <name>` - Switch to a different model (e.g., `model codellama`)

### Example Session

```
$ python local_chat.py
🤖 Local LLM Chat Interface
========================================
Commands:
  'quit' or 'exit' - End the conversation
  'clear' - Clear conversation history
  'save' - Save conversation to file
  'model <name>' - Switch to a different model
========================================
✓ Model 'llama3.2' is ready!

You: What's the capital of France?