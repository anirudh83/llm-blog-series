
import requests
import json
import os
import re

# Configuration - Change this to use different models
MODEL_NAME = "llama3.2:1b"  # Fast model, change to "phi3:mini" for even faster

def call_ollama(prompt, model=MODEL_NAME):
    """Call local Ollama"""
    try:
        response = requests.post('http://localhost:11434/api/generate', json={
            'model': model, 'prompt': prompt, 'stream': False
        })
        result = response.json()
        
        # Check if we got a proper response
        if 'response' in result:
            return result['response']
        else:
            return f"Ollama response missing 'response' key: {result}"
            
    except Exception as e:
        return f"Ollama error: {e}"

def calculate(expr):
    """Calculator tool"""
    try:
        return f"{expr} = {eval(expr)}"
    except:
        return f"Math error with: {expr}"

def list_files(directory="."):
    """List files tool"""
    try:
        files = os.listdir(directory)
        return f"Files in {directory}: {', '.join(files[:10])}"
    except:
        return f"Can't list files in {directory}"

def write_file(filename, content):
    """Write file tool"""
    try:
        with open(filename, 'w') as f:
            f.write(content)
        return f"Wrote to {filename}"
    except:
        return f"Can't write to {filename}"

def read_file(filename):
    """Read file tool"""
    try:
        with open(filename, 'r') as f:
            content = f.read()
        return f"Content of {filename}: {content[:200]}..."
    except:
        return f"Can't read {filename}"

class SimpleAgent:
    def __init__(self):
        self.history = []
        self.tools = {
            'calculate': calculate,
            'list_files': list_files,
            'write_file': write_file,
            'read_file': read_file
        }
    
    def chat(self, user_input):
        """Main chat function"""
        
        # Build prompt with history and tools
        history_text = ""
        for h in self.history[-5:]:  # Last 5 exchanges
            history_text += f"User: {h['user']}\nAgent: {h['agent']}\n"
        
        prompt = f"""You are a helpful AI assistant with access to tools.

Available tools:
- calculate(expression) - for math calculations only
- list_files(directory) - when user asks to see/list files  
- write_file(filename, content) - when user asks to write/save to a file
- read_file(filename) - when user asks to read a specific file

IMPORTANT: Only use tools when the user explicitly asks for them.
- For greetings (hi, hello, hey) - just respond normally
- For general questions - just respond normally  
- For math questions - use calculate tool
- For file requests - use appropriate file tool

When you need a tool, use this exact format:
TOOL: tool_name
ARGS: arguments

For write_file, use: filename|content

Previous conversation:
{history_text}

User: {user_input}
Assistant:"""

        response = call_ollama(prompt)
        
        # Check for tool use
        if "TOOL:" in response and "ARGS:" in response:
            final_response = self._use_tool(response, user_input)
        else:
            final_response = response
        
        # Save to history
        self.history.append({'user': user_input, 'agent': final_response})
        return final_response
    
    def _use_tool(self, response, original_question):
        """Execute tools"""
        try:
            tool_name = re.search(r'TOOL:\s*(\w+)', response).group(1)
            tool_args = re.search(r'ARGS:\s*(.+)', response).group(1).strip()
            
            print(f"🔧 Using {tool_name} with: {tool_args}")
            
            if tool_name == 'write_file' and '|' in tool_args:
                filename, content = tool_args.split('|', 1)
                result = self.tools[tool_name](filename.strip(), content.strip())
            elif tool_name in self.tools:
                result = self.tools[tool_name](tool_args)
            else:
                result = f"Unknown tool: {tool_name}"
            
            # Get final response
            final_prompt = f"""User asked: {original_question}
Tool {tool_name} returned: {result}
Give a helpful response:"""
            
            return call_ollama(final_prompt)
            
        except Exception as e:
            return f"Tool error: {e}"

def main():
    """Interactive demo"""
    print("🤖 Simple AI Agent")
    print("Try: math, file operations, or just chat!")
    print("Type 'quit' to exit\n")
    
    agent = SimpleAgent()
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit']:
            print("Bye!")
            break
            
        if user_input:
            print("🤔 Thinking...")
            response = agent.chat(user_input)
            print(f"Agent: {response}\n")

if __name__ == "__main__":
    main()