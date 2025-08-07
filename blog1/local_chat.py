#!/usr/bin/env python3
"""
Local LLM Chat Interface
A simple, privacy-focused ChatGPT alternative using Ollama
"""

import ollama
import sys
from datetime import datetime
from typing import List, Dict, Optional

class LocalLLM:
    """Main class for interacting with local LLM models via Ollama"""
    
    def __init__(self, model_name: str = 'gpt-oss:20b', system_prompt: Optional[str] = None):
        """
        Initialize the Local LLM interface
        
        Args:
            model_name: Name of the Ollama model to use
            system_prompt: Optional system prompt to set model behavior
        """
        self.model = model_name
        self.conversation_history: List[Dict[str, str]] = []
        
        # Add system prompt if provided
        if system_prompt:
            self.conversation_history.append({
                'role': 'system',
                'content': system_prompt
            })
        
        # Check if model exists, download if not
        self._initialize_model()
    
    def _initialize_model(self):
        """Check if model exists and download if necessary"""
        try:
            ollama.show(self.model)
            print(f"✓ Model '{self.model}' is ready!")
        except ollama.ResponseError:
            print(f"Model '{self.model}' not found. Downloading...")
            try:
                ollama.pull(self.model)
                print(f"✓ Model '{self.model}' downloaded successfully!")
            except Exception as e:
                print(f"❌ Error downloading model: {e}")
                print("Please run 'ollama pull {self.model}' manually")
                sys.exit(1)
    
    def chat(self, message: str, stream: bool = True, temperature: float = 0.7) -> str:
        """
        Send a message to the model and get a response
        
        Args:
            message: User message to send to the model
            stream: Whether to stream the response token by token
            temperature: Controls randomness in responses (0.0 to 1.0)
        
        Returns:
            The model's response as a string
        """
        # Add user message to history
        self.conversation_history.append({
            'role': 'user',
            'content': message
        })
        
        try:
            if stream:
                # Stream the response for a ChatGPT-like experience
                response_text = ""
                stream = ollama.chat(
                    model=self.model,
                    messages=self.conversation_history,
                    stream=True,
                    options={
                        'temperature': temperature
                    }
                )
                
                for chunk in stream:
                    content = chunk['message']['content']
                    print(content, end='', flush=True)
                    response_text += content
                
                print()  # New line after response
                
                # Add assistant response to history
                self.conversation_history.append({
                    'role': 'assistant',
                    'content': response_text
                })
                
                return response_text
            else:
                # Get the full response at once
                response = ollama.chat(
                    model=self.model,
                    messages=self.conversation_history,
                    options={
                        'temperature': temperature
                    }
                )
                
                response_text = response['message']['content']
                
                # Add assistant response to history
                self.conversation_history.append({
                    'role': 'assistant',
                    'content': response_text
                })
                
                return response_text
                
        except Exception as e:
            error_msg = f"Error generating response: {e}"
            print(f"❌ {error_msg}")
            # Remove the user message from history if response failed
            self.conversation_history.pop()
            return ""
    
    def clear_history(self):
        """Clear the conversation history"""
        # Keep system prompt if it exists
        system_messages = [msg for msg in self.conversation_history if msg['role'] == 'system']
        self.conversation_history = system_messages
        print("✓ Conversation history cleared.")
    
    def save_conversation(self, filename: Optional[str] = None) -> str:
        """
        Save the conversation to a text file
        
        Args:
            filename: Optional custom filename for the conversation
        
        Returns:
            The filename where the conversation was saved
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{timestamp}.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Conversation with {self.model}\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n\n")
                
                for message in self.conversation_history:
                    if message['role'] != 'system':  # Skip system prompts in saved file
                        role = message['role'].upper()
                        content = message['content']
                        f.write(f"{role}:\n{content}\n\n")
            
            print(f"✓ Conversation saved to {filename}")
            return filename
        except Exception as e:
            print(f"❌ Error saving conversation: {e}")
            return ""
    
    def get_conversation_stats(self) -> Dict[str, int]:
        """Get statistics about the current conversation"""
        user_messages = len([m for m in self.conversation_history if m['role'] == 'user'])
        assistant_messages = len([m for m in self.conversation_history if m['role'] == 'assistant'])
        total_chars = sum(len(m['content']) for m in self.conversation_history)
        
        return {
            'user_messages': user_messages,
            'assistant_messages': assistant_messages,
            'total_messages': user_messages + assistant_messages,
            'total_characters': total_chars
        }

def print_help():
    """Print help information about available commands"""
    print("\n📚 Available Commands:")
    print("  'quit' or 'exit' - End the conversation")
    print("  'clear' - Clear conversation history")
    print("  'save' - Save conversation to file")
    print("  'save <filename>' - Save with custom filename")
    print("  'model <name>' - Switch to a different model")
    print("  'stats' - Show conversation statistics")
    print("  'help' - Show this help message")
    print("  'models' - List available models")
    print()

def list_models():
    """List all available Ollama models"""
    try:
        models = ollama.list()
        if models['models']:
            print("\n📦 Available models:")
            for model in models['models']:
                size_gb = model['size'] / (1024**3)
                print(f"  • {model['name']} ({size_gb:.1f} GB)")
        else:
            print("\n❌ No models found. Pull a model first with 'ollama pull <model_name>'")
    except Exception as e:
        print(f"❌ Error listing models: {e}")

def main():
    """Main function to run the chat interface"""
    print("\n" + "="*50)
    print("🤖 Local LLM Chat Interface")
    print("="*50)
    print("Type 'help' for available commands")
    print("="*50)
    
    # Initialize with default model
    llm = LocalLLM('gpt-oss:20b')
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            # Check for empty input
            if not user_input:
                continue
            
            # Check for commands
            command_lower = user_input.lower()
            
            if command_lower in ['quit', 'exit']:
                # Offer to save before quitting
                save_prompt = input("💾 Save conversation before quitting? (y/n): ").strip().lower()
                if save_prompt == 'y':
                    llm.save_conversation()
                print("\n👋 Goodbye!")
                break
                
            elif command_lower == 'clear':
                llm.clear_history()
                continue
                
            elif command_lower == 'save':
                llm.save_conversation()
                continue
                
            elif command_lower.startswith('save '):
                filename = user_input[5:].strip()
                if filename:
                    llm.save_conversation(filename)
                else:
                    llm.save_conversation()
                continue
                
            elif command_lower.startswith('model '):
                model_name = user_input[6:].strip()
                if model_name:
                    print(f"Switching to model '{model_name}'...")
                    llm = LocalLLM(model_name)
                else:
                    print("❌ Please specify a model name")
                continue
                
            elif command_lower == 'stats':
                stats = llm.get_conversation_stats()
                print("\n📊 Conversation Statistics:")
                print(f"  User messages: {stats['user_messages']}")
                print(f"  Assistant messages: {stats['assistant_messages']}")
                print(f"  Total messages: {stats['total_messages']}")
                print(f"  Total characters: {stats['total_characters']:,}")
                continue
                
            elif command_lower == 'help':
                print_help()
                continue
                
            elif command_lower == 'models':
                list_models()
                continue
            
            # Send message to model
            print("\nAssistant: ", end='')
            llm.chat(user_input)
            
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\n\n⚠️  Interrupted. Type 'quit' to exit or continue chatting.")
            continue
        except EOFError:
            # Handle Ctrl+D
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            print("Please try again or type 'quit' to exit.")

if __name__ == "__main__":
    main()