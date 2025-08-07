#!/usr/bin/env python3
"""
Hugging Face Transformers Examples
Demonstrates text generation, translation, and image classification
"""

from transformers import pipeline
import requests
import os

def text_generation_example():
    """Generate text completion using GPT-style model"""
    print("🔤 Text Generation Example")
    print("-" * 40)
    
    generator = pipeline("text-generation", max_length=50)
    prompt = "In this tutorial, we will learn how to"
    
    result = generator(prompt)
    print(f"Prompt: {prompt}")
    print(f"Generated: {result[0]['generated_text']}")
    print()

def translation_example():
    """Translate French to English"""
    print("🌍 Translation Example")
    print("-" * 40)
    
    try:
        translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
        french_text = "Bonjour, comment allez-vous?"
        
        result = translator(french_text)
        print(f"French: {french_text}")
        print(f"English: {result[0]['translation_text']}")
        print()
    except Exception as e:
        print(f"Translation skipped: {e}")
        print("Install sentencepiece: pip install sentencepiece")
        print()

def image_classification_example():
    """Download and classify an image"""
    print("🖼️  Image Classification Example")
    print("-" * 40)
    
    # Download image
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
    image_path = "demo-image.jpeg"
    
    if not os.path.exists(image_path):
        print(f"Downloading image...")
        response = requests.get(image_url)
        with open(image_path, 'wb') as f:
            f.write(response.content)
        print(f"✅ Image saved to {image_path}")
    
    # Classify image
    classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
    results = classifier(image_path)
    
    print("Top predictions:")
    for i, result in enumerate(results[:3], 1):
        print(f"{i}. {result['label']}: {result['score']:.3f}")
    print()

def sentiment_analysis_example():
    """Analyze sentiment of text"""
    print("😊 Sentiment Analysis Example")
    print("-" * 40)
    
    sentiment_analyzer = pipeline("sentiment-analysis")
    texts = [
        "I love using Hugging Face!",
        "This is frustrating and confusing.",
        "The weather is okay today."
    ]
    
    for text in texts:
        result = sentiment_analyzer(text)
        label = result[0]['label']
        score = result[0]['score']
        print(f"Text: '{text}'")
        print(f"Sentiment: {label} (confidence: {score:.3f})")
        print()

def main():
    """Run all examples"""
    print("🤗 Hugging Face Transformers Examples")
    print("=" * 50)
    print()
    
    try:
        text_generation_example()
        translation_example()
        sentiment_analysis_example()
        image_classification_example()
        
        print("✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have installed: pip install transformers torch pillow requests sentencepiece")

if __name__ == "__main__":
    main()
