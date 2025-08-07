# Hugging Face Transformers Tutorial

A hands-on tutorial demonstrating text generation and image classification using Hugging Face pipelines.

## Quick Start

```bash
pip install -r requirements.txt
python examples.py  # Run all examples
```

## What You'll Learn

- **Transformers**: Pre-trained models for NLP and Computer Vision
- **Pipelines**: High-level API for model inference
- **Practical Implementation**: Text generation and image classification

## Project Files

| File | Purpose | What it demonstrates |
|------|---------|---------------------|
| `examples.py` | **Main demo** | All examples in one organized file |
| `text-gen.py` | Text generation | Basic pipeline usage |
| `test-models.py` | Image classification | Local file processing |
| `learning.py` | Advanced examples | Lower-level model access |

## Quick Demo

Run the comprehensive examples:
```bash
python examples.py
```

Or run individual examples:
```bash
python text-gen.py      # Text generation only
python test-models.py   # Image classification only
```

## Examples

### Text Generation (`text-gen.py`)
```python
from transformers import pipeline

generator = pipeline("text-generation")
result = generator("In this course, we will teach you how to")
print(result)
```

### Image Classification (`test-models.py`)
```python
from transformers import pipeline
import requests

# Download and classify image
image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
response = requests.get(image_url)
with open("cat-image.jpeg", 'wb') as f:
    f.write(response.content)

classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
result = classifier("cat-image.jpeg")
print(result)
```

## Key Concepts

### Pipeline
A pipeline combines preprocessing, model inference, and postprocessing:
```python
pipeline = pipeline("task", model="model-name")
result = pipeline(input_data)
```

### Available Tasks
- `text-generation`: Complete text prompts
- `image-classification`: Classify images
- `sentiment-analysis`: Analyze text sentiment
- `translation`: Translate between languages

## Output Examples

**Text Generation:**
```
Prompt: In this tutorial, we will learn how to
Generated: In this tutorial, we will learn how to build and test a simple library using Node.js...
```

**Translation:**
```
French: Bonjour, comment allez-vous?
English: Hello, how are you?
```

**Sentiment Analysis:**
```
Text: 'I love using Hugging Face!'
Sentiment: POSITIVE (confidence: 1.000)
```

**Image Classification:**
```
Top predictions:
1. lynx, catamount: 0.433
2. cougar, puma: 0.035
3. snow leopard: 0.032
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `PIL` import error | `pip install Pillow` |
| Model download fails | Check internet connection |
| Out of memory | Use smaller models or CPU |

## Next Steps

1. Try different models from [Hugging Face Hub](https://huggingface.co/models)
2. Experiment with other tasks (translation, summarization)
3. Fine-tune models on your own data
4. Deploy models to production
