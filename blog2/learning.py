from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

text = "I love this!"

# Method 1: Using pipeline (what you've been doing)
classifier = pipeline("sentiment-analysis")
result = classifier(text)
print(f"Pipeline result: {result}\n")

# Method 2: Doing it manually (what pipeline does internally)
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Step by step:
# 1. Tokenize
inputs = tokenizer(text, return_tensors="pt")
print(f"1. Token IDs: {inputs['input_ids'][0].tolist()}")

# 2. Run through model
outputs = model(**inputs)
print(f"2. Raw logits: {outputs.logits[0].tolist()}")

# 3. Convert to probabilities
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(f"3. Probabilities: {predictions[0].tolist()}")

# 4. Get the label
predicted_class = predictions.argmax().item()
labels = ['NEGATIVE', 'POSITIVE']
confidence = predictions[0][predicted_class].item()
print(f"4. Final: {labels[predicted_class]} ({confidence:.2%})")