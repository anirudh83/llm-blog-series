from transformers import pipeline

print("Lets generate...")
generator = pipeline("text-generation")
result = generator("In this course, we will teach you how to")
print(result)