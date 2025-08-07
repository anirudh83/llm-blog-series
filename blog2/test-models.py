from transformers import pipeline
import requests
from PIL import Image
import os

#generator
# print("Lets generate...")
# generator = pipeline("text-generation")
# result = generator("In this course, we will teach you how to")
# print(result)


# print("Now lets translate..")
# translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
# translation_result = translator("Ce cours est produit par Hugging Face.")
# print(translation_result)

print("Lets generate image..")

# Download the image to local file system
image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image_path = "cat-image.jpeg"

print(f"Downloading image to {image_path}...")
response = requests.get(image_url)
with open(image_path, 'wb') as f:
    f.write(response.content)
print(f"Image saved to {os.path.abspath(image_path)}")

# Now classify the local image
image_classifier = pipeline(
    task="image-classification", model="google/vit-base-patch16-224"
)
image_result = image_classifier(image_path)
print("Classification results:")
print(image_result)