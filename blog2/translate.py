from transformers import pipeline

print("Now lets translate..")
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
translation_result = translator("Ce cours est produit par Hugging Face.")
print(translation_result)