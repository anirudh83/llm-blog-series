# Flask web API
from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)
sentiment_analyzer = pipeline("sentiment-analysis")

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    text = request.json['text']
    result = sentiment_analyzer(text)
    return jsonify(result)
