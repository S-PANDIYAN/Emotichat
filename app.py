from flask import Flask, request, jsonify, render_template
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import aiml
import os

# Setup Flask app
app = Flask(__name__)

# Load tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained("saved_model")
model = RobertaForSequenceClassification.from_pretrained("saved_model")
model.eval()

# Load AIML kernel
kernel = aiml.Kernel()
kernel.learn("emotion-angry.aiml")
kernel.learn("emotion-neutral.aiml")
kernel.learn("emotion-sad.aiml")

# Label mapping (adjust if needed)
labels = {0: "anger", 1: "disgust", 2: "fear", 3: "joy", 4: "neutral", 5: "sadness", 6: "surprise"}

# Serve HTML
@app.route("/")
def index():
    return render_template("index.html")

# Chat route
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["message"]
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        prediction = torch.argmax(logits, dim=1).item()
        emotion = labels[prediction]
    
    response = kernel.respond(emotion)  # Emotion as input to AIML
    return jsonify({"emotion": emotion, "response": response})

if __name__ == "__main__":
    app.run(debug=True)
