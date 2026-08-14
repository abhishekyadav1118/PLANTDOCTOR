"""
Plant Disease Detection - Flask Backend with Gemini Integration
Run with: python app.py
Requires: flask, tensorflow, pillow, flask-cors, google-genai
Install with: pip install flask tensorflow pillow flask-cors google-genai

IMPORTANT: Set your Gemini API key as an environment variable before running.
Windows PowerShell:  $env:GEMINI_API_KEY="your_new_key_here"
Then run:            python app.py
"""

import json
import numpy as np
import os
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
from google import genai   # Gemini client

app = Flask(__name__)
CORS(app)  # allows the frontend (different device/port/file) to call this API

IMG_SIZE = 224

# ---- Gemini client setup (key read from environment, never hardcoded) ----
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
    print("Gemini client ready.")
else:
    client = None
    print("WARNING: GEMINI_API_KEY not set. AI explanations will be skipped.")

# ---- Load model and supporting files once when the server starts ----
print("Loading model... this may take a few seconds")
model = load_model("plant_model.h5")

with open("labels.json") as f:
    labels = json.load(f)  # {"0": "Tomato_Bacterial_spot", ...}

with open("solutions.json") as f:
    solutions = json.load(f)  # {"Tomato_Bacterial_spot": {"cause": ..., "solution": ...}}

print("Model loaded successfully. Server ready.")


def prepare_image(img_bytes):
    """Convert uploaded image bytes into the format the model expects."""
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = image.img_to_array(img)
    arr = arr / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def explain_with_gemini(disease, cause, solution):
    """Generate a simple Hindi + English explanation using Gemini.
    Returns a fallback message instead of crashing if Gemini is
    unavailable, misconfigured, or the request fails."""
    if client is None:
        return "AI explanation unavailable (no API key configured)."

    prompt = f"""
    Explain in simple Hindi and English:
    Disease: {disease}
    Cause: {cause}
    Solution: {solution}
    """

    try:
        response = client.models.generate_content(
            model="gemini-flash-latest",  # always points to the current stable flash model
            contents=prompt
        )
        return response.text
    except Exception as e:
        print(f"Gemini call failed: {e}")
        return "AI explanation unavailable right now (network or API issue)."


@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Plant Disease Detection API is running"})


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Send an image with key 'file'."}), 400

    file = request.files["file"]
    img_bytes = file.read()

    try:
        img_array = prepare_image(img_bytes)
    except Exception as e:
        return jsonify({"error": f"Could not process image: {str(e)}"}), 400

    # Run prediction
    predictions = model.predict(img_array)[0]
    predicted_index = int(np.argmax(predictions))
    confidence = float(predictions[predicted_index])
    predicted_class = labels[str(predicted_index)]

    # Look up cause + solution
    info = solutions.get(predicted_class, {
        "cause": "Not available",
        "solution": "Not available"
    })

    # Make the class name a bit more readable for display
    display_name = predicted_class.replace("___", " - ").replace("__", " - ").replace("_", " ")

    # Gemini explanation (never crashes the whole request if it fails)
    gemini_explanation = explain_with_gemini(display_name, info["cause"], info["solution"])

    return jsonify({
        "disease": display_name,
        "raw_class": predicted_class,
        "confidence": round(confidence * 100, 2),
        "cause": info["cause"],
        "solution": info["solution"],
        "gemini_explanation": gemini_explanation
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)