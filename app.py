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
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing import image  # type: ignore
from PIL import Image
try:
    from google import genai   # Gemini client
except ImportError:
    genai = None

from flask import send_file

app = Flask(__name__)
CORS(app)  # allows the frontend (different device/port/file) to call this API

IMG_SIZE = 224

# Auto-load .env file if present
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))
    except Exception as e:
        print(f"Warning: Could not read .env file: {e}")

# ---- Gemini client setup ----
client = None

def get_gemini_client():
    global client
    if client is not None:
        return client
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if api_key and genai:
        try:
            client = genai.Client(api_key=api_key)
            print("Gemini client ready.")
            return client
        except Exception as e:
            print(f"Gemini client initialization failed: {e}")
            return None
    return None

client = get_gemini_client()
if client is None:
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        print("WARNING: GEMINI_API_KEY / GOOGLE_API_KEY not set. AI explanations will be skipped.")
    elif not genai:
        print("WARNING: google-genai not installed. AI explanations will be skipped.")

# ---- Load model and supporting files once when the server starts ----
print("Loading model... this may take a few seconds")
model = load_model("plant_model.h5")

with open("labels.json") as f:
    labels = json.load(f)  # {"0": "Tomato_Bacterial_spot", ...}

with open("solutions.json") as f:
    solutions = json.load(f)  # {"Tomato_Bacterial_spot": {"cause": ..., "solution": ...}}

print("Model loaded successfully. Server ready.")


def prepare_image(img_bytes):
    """Convert uploaded image bytes into PIL Image and model tensor format."""
    img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_resized = img_pil.resize((IMG_SIZE, IMG_SIZE))
    arr = image.img_to_array(img_resized)
    arr = arr / 255.0
    arr = np.expand_dims(arr, axis=0)
    return img_pil, arr


def validate_plant_image(img_pil, img_bytes):
    """Checks if the uploaded image is actually a plant/crop leaf.
    Uses Gemini Vision if configured, alongside local digital skin-tone
    and foliage color distribution analysis.
    Returns: (is_plant, reason, error_message)"""
    # 1. Multimodal Gemini Vision check if available
    c = get_gemini_client()
    if c is not None:
        for m_name in ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-2.5-flash"]:
            try:
                prompt = (
                    "Examine this image carefully. Is this image a photo of a plant leaf or crop leaf? "
                    "Or does it show a person, human face, animal, vehicle, indoor room, or non-plant object? "
                    "Reply with ONLY JSON in this exact format: {\"is_plant_leaf\": true or false, \"reason\": \"brief explanation\"}"
                )
                resp = c.models.generate_content(
                    model=m_name,
                    contents=[img_pil, prompt]
                )
                text = resp.text.strip()
                if "{" in text and "}" in text:
                    parsed = json.loads(text[text.find("{"):text.rfind("}") + 1])
                    if not parsed.get("is_plant_leaf", True):
                        return False, "not_plant", parsed.get("reason", "Photo does not appear to be a plant leaf.")
                break
            except Exception as e:
                print(f"Gemini leaf validation ({m_name}) error: {e}")

    # 2. Local Digital Skin Tone and Foliage Verification (fast, offline)
    rgb = np.array(img_pil.convert("RGB"), dtype=np.float32)
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    total_pixels = rgb.shape[0] * rgb.shape[1]

    # Standard YCbCr skin tone detection: Cb in [77, 127], Cr in [133, 173], R > G > B
    cb = -0.1687 * r - 0.3313 * g + 0.5 * b + 128.0
    cr = 0.5 * r - 0.4187 * g - 0.0813 * b + 128.0
    skin_mask = (cb >= 77) & (cb <= 127) & (cr >= 133) & (cr <= 173) & (r > g) & (g > b)
    skin_pct = np.sum(skin_mask) / total_pixels

    # Plant foliage detection (green foliage, chlorotic yellow foliage, necrotic brown spots)
    green_foliage = (g > r * 1.05) & (g > b * 1.05) & (g > 35)
    yellow_foliage = (r > 60) & (g > 60) & (b < np.minimum(r, g) * 0.8) & (np.abs(r - g) < 50)
    brown_foliage = (r > 40) & (g > 30) & (b < 60) & (r > g) & (g >= b) & (~skin_mask)
    foliage_mask = green_foliage | yellow_foliage | brown_foliage
    foliage_pct = np.sum(foliage_mask) / total_pixels

    # A) Human portrait or face: noticeable skin tone with low/no foliage
    if skin_pct > 0.12 and skin_pct > foliage_pct * 0.8:
        return False, "human_face", "Human face or person detected. Please upload a photo of a plant leaf."

    # B) Non-plant object: virtually no foliage colors detected
    if foliage_pct < 0.12:
        return False, "not_plant", "This photo does not look like a plant leaf. Please upload a clear photo of Tomato, Potato, or Pepper leaves."

    return True, "plant_leaf", "Valid plant leaf detected."


def explain_with_gemini(disease, cause, solution):
    """Generate a simple Hindi + English explanation using Gemini.
    Returns a fallback message instead of crashing if Gemini is
    unavailable, misconfigured, or the request fails."""
    c = get_gemini_client()
    if c is None:
        return "AI explanation unavailable (no API key configured)."

    prompt = f"""
    Explain in simple Hindi and English:
    Disease: {disease}
    Cause: {cause}
    Solution: {solution}
    """

    for m_name in ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-2.5-flash"]:
        try:
            response = c.models.generate_content(
                model=m_name,
                contents=prompt
            )
            return response.text
        except Exception as e:
            print(f"Gemini call failed with model {m_name}: {e}")
    
    return "AI explanation unavailable right now (network or API issue)."


@app.route("/", methods=["GET"])
def home():
    if os.path.exists("index-8.html"):
        return send_file("index-8.html")
    return jsonify({"status": "Plant Disease Detection API is running"})


@app.route("/<path:filename>", methods=["GET"])
def static_files(filename):
    if os.path.exists(filename) and not filename.endswith((".py", ".env", ".h5")):
        return send_file(filename)
    return jsonify({"error": "File not found"}), 404


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Send an image with key 'file'."}), 400

    file = request.files["file"]
    img_bytes = file.read()

    try:
        img_pil, img_array = prepare_image(img_bytes)
    except Exception as e:
        return jsonify({"error": f"Could not process image: {str(e)}"}), 400

    # Validate whether the image is actually a plant leaf
    is_plant, reason, reason_msg = validate_plant_image(img_pil, img_bytes)
    if not is_plant:
        return jsonify({
            "is_plant": False,
            "reason": reason,
            "disease": "Not a plant leaf",
            "raw_class": "Unknown",
            "confidence": 0,
            "cause": reason_msg,
            "solution": "Take a photo of a single crop leaf (Tomato, Potato, or Pepper) with good lighting and the leaf filling the frame.",
            "gemini_explanation": ""
        })

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
        "is_plant": True,
        "disease": display_name,
        "raw_class": predicted_class,
        "confidence": round(confidence * 100, 2),
        "cause": info["cause"],
        "solution": info["solution"],
        "gemini_explanation": gemini_explanation
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)