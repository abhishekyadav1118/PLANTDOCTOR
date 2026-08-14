🌱 PlantDoctor

AI-Powered Plant Disease Detection & Solution System

«PlantDoctor is an AI-powered web application that detects plant diseases from leaf images using Deep Learning and provides disease-related information and solutions.»

---

✨ Overview

PlantDoctor combines Computer Vision, Deep Learning, Flask, and Generative AI to create an easy-to-use plant disease detection system.

Users can upload or capture a plant-leaf image, receive a predicted disease with a confidence score, and get a natural-language explanation with relevant information and solutions.

The project was developed as a College Mini Project with a focus on practical AI/ML implementation.

---

🚀 Key Features

- 🌿 Plant Disease Detection from leaf images
- 🧠 MobileNetV2 + Transfer Learning based model
- 📷 Upload images or capture photos directly from a phone
- 📊 Prediction confidence score
- 💡 Disease cause and treatment information
- 🤖 Google Gemini AI explanations
- 🇮🇳 English & Hindi support
- 🌙 Dark / Light mode
- 📋 Recent prediction history
- ⚠️ Low-confidence warning
- 📱 Same-WiFi mobile access
- 🔌 Flask REST API for frontend-backend communication

---

🧠 AI / ML Pipeline

        🌿 Leaf Image
             │
             ▼
     Image Preprocessing
        (224 × 224)
             │
             ▼
       MobileNetV2
      Transfer Learning
             │
             ▼
      Disease Prediction
             │
       ┌─────┴─────┐
       ▼           ▼
 Disease Info   Confidence
       │
       ▼
   Gemini AI
       │
       ▼
 Hindi / English
   Explanation

The model was trained using the PlantVillage dataset, containing approximately 20,600 leaf images across 15 classes. Training was performed in Google Colab using a T4 GPU.

---

🛠️ Tech Stack

🤖 Machine Learning & Deep Learning

Technology| Purpose
TensorFlow| Deep Learning framework
Keras| Model development & training
MobileNetV2| Pre-trained CNN base model
Transfer Learning| Model adaptation
ImageDataGenerator| Data augmentation
NumPy| Numerical operations

🔧 Backend

Technology| Purpose
Python| Backend programming
Flask| REST API
Flask-CORS| Frontend ↔ Backend communication
Pillow| Image processing
JSON| Disease information & solutions
Socket| Local IP detection

The backend exposes a "/predict" endpoint that accepts an image and returns disease, confidence, cause, and solution information.

🎨 Frontend

- HTML
- CSS
- JavaScript
- Fetch API
- FormData
- Drag & Drop Upload
- Camera Capture
- Inline SVG
- English/Hindi UI
- Dark/Light Mode

🤖 Generative AI

- Google Gemini API
- "google-genai"
- Google AI Studio
- "gemini-flash-latest"
- Environment variables for API-key security
- Error handling with fallback responses

---

📊 Dataset

PlantVillage Dataset

- 🌿 Approximately 20,600 leaf images
- 🏷️ 15 plant disease classes
- 🍅 Tomato
- 🥔 Potato
- 🌶️ Pepper
- 🌱 Healthy & diseased categories

Dataset access and downloading were handled through Kaggle and Kaggle API inside Google Colab.

---

🏗️ Project Structure

PlantDoctor/
│
├── app.py
├── index.html
├── plant_model.h5
├── labels.json
├── solutions.json
├── list_models.py
├── requirements.txt
└── README.md

«File names may vary slightly depending on the local project version.»

---

⚙️ Installation & Setup

1. Clone the Repository

git clone https://github.com/abhishekyadav1118/PlantDoctor.git
cd PlantDoctor

2. Create a Virtual Environment

python -m venv venv

3. Activate Environment

Windows PowerShell:

venv\Scripts\Activate.ps1

4. Install Dependencies

pip install -r requirements.txt

5. Configure Gemini API

Set your Gemini API key as an environment variable instead of placing it directly inside the source code.

PowerShell:

$env:GEMINI_API_KEY="YOUR_API_KEY"

The project uses environment variables to keep the API key out of the source code.

6. Run the Backend

python app.py

7. Open the Application

http://127.0.0.1:5000/

---

📱 Mobile Testing

PlantDoctor can also be tested from a phone connected to the same Wi-Fi network as the laptop.

The backend can listen on all network interfaces and determine the laptop's local IP address for easier team/mobile testing.

---

🔄 How It Works

1. User uploads / captures leaf image
                ↓
2. Image preprocessing
                ↓
3. MobileNetV2 model prediction
                ↓
4. Disease + confidence generated
                ↓
5. Disease information retrieved
                ↓
6. Gemini generates explanation
                ↓
7. Result displayed in English / Hindi

---

🔐 Security

API credentials should never be hardcoded inside the source code.

PlantDoctor uses environment variables for the Gemini API key, helping keep sensitive credentials outside the application code.

---

🧪 Testing & Debugging

The project includes:

- Localhost testing
- Same-WiFi phone testing
- Browser debugging
- CORS troubleshooting
- Iterative bug fixing
- Mobile camera/upload testing

---

🔮 Future Enhancements

- 📱 Dedicated Android application
- 🌍 More plant species and disease classes
- ☁️ Cloud deployment
- 📈 Advanced disease analytics
- 🗃️ Persistent prediction history
- 🌐 Additional language support
- 🧠 Further model improvements

---

👨‍💻 Team

PlantDoctor — College Mini Project

A collaborative project developed by a B.Tech Computer Science Engineering team.

---

⭐ Project Highlights

Deep Learning        → MobileNetV2 + Transfer Learning
Backend              → Python + Flask
Frontend             → HTML + CSS + JavaScript
Generative AI        → Google Gemini
Dataset              → PlantVillage
Training             → Google Colab + T4 GPU
API                  → Flask REST API
Mobile Testing       → Same-WiFi Network

---

🌱 Built with AI • Deep Learning • Python • Passion

⭐ If you find PlantDoctor useful, consider starring the repository.
