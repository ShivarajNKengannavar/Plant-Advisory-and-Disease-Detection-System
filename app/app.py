# ------------------ Import Libraries ------------------
from flask import Flask, render_template, request, jsonify
from markupsafe import Markup
import numpy as np
import pandas as pd
import requests
import config
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
import joblib
import warnings
from deep_translator import GoogleTranslator
import google.generativeai as genai  # ✅ Gemini SDK

# ------------------ Load Models ------------------
disease_classes = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Load Plant Disease Model
disease_model_path = 'models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    disease_model.load_state_dict(torch.load(disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()

# Load Crop Recommendation Model
crop_model_path = 'models/crop_recommendation_model.pkl'
crop_recommendation_model = joblib.load(open(crop_model_path, 'rb'))

# ------------------ Gemini API Setup ------------------
genai.configure(api_key=config.gemini_api_key)

# ------------------ Helper Functions ------------------
def weather_fetch(city_name):
    api_key = config.weather_api_key
    url = f"http://api.openweathermap.org/data/2.5/weather?appid={api_key}&q={city_name}"
    try:
        response = requests.get(url).json()
        if response.get("cod") != "404":
            temp = round((response["main"]["temp"] - 273.15), 2)
            humidity = response["main"]["humidity"]
            description = response["weather"][0]["description"].capitalize()
            return temp, humidity, description
    except Exception:
        pass
    return None

def get_location_city():
    try:
        ip = requests.get("https://api64.ipify.org").text
        response = requests.get(f"http://ip-api.com/json/{ip}").json()
        return response.get("city", "")
    except Exception:
        return ""

def predict_image(img, model=disease_model):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img)).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, pred_idx = torch.max(probabilities, dim=0)
        prediction = disease_classes[pred_idx.item()]
    return prediction, confidence.item()

def translate_text(text, target_lang):
    try:
        if target_lang and target_lang != "en":
            return GoogleTranslator(source='en', target=target_lang).translate(text)
        return text
    except Exception as e:
        return f"[Translation error: {str(e)}]"

# ------------------ Chatbot Functions ------------------
def ask_gemini(prompt):
    try:
        model = genai.GenerativeModel("gemini-2.5-pro")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"[Gemini Error] {e}")
        return None

def ask_chatollama(prompt):
    try:
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        })
        return response.json().get('response', '')
    except Exception:
        return None

# ------------------ Flask App ------------------
app = Flask(__name__)

@app.route('/')
def home():
    city = get_location_city()
    weather = weather_fetch(city) if city else None
    return render_template('index.html', title='Harvestify - Home', city=city, weather=weather)

@app.route('/crop-recommend')
def crop_recommend():
    return render_template('crop.html', title='Harvestify - Crop Recommendation')

@app.route('/fertilizer')
def fertilizer_recommendation():
    return render_template('fertilizer.html', title='Harvestify - Fertilizer Suggestion')

@app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    try:
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])

        data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = crop_recommendation_model.predict(data)[0]

        return render_template('crop.html', prediction=prediction, title='Harvestify - Crop Recommendation')
    except Exception as e:
        return render_template('error.html', error=f"An error occurred: {str(e)}")

@app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    try:
        crop_name = str(request.form['cropname'])
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])

        df = pd.read_csv('Data/fertilizer.csv')
        nr, pr, kr = df[df['Crop'] == crop_name][['N', 'P', 'K']].iloc[0]

        n = nr - N
        p = pr - P
        k = kr - K
        temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
        max_key = temp[max(temp)]
        key = f"{max_key}{'High' if eval(max_key.lower()) < 0 else 'low'}"

        recommendation = fertilizer_dic.get(key, "No suggestion found")
        return render_template('fertilizer-result.html', recommendation=Markup(recommendation), title='Harvestify - Fertilizer Suggestion')
    except Exception as e:
        return render_template('error.html', error=f"Fertilizer prediction failed: {str(e)}")

@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Harvestify - Disease Detection'
    if request.method == 'POST':
        file = request.files.get('file')
        lang = request.form.get('language', 'en')

        if not file:
            return render_template('disease.html', title=title, weather_api_key=config.weather_api_key)

        try:
            img = file.read()
            prediction, confidence = predict_image(img)
            info = disease_dic.get(prediction, "No additional information available.")

            translated_pred = translate_text(prediction.replace('_', ' '), lang)
            translated_info = translate_text(info, lang)

            formatted = f"<h4>{translated_pred}<br><small>(Confidence: {confidence * 100:.2f}%)</small></h4><hr>{translated_info}"
            return render_template('disease-result.html', prediction=Markup(formatted), title=title)
        except Exception as e:
            return render_template('error.html', error=f"Prediction failed: {str(e)}")

    return render_template('disease.html', title=title, weather_api_key=config.weather_api_key)

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json(force=True)
        user_input = data.get('question', '').strip()
        print("[User]:", user_input)

        if not user_input:
            return jsonify({'response': "Please ask something."})

        response = ask_gemini(user_input)
        print("[Gemini]:", response)

        if not response:
            response = ask_chatollama(user_input)
        if not response:
            response = "I'm unable to respond at the moment."

        return jsonify({'response': response})
    except Exception as e:
        print("Chatbot Error:", e)
        return jsonify({'response': f"Error: {str(e)}"})

# ------------------ Run ------------------
if __name__ == '__main__':
    app.run(debug=True)

