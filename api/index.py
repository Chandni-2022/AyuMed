import os
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, auth
import requests
import json
import base64
import re
from groq import Groq

# -------------------- Initialization --------------------

load_dotenv()

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(
    __name__,
    template_folder=os.path.join(project_root, 'templates'),
    static_folder=os.path.join(project_root, 'static')
)

app.secret_key = os.urandom(24)

# -------------------- Firebase Setup --------------------

try:
    if not firebase_admin._apps:

        firebase_creds_b64 = os.getenv('FIREBASE_SERVICE_ACCOUNT_KEY_B64')

        if firebase_creds_b64:

            firebase_creds_json = base64.b64decode(firebase_creds_b64).decode('utf-8')
            firebase_creds = json.loads(firebase_creds_json)

            cred = credentials.Certificate(firebase_creds)
            firebase_admin.initialize_app(cred)

            print("Firebase initialized from ENV")

        else:

            current_dir = os.path.dirname(__file__)
            key_path = os.path.join(current_dir, "serviceAccountKey.json")

            cred = credentials.Certificate(key_path)
            firebase_admin.initialize_app(cred)

            print("Firebase initialized from local file")

except Exception as e:
    print("Firebase init error:", e)

# -------------------- API KEYS --------------------

PLANT_ID_API_KEY = os.getenv("PLANT_ID_API_KEY")
PLANT_ID_API_URL = "https://api.plant.id/v2/identify"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

print("Groq key loaded:", bool(GROQ_API_KEY))

groq_client = Groq(api_key=GROQ_API_KEY)

# -------------------- Routes --------------------

@app.route('/')
def index():
    if 'user' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/signin')
def signin():
    if 'user' in session:
        return redirect(url_for('dashboard'))
    return render_template('signin.html')

@app.route('/signup')
def signup():
    if 'user' in session:
        return redirect(url_for('dashboard'))
    return render_template('signup.html')

@app.route('/dashboard')
def dashboard():

    if 'user' not in session:
        return redirect(url_for('signin'))

    return render_template(
        'dashboard.html',
        user_email=session['user']['email']
    )

@app.route('/signout')
def signout():
    session.pop('user', None)
    return redirect(url_for('index'))

# -------------------- Authentication --------------------

@app.route('/session-login', methods=['POST'])
def session_login():

    try:

        id_token = request.json['idToken']
        decoded_token = auth.verify_id_token(id_token)

        session['user'] = {
            'uid': decoded_token['uid'],
            'email': decoded_token.get('email', '')
        }

        return jsonify({"status": "success"}), 200

    except Exception:
        return jsonify({"error": "Failed to authenticate"}), 401

# -------------------- Identify Plant --------------------

@app.route('/identify', methods=['POST'])
def identify():

    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    if 'image' not in request.files:
        return jsonify({'error': 'No image file.'}), 400

    try:

        suggestions = get_suggestions_from_plant_id(request.files['image'])

        if not suggestions:
            return jsonify({'error': "Could not identify the plant."}), 404

        plant_name = suggestions[0]['plant_name']

        description = get_description_from_groq(plant_name)

        return jsonify({
            'suggestions': suggestions,
            'description': description
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# -------------------- Chat API --------------------

@app.route('/chat', methods=['POST'])
def chat():

    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    data = request.json

    question = data.get('question')
    plant_name = data.get('plant_name')

    if not question or not plant_name:
        return jsonify({'error': 'Missing question or plant context.'}), 400

    try:

        prompt = f"""
You are a medicinal plant expert.

Plant: {plant_name}

User Question: {question}

Give a clear answer.
"""

        response = groq_client.chat.completions.create(

            model="llama-3.1-8b-instant",

            temperature=0.4,

            messages=[
                {"role": "system", "content": "You are a botanist expert."},
                {"role": "user", "content": prompt}
            ]
        )

        answer = response.choices[0].message.content.strip()

        return jsonify({'answer': answer})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# -------------------- Plant.id --------------------

def get_suggestions_from_plant_id(image_file):

    files = {
        'images': (
            image_file.filename,
            image_file.read(),
            image_file.mimetype
        )
    }

    headers = {
        'Api-Key': PLANT_ID_API_KEY
    }

    response = requests.post(
        PLANT_ID_API_URL,
        files=files,
        headers=headers
    )

    response.raise_for_status()

    data = response.json()

    if data.get('suggestions'):

        return [
            {
                'plant_name': s['plant_name'],
                'probability': s['probability']
            }
            for s in data['suggestions']
        ]

    return None

# -------------------- CLEAN JSON --------------------

def clean_ai_json(text):

    try:

        text = text.strip()

        # remove markdown
        text = text.replace("```json", "").replace("```", "")

        # find first { and last }
        start = text.find("{")
        end = text.rfind("}") + 1

        json_text = text[start:end]

        return json.loads(json_text)

    except Exception as e:
        print("JSON CLEAN ERROR:", e)
        return None

# -------------------- AI DESCRIPTION --------------------

def get_description_from_groq(plant_name):

    prompt = f"""
Return ONLY valid JSON. No explanation.

Plant: {plant_name}

Format exactly like this:

{{
"medicinal_uses": "...",
"how_to_grow": "...",
"warnings": "...",
"home_remedies": [
"Remedy 1",
"Remedy 2",
"Remedy 3",
"Remedy 4",
"Remedy 5",
"Remedy 6",
"Remedy 7",
"Remedy 8"
]
}}
"""

    try:

        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            temperature=0.2,
            messages=[
                {"role": "system", "content": "Return only pure JSON."},
                {"role": "user", "content": prompt}
            ]
        )

        text = response.choices[0].message.content.strip()

        print("AI RAW:", text)

        data = clean_ai_json(text)

        if not data:
            raise Exception("JSON parse failed")

        if isinstance(data.get("home_remedies"), list):
            data["home_remedies"] = "\n".join(
                [f"• {r}" for r in data["home_remedies"]]
            )

        return data

    except Exception as e:

        print("Groq error:", e)

        return {
            "medicinal_uses": "Medicinal information unavailable.",
            "how_to_grow": "Growing information unavailable.",
            "warnings": "Warning information unavailable.",
            "home_remedies": "Home remedy information unavailable."
        }

# -------------------- Run --------------------

if __name__ == '__main__':
    app.run(debug=True)