import os
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, auth
import requests
import json
import base64
import time
from groq import Groq

load_dotenv()

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(
    __name__,
    template_folder=os.path.join(project_root, 'templates'),
    static_folder=os.path.join(project_root, 'static')
)

app.secret_key = os.urandom(24)

# -------------------- Firebase --------------------
try:
    if not firebase_admin._apps:
        firebase_creds_b64 = os.getenv('FIREBASE_SERVICE_ACCOUNT_KEY_B64')
        if firebase_creds_b64:
            cred = credentials.Certificate(json.loads(base64.b64decode(firebase_creds_b64).decode('utf-8')))
            firebase_admin.initialize_app(cred)
            print("Firebase initialized from ENV")
        else:
            cred = credentials.Certificate(os.path.join(os.path.dirname(__file__), "serviceAccountKey.json"))
            firebase_admin.initialize_app(cred)
            print("Firebase initialized from local file")
except Exception as e:
    print("Firebase init error:", e)

# -------------------- API Setup --------------------
PLANT_ID_API_KEY = os.getenv("PLANT_ID_API_KEY")
PLANT_ID_API_URL = "https://api.plant.id/v2/identify"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)
print("Groq key loaded:", bool(GROQ_API_KEY))

CONFIDENCE_THRESHOLD = 0.50
# Use best available Groq models
TEXT_MODEL   = "llama-3.3-70b-versatile"   # best text model on Groq
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# -------------------- Helpers --------------------

def groq_text(messages, temperature=0.3, retries=2):
    """Call Groq text model with retry logic."""
    for attempt in range(retries + 1):
        try:
            resp = groq_client.chat.completions.create(
                model=TEXT_MODEL,
                temperature=temperature,
                max_tokens=1500,
                messages=messages
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < retries:
                time.sleep(1)
            else:
                raise e

def clean_json(text):
    text = text.strip().replace("```json", "").replace("```", "")
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        return json.loads(text[start:end])
    except Exception as e:
        print("JSON parse error:", e)
        return None

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
    return render_template('dashboard.html', user_email=session['user']['email'])

@app.route('/signout')
def signout():
    session.pop('user', None)
    return redirect(url_for('index'))

@app.route('/session-login', methods=['POST'])
def session_login():
    try:
        decoded = auth.verify_id_token(request.json['idToken'])
        session['user'] = {'uid': decoded['uid'], 'email': decoded.get('email', '')}
        return jsonify({"status": "success"}), 200
    except Exception:
        return jsonify({"error": "Failed to authenticate"}), 401

# -------------------- Identify --------------------

@app.route('/identify', methods=['POST'])
def identify():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    if 'image' not in request.files:
        return jsonify({'error': 'No image file.'}), 400

    img = request.files['image']
    img_bytes    = img.read()
    img_mimetype = img.mimetype
    img_filename = img.filename

    used_fallback = False
    suggestions   = None

    try:
        suggestions = _plant_id_identify(img_bytes, img_filename, img_mimetype)
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 429:
            print("Plant.id rate limit — switching to Groq vision")
            used_fallback = True
        else:
            return jsonify({'error': f'Plant.id API error: {str(e)}'}), 500
    except Exception as e:
        print(f"Plant.id error: {e} — switching to Groq vision")
        used_fallback = True

    if used_fallback or not suggestions:
        try:
            suggestions   = _groq_vision_identify(img_bytes, img_mimetype)
            used_fallback = True
        except Exception as e:
            return jsonify({'error': f'All identification methods failed: {str(e)}'}), 500

    if not suggestions:
        return jsonify({'error': 'Could not identify the plant. Try a clearer image.'}), 404

    plant_name   = suggestions[0]['plant_name']
    low_conf     = suggestions[0]['probability'] < CONFIDENCE_THRESHOLD
    description  = get_plant_description(plant_name)
    safety       = get_safety_profile(plant_name)
    growing      = get_growing_guide(plant_name)

    return jsonify({
        'suggestions': suggestions,
        'description': description,
        'safety':      safety,
        'growing':     growing,
        'low_confidence': low_conf,
        'used_fallback':  used_fallback,
    })

# -------------------- Chat --------------------

@app.route('/chat', methods=['POST'])
def chat():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    data       = request.json
    question   = data.get('question', '').strip()
    plant_name = data.get('plant_name', '').strip()
    dosha      = data.get('dosha', '')

    if not question or not plant_name:
        return jsonify({'error': 'Missing question or plant context.'}), 400

    dosha_ctx = f"\nThe user's Ayurvedic dosha is: {dosha}. Factor this into your answer where relevant." if dosha else ""

    prompt = f"""You are an expert Ayurvedic physician and botanist.
Plant in context: {plant_name}{dosha_ctx}

User question: {question}

Give a clear, helpful, medically conservative answer focused on this plant's Ayurvedic and medicinal properties. If unrelated to plants or Ayurveda, politely redirect. End with a one-line safety reminder if the answer involves dosage or internal use."""

    try:
        answer = groq_text([
            {"role": "system", "content": "You are a renowned Ayurvedic physician and botanist. Be helpful, accurate, and medically responsible."},
            {"role": "user", "content": prompt}
        ], temperature=0.4)
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# -------------------- Symptom Search --------------------

@app.route('/symptom-search', methods=['POST'])
def symptom_search():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    symptoms = request.json.get('symptoms', '').strip()
    dosha    = request.json.get('dosha', '')
    if not symptoms:
        return jsonify({'error': 'No symptoms provided.'}), 400

    dosha_line = f'\nThe user\'s Ayurvedic dosha is: {dosha}. Prioritize plants suited to this dosha.' if dosha else ''

    prompt = f"""You are an expert Ayurvedic physician.

Patient symptoms: "{symptoms}"{dosha_line}

Return ONLY valid JSON — no markdown, no explanation.

{{
  "recommendations": [
    {{
      "plant_name": "Withania somnifera",
      "common_name": "Ashwagandha",
      "sanskrit_name": "Ashwagandha",
      "why_it_helps": "Adaptogen that reduces cortisol and balances stress response",
      "how_to_use": "500mg root powder in warm milk at bedtime",
      "dosage": "300-600mg standardised extract daily",
      "duration": "4-8 weeks for best results",
      "safety_note": "Avoid during pregnancy"
    }},
    {{
      "plant_name": "...",
      "common_name": "...",
      "sanskrit_name": "...",
      "why_it_helps": "...",
      "how_to_use": "...",
      "dosage": "...",
      "duration": "...",
      "safety_note": "..."
    }},
    {{
      "plant_name": "...",
      "common_name": "...",
      "sanskrit_name": "...",
      "why_it_helps": "...",
      "how_to_use": "...",
      "dosage": "...",
      "duration": "...",
      "safety_note": "..."
    }}
  ],
  "general_advice": "Brief lifestyle/dietary advice alongside these herbs",
  "avoid_foods": "Foods to avoid with these symptoms according to Ayurveda"
}}"""

    try:
        text   = groq_text([
            {"role": "system", "content": "You are an expert Ayurvedic physician. Return only pure valid JSON."},
            {"role": "user", "content": prompt}
        ])
        result = clean_json(text)
        if not result:
            raise Exception("JSON parse failed")
        return jsonify(result)
    except Exception as e:
        print("Symptom search error:", e)
        return jsonify({'error': str(e)}), 500

# -------------------- Compare Plants --------------------

@app.route('/compare', methods=['POST'])
def compare_plants():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    plants = request.json.get('plants', [])
    if len(plants) < 2:
        return jsonify({'error': 'Provide at least 2 plants.'}), 400
    plants = plants[:4]

    prompt = f"""You are an Ayurvedic expert. Compare: {', '.join(plants)}

Return ONLY valid JSON.

{{
  "plants": [
    {{
      "name": "...",
      "sanskrit_name": "...",
      "primary_use": "...",
      "key_compounds": "...",
      "best_for": "...",
      "avoid_if": "...",
      "potency": "Low/Medium/High",
      "taste": "...",
      "dosha_balance": "Vata/Pitta/Kapha",
      "preparation": "...",
      "onset_time": "..."
    }}
  ],
  "best_for_summary": "...",
  "combination_advice": "...",
  "winner_for_beginners": "plant name"
}}

One entry per plant, same order as input."""

    try:
        text   = groq_text([
            {"role": "system", "content": "Return only pure valid JSON. No preamble."},
            {"role": "user", "content": prompt}
        ])
        result = clean_json(text)
        if not result:
            raise Exception("JSON parse failed")
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# -------------------- Drug Interaction Checker --------------------

@app.route('/drug-check', methods=['POST'])
def drug_check():
    """NEW: Check if a medication interacts with an identified plant."""
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    plant_name = request.json.get('plant_name', '').strip()
    medication = request.json.get('medication', '').strip()
    if not plant_name or not medication:
        return jsonify({'error': 'Missing plant or medication name.'}), 400

    prompt = f"""You are a pharmacognosy and drug interaction expert.

Plant: {plant_name}
Medication: {medication}

Analyze potential interactions between this medicinal plant and the medication.

Return ONLY valid JSON.

{{
  "interaction_level": "None/Minor/Moderate/Major/Unknown",
  "interaction_score": <integer 0-10, where 0=no interaction, 10=dangerous>,
  "summary": "One sentence summary of the interaction",
  "mechanism": "How the interaction occurs biochemically",
  "clinical_effects": "What the patient might experience",
  "recommendation": "Safe to use together / Use with caution / Avoid combination / Consult doctor",
  "timing_advice": "e.g. separate by 2 hours if minor interaction",
  "evidence_level": "Strong/Moderate/Limited/Theoretical",
  "references": "General reference to type of studies (not fake citations)"
}}"""

    try:
        text   = groq_text([
            {"role": "system", "content": "You are a pharmacognosy expert. Be medically conservative. Return only pure valid JSON."},
            {"role": "user", "content": prompt}
        ], temperature=0.2)
        result = clean_json(text)
        if not result:
            raise Exception("JSON parse failed")
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# -------------------- Dosha Quiz --------------------

@app.route('/dosha-result', methods=['POST'])
def dosha_result():
    """NEW: Process dosha quiz answers and return dosha profile."""
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    answers = request.json.get('answers', {})

    prompt = f"""You are an Ayurvedic Vaidya (physician).

Based on these quiz answers, determine the user's Prakriti (body constitution):
{json.dumps(answers, indent=2)}

Return ONLY valid JSON.

{{
  "primary_dosha": "Vata/Pitta/Kapha",
  "secondary_dosha": "Vata/Pitta/Kapha/None",
  "dosha_breakdown": {{"vata": 35, "pitta": 45, "kapha": 20}},
  "description": "2-3 sentence description of this constitution",
  "strengths": ["...", "..."],
  "imbalance_signs": ["...", "..."],
  "recommended_plants": ["plant1", "plant2", "plant3"],
  "foods_to_favor": "...",
  "foods_to_avoid": "...",
  "lifestyle_tips": "..."
}}"""

    try:
        text   = groq_text([
            {"role": "system", "content": "You are an expert Ayurvedic Vaidya. Return only pure valid JSON."},
            {"role": "user", "content": prompt}
        ], temperature=0.3)
        result = clean_json(text)
        if not result:
            raise Exception("JSON parse failed")
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# -------------------- Internal Functions --------------------

def _plant_id_identify(img_bytes, filename, mimetype):
    resp = requests.post(
        PLANT_ID_API_URL,
        files={'images': (filename, img_bytes, mimetype)},
        headers={'Api-Key': PLANT_ID_API_KEY}
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get('suggestions'):
        return [{'plant_name': s['plant_name'], 'probability': s['probability']} for s in data['suggestions']]
    return None

def _groq_vision_identify(img_bytes, mimetype):
    b64 = base64.standard_b64encode(img_bytes).decode('utf-8')
    prompt = """You are an expert botanist. Identify the plant in this image.

Return ONLY valid JSON:
{
  "suggestions": [
    {"plant_name": "Scientific name", "common_name": "Common name", "probability": 0.85, "features": "Key visual features"},
    {"plant_name": "Second possibility", "common_name": "Common name", "probability": 0.10, "features": "Why this matches"},
    {"plant_name": "Third possibility", "common_name": "Common name", "probability": 0.05, "features": "Brief reason"}
  ]
}

Be conservative with probabilities. If the image is blurry or unclear, lower all probabilities."""

    resp = groq_client.chat.completions.create(
        model=VISION_MODEL,
        temperature=0.2,
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mimetype};base64,{b64}"}},
            {"type": "text", "text": prompt}
        ]}]
    )
    data = clean_json(resp.choices[0].message.content)
    if not data or not data.get('suggestions'):
        raise Exception("Groq vision returned no suggestions")
    return [{'plant_name': s['plant_name'], 'probability': float(s.get('probability', 0.5))} for s in data['suggestions']]

def get_plant_description(plant_name):
    prompt = f"""You are a senior Ayurvedic physician and botanist with 30 years of experience.

Provide comprehensive information about: {plant_name}

This plant is well-documented in Ayurvedic medicine. Provide rich, accurate, detailed information.

Return ONLY valid JSON — no markdown fences, no extra text.

{{
  "sanskrit_name": "Sanskrit/traditional name if known",
  "common_names": "Common names in English and Indian languages",
  "medicinal_uses": "Detailed paragraph: list the primary medicinal uses, active compounds, and which body systems it supports. Include traditional Ayurvedic uses and modern research findings.",
  "how_to_grow": "Detailed growing guide: soil type, sunlight, watering, temperature range, propagation method, container vs ground growing, harvest time.",
  "warnings": "Important safety warnings, overdose risks, who should avoid this plant, known side effects.",
  "home_remedies": [
    "Remedy 1: Name — Ingredients and preparation method — How to use — For what condition",
    "Remedy 2: Name — Ingredients and preparation method — How to use — For what condition",
    "Remedy 3: Name — Ingredients and preparation method — How to use — For what condition",
    "Remedy 4: Name — Ingredients and preparation method — How to use — For what condition",
    "Remedy 5: Name — Ingredients and preparation method — How to use — For what condition",
    "Remedy 6: Name — Ingredients and preparation method — How to use — For what condition"
  ],
  "active_compounds": "Key phytochemicals and their primary actions",
  "parts_used": "Which parts of the plant are used medicinally",
  "taste_in_ayurveda": "Rasa (taste) and Guna (qualities) in Ayurvedic terms"
}}"""

    try:
        text = groq_text([
            {"role": "system", "content": "You are a senior Ayurvedic physician. Always provide complete, detailed, accurate information. Never say information is unavailable — use your full medical and botanical knowledge. Return only pure valid JSON."},
            {"role": "user", "content": prompt}
        ], temperature=0.2)
        print("Description RAW:", text[:200])
        data = clean_json(text)
        if not data:
            raise Exception("JSON parse failed")
        if isinstance(data.get("home_remedies"), list):
            data["home_remedies"] = "\n\n".join([f"**{i+1}.** {r}" for i, r in enumerate(data["home_remedies"])])
        return data
    except Exception as e:
        print("Description error:", e)
        # Retry with simpler prompt
        return _get_description_simple(plant_name)

def _get_description_simple(plant_name):
    """Simplified fallback prompt for stubborn models."""
    prompt = f"Describe the medicinal plant {plant_name} in JSON with keys: medicinal_uses, how_to_grow, warnings, home_remedies (array of 6 strings), active_compounds, parts_used, taste_in_ayurveda. Be detailed and specific. Return only JSON."
    try:
        text = groq_text([
            {"role": "system", "content": "Expert botanist. Return only valid JSON. Never say unavailable."},
            {"role": "user", "content": prompt}
        ], temperature=0.1)
        data = clean_json(text)
        if data and isinstance(data.get("home_remedies"), list):
            data["home_remedies"] = "\n\n".join([f"**{i+1}.** {r}" for i, r in enumerate(data["home_remedies"])])
        return data or {"medicinal_uses": f"{plant_name} is a medicinal plant used in traditional medicine.", "how_to_grow": "Grows in tropical/subtropical climates.", "warnings": "Consult a practitioner before use.", "home_remedies": "Consult an Ayurvedic practitioner for specific remedies.", "active_compounds": "Various phytochemicals.", "parts_used": "Leaves, roots, and seeds.", "taste_in_ayurveda": "Consult Ayurvedic texts."}
    except:
        return {"medicinal_uses": f"{plant_name} is recognized in traditional Ayurvedic medicine.", "how_to_grow": "Thrives in warm climates with well-drained soil.", "warnings": "Always consult a qualified practitioner before use.", "home_remedies": "Traditional preparations vary — consult an Ayurvedic physician.", "active_compounds": "Rich in bioactive phytochemicals.", "parts_used": "Multiple plant parts used therapeutically.", "taste_in_ayurveda": "Properties documented in classical Ayurvedic texts."}

def get_safety_profile(plant_name):
    prompt = f"""You are a pharmacognosy expert. Assess the complete safety profile of: {plant_name}

Return ONLY valid JSON:

{{
  "overall_safety": "Safe",
  "safety_score": 8,
  "toxicity_level": "Low",
  "drug_interactions": "May interact with sedatives and thyroid medications",
  "pregnancy_safety": "Caution",
  "children_safety": "Caution",
  "max_dosage_note": "Standard therapeutic dose and form",
  "contraindications": ["Autoimmune conditions", "Pregnancy"],
  "safe_parts": "Root, leaves",
  "toxic_parts": "None known",
  "first_aid": "Discontinue use and consult a doctor if adverse effects occur"
}}

overall_safety must be exactly: Safe, Caution, or Toxic
safety_score must be integer 1-10"""

    try:
        text   = groq_text([
            {"role": "system", "content": "Pharmacognosy expert. Be accurate and conservative. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ], temperature=0.1)
        result = clean_json(text)
        if not result:
            raise Exception("parse failed")
        return result
    except Exception as e:
        print("Safety error:", e)
        return {"overall_safety": "Caution", "safety_score": 6, "toxicity_level": "Unknown", "drug_interactions": "Consult a pharmacist before combining with medications.", "pregnancy_safety": "Caution", "children_safety": "Caution", "max_dosage_note": "Follow practitioner guidance on dosage.", "contraindications": ["Consult a doctor before use"], "safe_parts": "Traditional preparation recommended", "toxic_parts": "Unknown — use only prepared forms", "first_aid": "Seek medical attention if adverse reaction occurs."}

def get_growing_guide(plant_name):
    prompt = f"""You are a master horticulturist and Ayurvedic garden specialist.

Provide a seasonal and regional growing guide for: {plant_name}

Return ONLY valid JSON:

{{
  "climate_zones": "Which climate zones this plant thrives in (tropical, subtropical, temperate, etc.)",
  "best_planting_season": "Best months to plant (e.g. March-May for temperate zones)",
  "harvest_season": "When to harvest for maximum potency",
  "soil_ph": "Ideal soil pH range",
  "sunlight": "Full sun / Partial shade / Full shade",
  "watering": "Watering frequency and method",
  "propagation": "Seeds / Cuttings / Division — step by step",
  "container_friendly": true,
  "companion_plants": "Plants that grow well alongside it",
  "pests_diseases": "Common problems and organic solutions",
  "india_growing_regions": "Specific Indian states/regions where it grows naturally",
  "harvest_tip": "When and how to harvest for medicinal potency"
}}"""

    try:
        text   = groq_text([
            {"role": "system", "content": "Master horticulturist. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ], temperature=0.2)
        result = clean_json(text)
        return result or {}
    except Exception as e:
        print("Growing guide error:", e)
        return {}

# -------------------- Run --------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)