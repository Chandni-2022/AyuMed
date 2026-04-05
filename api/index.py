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
GROQ_API_KEY     = os.getenv("GROQ_API_KEY")
groq_client      = Groq(api_key=GROQ_API_KEY)
print("Groq key loaded:", bool(GROQ_API_KEY))

CONFIDENCE_THRESHOLD = 0.50
TEXT_MODEL   = "llama-3.3-70b-versatile"
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# -------------------- Safety classification helpers --------------------

def is_toxic_plant(safety: dict) -> bool:
    """Return True if plant must be treated as dangerous."""
    if not safety:
        return False
    level = (safety.get("overall_safety") or "").strip().lower()
    score = int(safety.get("safety_score") or 10)
    tox   = (safety.get("toxicity_level") or "").strip().lower()
    return (
        level == "toxic"
        or score <= 3
        or tox in ("high", "severe", "very high")
    )

def is_caution_plant(safety: dict) -> bool:
    """Return True if plant needs warnings but remedies can still be shown."""
    if not safety:
        return False
    level = (safety.get("overall_safety") or "").strip().lower()
    score = int(safety.get("safety_score") or 10)
    preg  = (safety.get("pregnancy_safety") or "").strip().lower()
    tox   = (safety.get("toxicity_level") or "").strip().lower()
    return (
        level == "caution"
        or score <= 6
        or preg in ("avoid",)
        or tox in ("moderate",)
    )

def safety_gate(safety: dict) -> dict:
    """
    Returns a dict describing what content is allowed for this plant.
    Keys: show_remedies, show_drug_checker, show_chatbot_remedies,
          show_growing_harvest, danger_level, danger_message
    """
    if is_toxic_plant(safety):
        return {
            "show_remedies":          False,
            "show_drug_checker":      False,
            "show_chatbot_remedies":  False,
            "show_growing_harvest":   False,
            "danger_level":           "toxic",
            "danger_message": (
                "DANGER — This plant is highly toxic and must NOT be used medicinally or consumed. "
                "Home remedies, drug interaction checks, and harvest guidance have been disabled. "
                "If you or someone has been exposed to this plant, seek emergency medical attention immediately."
            )
        }
    if is_caution_plant(safety):
        return {
            "show_remedies":          True,
            "show_drug_checker":      True,
            "show_chatbot_remedies":  True,
            "show_growing_harvest":   True,
            "danger_level":           "caution",
            "danger_message": (
                "CAUTION — This plant requires careful handling. "
                "Remedies are shown for educational purposes only. "
                "Do NOT use without consulting a qualified Ayurvedic practitioner or physician. "
                + (
                    "Avoid completely during pregnancy. "
                    if (safety.get("pregnancy_safety") or "").lower() in ("avoid",) else ""
                )
            )
        }
    return {
        "show_remedies":          True,
        "show_drug_checker":      True,
        "show_chatbot_remedies":  True,
        "show_growing_harvest":   True,
        "danger_level":           "safe",
        "danger_message":         ""
    }

# -------------------- Helpers --------------------

def groq_text(messages, temperature=0.3, retries=2):
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

    img          = request.files['image']
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

    plant_name = suggestions[0]['plant_name']
    low_conf   = suggestions[0]['probability'] < CONFIDENCE_THRESHOLD

    # Always get safety FIRST — it gates all other content
    safety = get_safety_profile(plant_name)
    gate   = safety_gate(safety)

    # Only fetch description/growing if plant is not fully toxic
    if gate["danger_level"] == "toxic":
        # For toxic plants: fetch description (educational info only),
        # but home_remedies will be stripped server-side
        description = get_plant_description_safe(plant_name, allow_remedies=False)
        growing     = get_growing_guide(plant_name, harvest_blocked=True)
    else:
        description = get_plant_description_safe(plant_name, allow_remedies=True)
        growing     = get_growing_guide(plant_name, harvest_blocked=False)

    # Get Ayurvedic classification — core to paper title alignment
    ayurvedic_info = get_ayurvedic_classification(plant_name)

    return jsonify({
        'suggestions':    suggestions,
        'description':    description,
        'safety':         safety,
        'growing':        growing,
        'gate':           gate,
        'ayurvedic_info': ayurvedic_info,
        'low_confidence': low_conf,
        'used_fallback':  used_fallback,
    })

# -------------------- Chat --------------------

@app.route('/chat', methods=['POST'])
def chat():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    data           = request.json
    question       = data.get('question', '').strip()
    plant_name     = data.get('plant_name', '').strip()
    dosha          = data.get('dosha', '')
    danger_level   = data.get('danger_level', 'safe')   # passed from frontend

    if not question or not plant_name:
        return jsonify({'error': 'Missing question or plant context.'}), 400

    # EDGE CASE: toxic plant — refuse all remedy/dosage questions
    if danger_level == "toxic":
        toxic_keywords = [
            'remedy', 'remedies', 'use', 'dose', 'dosage', 'eat', 'consume',
            'drink', 'apply', 'prepare', 'cook', 'harvest', 'treatment', 'cure',
            'medicine', 'medicinal', 'take', 'ingest', 'rub', 'extract'
        ]
        q_lower = question.lower()
        if any(kw in q_lower for kw in toxic_keywords):
            return jsonify({'answer': (
                f"⚠️ **Safety Alert:** *{plant_name}* is classified as a **highly toxic plant** "
                f"and cannot be used medicinally, consumed, or applied to the body in any form. "
                f"I am unable to provide dosage, remedy, or preparation guidance for this plant.\n\n"
                f"If you or someone has been exposed to this plant, please seek **emergency medical attention** immediately."
            )}), 200

    dosha_ctx = f"\nThe user's Ayurvedic dosha is: {dosha}. Factor this into your answer where relevant." if dosha else ""
    caution_ctx = "\nIMPORTANT: This plant requires caution. Do not recommend specific dosages — always advise consulting a practitioner." if danger_level == "caution" else ""

    prompt = f"""You are an expert Ayurvedic physician and botanist.
Plant in context: {plant_name}{dosha_ctx}{caution_ctx}

User question: {question}

Give a clear, helpful, medically conservative answer. If the question involves dosage or internal use, always end with a safety reminder to consult a practitioner."""

    try:
        answer = groq_text([
            {"role": "system", "content": "You are a renowned Ayurvedic physician and botanist. Be helpful, accurate, and medically responsible. Never recommend toxic plants for internal use."},
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

CRITICAL SAFETY RULE: You must ONLY recommend plants that are:
1. Well-established in Ayurvedic or traditional medicine
2. Considered Safe or at most Caution level — NEVER recommend toxic, poisonous, or dangerous plants
3. Appropriate for general public use with standard precautions

Return ONLY valid JSON — no markdown, no explanation.

{{
  "recommendations": [
    {{
      "plant_name": "Withania somnifera",
      "common_name": "Ashwagandha",
      "sanskrit_name": "Ashwagandha",
      "safety_level": "Safe",
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
      "safety_level": "Safe or Caution",
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
      "safety_level": "Safe or Caution",
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
            {"role": "system", "content": "You are an expert Ayurvedic physician. Only recommend safe, well-established medicinal plants. NEVER recommend toxic or dangerous plants. Return only pure valid JSON."},
            {"role": "user", "content": prompt}
        ])
        result = clean_json(text)
        if not result:
            raise Exception("JSON parse failed")

        # EDGE CASE: Backend filter — remove any recommendation flagged as Toxic
        recs = result.get("recommendations", [])
        filtered = [r for r in recs if (r.get("safety_level") or "").lower() not in ("toxic", "dangerous", "poison")]
        result["recommendations"] = filtered

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

    # Check safety of each plant before comparing
    safety_flags = {}
    for p in plants:
        s = get_safety_profile(p)
        safety_flags[p] = {
            "is_toxic":   is_toxic_plant(s),
            "is_caution": is_caution_plant(s),
            "score":      s.get("safety_score", 10),
            "level":      s.get("overall_safety", "Unknown")
        }

    prompt = f"""You are an Ayurvedic expert. Compare these medicinal plants: {', '.join(plants)}

IMPORTANT: For any plant that is toxic or poisonous, clearly state "TOXIC — NOT FOR MEDICINAL USE" in the primary_use field.

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
      "onset_time": "...",
      "safety_warning": "None / Brief caution note / TOXIC - DO NOT USE"
    }}
  ],
  "best_for_summary": "...",
  "combination_advice": "...",
  "winner_for_beginners": "plant name (only safe plants)"
}}

One entry per plant, same order as input."""

    try:
        text   = groq_text([
            {"role": "system", "content": "Return only pure valid JSON. Always flag toxic plants clearly."},
            {"role": "user", "content": prompt}
        ])
        result = clean_json(text)
        if not result:
            raise Exception("JSON parse failed")

        # Attach safety flags to each plant entry
        for plant_entry in result.get("plants", []):
            name = plant_entry.get("name", "")
            flag = safety_flags.get(name) or {}
            plant_entry["is_toxic"]   = flag.get("is_toxic", False)
            plant_entry["is_caution"] = flag.get("is_caution", False)
            plant_entry["safety_score"] = flag.get("score", 10)
            plant_entry["safety_level"] = flag.get("level", "Unknown")

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# -------------------- Drug Interaction Checker --------------------

@app.route('/drug-check', methods=['POST'])
def drug_check():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    plant_name   = request.json.get('plant_name', '').strip()
    medication   = request.json.get('medication', '').strip()
    danger_level = request.json.get('danger_level', 'safe')

    if not plant_name or not medication:
        return jsonify({'error': 'Missing plant or medication name.'}), 400

    # EDGE CASE: Block drug check on toxic plants entirely
    if danger_level == "toxic":
        return jsonify({
            'error': f'{plant_name} is a toxic plant. Drug interaction checking is disabled for dangerous plants. Do not attempt to use this plant with any medication.'
        }), 400

    prompt = f"""You are a pharmacognosy and drug interaction expert.

Plant: {plant_name}
Medication: {medication}

Analyze potential interactions. Be medically conservative.

Return ONLY valid JSON.

{{
  "interaction_level": "None/Minor/Moderate/Major/Unknown",
  "interaction_score": <integer 0-10, where 0=no interaction, 10=dangerous>,
  "summary": "One sentence summary of the interaction",
  "mechanism": "How the interaction occurs biochemically",
  "clinical_effects": "What the patient might experience",
  "recommendation": "Safe to use together / Use with caution / Avoid combination / Consult doctor",
  "timing_advice": "e.g. separate by 2 hours if minor interaction",
  "evidence_level": "Strong/Moderate/Limited/Theoretical"
}}"""

    try:
        text   = groq_text([
            {"role": "system", "content": "Pharmacognosy expert. Be medically conservative. Return only pure valid JSON."},
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
}}

IMPORTANT: Only recommend plants that are well-established, safe, and commonly used in Ayurveda."""

    try:
        text   = groq_text([
            {"role": "system", "content": "Expert Ayurvedic Vaidya. Only recommend safe, established medicinal plants. Return only pure valid JSON."},
            {"role": "user", "content": prompt}
        ], temperature=0.3)
        result = clean_json(text)
        if not result:
            raise Exception("JSON parse failed")
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# -------------------- Internal: Plant identification --------------------

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
Be conservative with probabilities. Lower all probabilities if the image is blurry or unclear."""

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

# -------------------- Internal: Plant description (safety-gated) --------------------

def get_plant_description_safe(plant_name: str, allow_remedies: bool) -> dict:
    """
    Fetch plant description. If allow_remedies=False (toxic plant),
    home_remedies field is forced to None and medicinal_uses is educational only.
    """
    if not allow_remedies:
        # For toxic plants: educational botanical info only, no remedies
        prompt = f"""You are a botanist providing EDUCATIONAL INFORMATION ONLY about a toxic plant.

Plant: {plant_name}

This plant is TOXIC. Provide ONLY botanical and scientific information.
Do NOT provide any medicinal uses, dosages, home remedies, or preparation methods.

Return ONLY valid JSON:
{{
  "sanskrit_name": "Traditional name if any",
  "common_names": "Common names",
  "medicinal_uses": "This plant is classified as toxic and is NOT suitable for medicinal use. Provide only its botanical classification and the scientific reason for its toxicity.",
  "how_to_grow": "Botanical growing information for academic/ornamental study only. Note that harvesting any part for consumption is dangerous.",
  "warnings": "Comprehensive toxicity warning: explain WHY this plant is dangerous, what toxins it contains, symptoms of exposure, and emergency response.",
  "home_remedies": null,
  "active_compounds": "List of toxic compounds present in this plant",
  "parts_used": "All parts are toxic — do not handle without protection",
  "taste_in_ayurveda": "Not applicable — this plant is not used in Ayurvedic medicine due to its toxicity"
}}"""
    else:
        prompt = f"""You are a senior Ayurvedic physician and botanist with 30 years of experience.

Provide comprehensive information about: {plant_name}

Return ONLY valid JSON — no markdown fences, no extra text.

{{
  "sanskrit_name": "Sanskrit/traditional name if known",
  "common_names": "Common names in English and Indian languages",
  "medicinal_uses": "Detailed paragraph covering primary medicinal uses, active compounds, and body systems supported. Include traditional Ayurvedic uses and modern research.",
  "how_to_grow": "Detailed growing guide: soil type, sunlight, watering, temperature, propagation, harvest time.",
  "warnings": "Safety warnings, overdose risks, who should avoid this plant, known side effects.",
  "home_remedies": [
    "Remedy 1: Name — Ingredients and preparation — How to use — For what condition",
    "Remedy 2: Name — Ingredients and preparation — How to use — For what condition",
    "Remedy 3: Name — Ingredients and preparation — How to use — For what condition",
    "Remedy 4: Name — Ingredients and preparation — How to use — For what condition",
    "Remedy 5: Name — Ingredients and preparation — How to use — For what condition",
    "Remedy 6: Name — Ingredients and preparation — How to use — For what condition"
  ],
  "active_compounds": "Key phytochemicals and their primary actions",
  "parts_used": "Which parts of the plant are used medicinally",
  "taste_in_ayurveda": "Rasa (taste) and Guna (qualities) in Ayurvedic terms"
}}"""

    try:
        text = groq_text([
            {"role": "system", "content": "You are a senior Ayurvedic physician. Provide complete, accurate information. Return only pure valid JSON. Never say information is unavailable."},
            {"role": "user", "content": prompt}
        ], temperature=0.2)
        print("Description RAW:", text[:200])
        data = clean_json(text)
        if not data:
            raise Exception("JSON parse failed")

        # Server-side safety gate on home_remedies
        if not allow_remedies:
            data["home_remedies"] = None
        elif isinstance(data.get("home_remedies"), list):
            data["home_remedies"] = "\n\n".join([f"**{i+1}.** {r}" for i, r in enumerate(data["home_remedies"])])

        return data

    except Exception as e:
        print("Description error:", e)
        return _get_description_simple(plant_name, allow_remedies)

def _get_description_simple(plant_name: str, allow_remedies: bool) -> dict:
    """Simplified fallback."""
    try:
        prompt = f"Describe {plant_name} in JSON with keys: medicinal_uses, how_to_grow, warnings, home_remedies (array of 6 strings or null if toxic), active_compounds, parts_used, taste_in_ayurveda. Be detailed. Return only JSON."
        text   = groq_text([
            {"role": "system", "content": "Expert botanist. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ], temperature=0.1)
        data = clean_json(text)
        if data:
            if not allow_remedies:
                data["home_remedies"] = None
            elif isinstance(data.get("home_remedies"), list):
                data["home_remedies"] = "\n\n".join([f"**{i+1}.** {r}" for i, r in enumerate(data["home_remedies"])])
        return data or _fallback_description(plant_name, allow_remedies)
    except:
        return _fallback_description(plant_name, allow_remedies)

def _fallback_description(plant_name: str, allow_remedies: bool) -> dict:
    return {
        "medicinal_uses": f"{plant_name} is documented in botanical literature." if not allow_remedies else f"{plant_name} is recognized in traditional Ayurvedic medicine.",
        "how_to_grow": "Consult a horticulturist for specific cultivation details.",
        "warnings": "Always consult a qualified practitioner before any use.",
        "home_remedies": None if not allow_remedies else "Consult an Ayurvedic physician.",
        "active_compounds": "Refer to peer-reviewed botanical literature.",
        "parts_used": "Varies — professional guidance required.",
        "taste_in_ayurveda": "Not applicable." if not allow_remedies else "Consult Ayurvedic texts."
    }

def get_safety_profile(plant_name: str) -> dict:
    prompt = f"""You are a pharmacognosy expert. Assess the complete safety profile of: {plant_name}

Be VERY conservative. For plants known to be toxic (e.g. Dendrocnide moroides, Aconitum, Abrus precatorius, Ricinus communis, Nerium oleander, Datura, etc.), always classify them as Toxic.

Return ONLY valid JSON:

{{
  "overall_safety": "Safe",
  "safety_score": 8,
  "toxicity_level": "None",
  "drug_interactions": "Interaction details",
  "pregnancy_safety": "Safe/Caution/Avoid/Unknown",
  "children_safety": "Safe/Caution/Avoid",
  "max_dosage_note": "Standard therapeutic dose",
  "contraindications": ["condition1", "condition2"],
  "safe_parts": "Root, leaves",
  "toxic_parts": "None known",
  "first_aid": "Seek medical attention if adverse reaction occurs"
}}

overall_safety MUST be exactly one of: Safe, Caution, or Toxic
safety_score MUST be integer 1-10 (1=extremely toxic, 10=completely safe)
For poisonous/toxic plants: overall_safety="Toxic", safety_score=1-3, toxicity_level="High" or "Severe" """

    try:
        text   = groq_text([
            {"role": "system", "content": "Pharmacognosy expert. Always classify truly toxic/poisonous plants as Toxic with low safety scores. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ], temperature=0.1)
        result = clean_json(text)
        if not result:
            raise Exception("parse failed")
        return result
    except Exception as e:
        print("Safety error:", e)
        return {
            "overall_safety":   "Caution",
            "safety_score":     5,
            "toxicity_level":   "Unknown",
            "drug_interactions": "Consult a pharmacist before combining with medications.",
            "pregnancy_safety": "Caution",
            "children_safety":  "Caution",
            "max_dosage_note":  "Follow practitioner guidance.",
            "contraindications": ["Consult a doctor before use"],
            "safe_parts":       "Unknown",
            "toxic_parts":      "Unknown",
            "first_aid":        "Seek medical attention if adverse reaction occurs."
        }

def get_growing_guide(plant_name: str, harvest_blocked: bool = False) -> dict:
    harvest_note = "IMPORTANT: Do NOT include any harvest instructions for medicinal/consumption use. This plant is toxic." if harvest_blocked else ""

    prompt = f"""You are a master horticulturist. {harvest_note}

Provide a growing guide for: {plant_name}

Return ONLY valid JSON:

{{
  "climate_zones": "Climate zones this plant thrives in",
  "best_planting_season": "Best months to plant",
  "harvest_season": {"\"NOT APPLICABLE — toxic plant. Do not harvest.\"" if harvest_blocked else "\"When to harvest for maximum potency\""},
  "soil_ph": "Ideal soil pH range",
  "sunlight": "Full sun / Partial shade / Full shade",
  "watering": "Watering frequency and method",
  "propagation": "Seeds / Cuttings / Division",
  "container_friendly": true,
  "companion_plants": "Compatible companion plants",
  "pests_diseases": "Common problems and organic solutions",
  "india_growing_regions": "Specific Indian states/regions",
  "harvest_tip": {"\"DANGER: Do not harvest or handle without protective equipment. This plant is toxic.\"" if harvest_blocked else "\"When and how to harvest for medicinal potency\""}
}}"""

    try:
        text   = groq_text([
            {"role": "system", "content": "Master horticulturist. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ], temperature=0.2)
        result = clean_json(text)

        # Enforce harvest block server-side regardless of model output
        if harvest_blocked and result:
            result["harvest_season"] = "NOT APPLICABLE — This plant is toxic. Do not harvest for consumption."
            result["harvest_tip"]    = "DANGER: Do not harvest, handle, or consume any part of this plant without full protective equipment. Seek specialist guidance."

        return result or {}
    except Exception as e:
        print("Growing guide error:", e)
        return {}


# -------------------- Ayurvedic Classification --------------------

def get_ayurvedic_classification(plant_name: str) -> dict:
    """
    Determines whether an identified plant is used in Ayurvedic medicine,
    and if so, classifies it within the Ayurvedic system.
    This directly supports the paper title: EffNetB0-Based Ayurvedic Plant Identification.
    """
    prompt = f"""You are an expert Ayurvedic Vaidya and pharmacognosist.

Classify this plant in the Ayurvedic system: {plant_name}

Return ONLY valid JSON:

{{
  "is_ayurvedic": true,
  "ayurvedic_name": "Sanskrit/classical name (e.g. Ashwagandha, Brahmi)",
  "classical_texts": "Which Ayurvedic texts mention this plant (e.g. Charaka Samhita, Sushruta Samhita, Ashtanga Hridayam)",
  "classification": "Aushadhi (medicinal) / Ahara (food) / Visha (toxic) / Not in Ayurveda",
  "primary_karma": "Primary therapeutic action in Ayurveda (e.g. Rasayana, Medhya, Deepana)",
  "guna": "Qualities: Laghu/Guru/Snigdha/Ruksha etc.",
  "rasa": "Taste: Madhura/Tikta/Katu/Kashaya/Amla/Lavana",
  "vipaka": "Post-digestive effect",
  "virya": "Potency: Ushna (hot) or Sheeta (cold)",
  "dosha_action": "Effect on Vata/Pitta/Kapha (e.g. Kapha-Vata shamaka)",
  "therapeutic_categories": ["category1", "category2"],
  "ayurvedic_formulations": "Classical formulations containing this plant (e.g. Chyawanprash, Triphala)",
  "regional_ayurvedic_use": "How this plant is used in traditional Indian medicine regionally",
  "not_ayurvedic_reason": null
}}

If the plant is NOT used in Ayurveda, set is_ayurvedic to false and explain in not_ayurvedic_reason.
For toxic/poisonous plants: classification must be "Visha (toxic)" and is_ayurvedic should reflect whether it appears in Ayurvedic toxicology (Visha Chikitsa) texts."""

    try:
        text   = groq_text([
            {"role": "system", "content": "You are an expert Ayurvedic Vaidya. Return only pure valid JSON. Be accurate about which plants are and are not in the Ayurvedic pharmacopoeia."},
            {"role": "user", "content": prompt}
        ], temperature=0.2)
        result = clean_json(text)
        if not result:
            raise Exception("parse failed")
        return result
    except Exception as e:
        print("Ayurvedic classification error:", e)
        return {
            "is_ayurvedic": None,
            "ayurvedic_name": "Unknown",
            "classical_texts": "Unable to determine",
            "classification": "Unknown",
            "primary_karma": "Unknown",
            "guna": "Unknown",
            "rasa": "Unknown",
            "vipaka": "Unknown",
            "virya": "Unknown",
            "dosha_action": "Unknown",
            "therapeutic_categories": [],
            "ayurvedic_formulations": "Unknown",
            "regional_ayurvedic_use": "Unknown",
            "not_ayurvedic_reason": None
        }

# -------------------- Run --------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)