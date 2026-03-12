from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import json
from PIL import Image

app = Flask(__name__)

IMG_SIZE = 224

# Load models
leaf_model = tf.keras.models.load_model("models/leaf_model.keras")
plant_model = tf.keras.models.load_model("models/plant_model.keras")

# Load plant knowledge base
with open("data/data.json") as f:
    plant_info = json.load(f)

# Load class names
with open("data/leaf_classes.json") as f:
    class_names_leaf = json.load(f)

with open("data/plant_classes.json") as f:
    class_names_plant = json.load(f)

print("Leaf model classes:", leaf_model.output_shape)
print("Plant model classes:", plant_model.output_shape)


def preprocess_for_models(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)

    # SAME preprocessing used during training
    leaf_input = tf.keras.applications.efficientnet.preprocess_input(img.copy())
    plant_input = tf.keras.applications.mobilenet_v3.preprocess_input(img.copy())

    return leaf_input, plant_input


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img = Image.open(file).convert("RGB")

    leaf_input, plant_input = preprocess_for_models(img)

    # Predictions
    leaf_preds = leaf_model.predict(leaf_input, verbose=0)[0]
    plant_preds = plant_model.predict(plant_input, verbose=0)[0]

    leaf_conf = float(np.max(leaf_preds))
    plant_conf = float(np.max(plant_preds))

    # Auto detect
    if leaf_conf > plant_conf:
        index = int(np.argmax(leaf_preds))
        class_name = class_names_leaf[index]
        model_used = "Leaf Model"
        confidence = leaf_conf * 100
    else:
        index = int(np.argmax(plant_preds))
        class_name = class_names_plant[index]
        model_used = "Plant Model"
        confidence = plant_conf * 100

    print("Model:", model_used)
    print("Prediction:", class_name)
    print("Confidence:", confidence)

    info = plant_info.get(class_name.lower(), {})

    response = {
        "name": info.get("display_name", class_name),
        "model_used": model_used,
        "confidence": round(confidence, 2),
        "ayurvedic": info.get("ayurvedic"),
        "recommendations": info.get("therapeutic_recommendations"),
        "precautions": info.get("precautions")
    }

    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)