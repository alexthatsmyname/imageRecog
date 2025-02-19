import os
import json
import openai
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Setează cheia de API OpenAI (ideal din variabila de mediu OPENAI_API_KEY)
openai.api_key = os.getenv("OPENAI_API_KEY", "your_openai_api_key_here")

# Încarcă modelul salvat
model = tf.keras.models.load_model("flower_model.h5")

# Încarcă mapping-ul din fișierul JSON și convertește-l într-o listă sortată
with open("class_names.json", "r", encoding="utf-8") as f:
    mapping_dict = json.load(f)
sorted_keys = sorted(mapping_dict, key=lambda x: int(x))
class_names = [mapping_dict[k] for k in sorted_keys]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"})
    file_path = "temp.jpg"
    file.save(file_path)
    img = image.load_img(file_path, target_size=(224, 224))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    idx = np.argmax(preds[0])
    conf = float(preds[0][idx])
    os.remove(file_path)
    return jsonify({"flower_class": class_names[idx], "confidence": conf})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_question = data.get("question", "")
    flower_name = data.get("flower_name", "")
    if not user_question or not flower_name:
        return jsonify({"error": "Missing question or flower_name"}), 400

    system_prompt = f"Ești un expert bot despre florile de tip '{flower_name}'. Răspunde la întrebările despre această floare cu detalii și claritate."
    user_prompt = user_question

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=256,
            temperature=0.7
        )
        answer = response["choices"][0]["message"]["content"]
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
