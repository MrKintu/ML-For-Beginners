from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import onnxruntime as ort
import os

app = Flask(__name__)
CORS(app)

# Load the ONNX model once at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.onnx")
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name
output_label = session.get_outputs()[0].name
output_prob = session.get_outputs()[1].name

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    ingredients = np.array(data['ingredients'], dtype=np.float32).reshape(1, -1)
    feeds = {input_name: ingredients}
    results = session.run(None, feeds)
    label = results[0][0]
    prob = results[1][0] if len(results) > 1 else None

    # If prob is a dict, extract the probability for the predicted label
    if isinstance(prob, dict):
        prob_val = prob.get(label, None)
    else:
        prob_val = prob

    final = {
        'label': str(label),
        'probability': float(prob_val) if prob_val is not None else None
    }

    print(final)
    return jsonify(final)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
