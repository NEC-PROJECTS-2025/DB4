from flask import Flask, render_template, request
import pickle
import librosa
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("cry_classification_model.pkl", "rb") as file:
    model = pickle.load(file)

CATEGORIES = ["tired", "hungry", "discomfort", "burping", "belly_pain"]

# Function to extract features from audio file
def extract_features(file_path):
    audio, _ = librosa.load(file_path, sr=22050, duration=5)
    mfcc_features = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=20)
    return np.mean(mfcc_features.T, axis=0).reshape(1, -1)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return "No file uploaded", 400

    file = request.files["audio"]
    if file.filename == "":
        return "No selected file", 400

    file_path = "temp.wav"
    file.save(file_path)

    # Extract features and predict
    features = extract_features(file_path)
    prediction_idx = model.predict(features)[0]
    prediction = CATEGORIES[prediction_idx]

    return render_template("result.html", prediction=prediction, accuracy=95)  # Replace with actual accuracy

@app.route("/evaluation")
def evaluation():
    return "<h1>Model Evaluation Page</h1>"

@app.route("/about")
def about():
    return "<h1>About Project Page</h1>"

if __name__ == "__main__":
    app.run(debug=True)
