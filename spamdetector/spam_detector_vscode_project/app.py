from flask import Flask, render_template, request
import pickle
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "spam_model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"

model = None
vectorizer = None

def load_ai_files():
    global model, vectorizer

    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        return False

    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)

    with open(VECTORIZER_PATH, "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    return True

ai_ready = load_ai_files()

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    message = ""

    if request.method == "POST":
        message = request.form.get("message", "").strip()

        if not message:
            prediction = "Please enter a message."
        elif not ai_ready:
            prediction = "AI model files are missing. Run the Jupyter Notebook first to create spam_model.pkl and vectorizer.pkl."
        else:
            message_vectorized = vectorizer.transform([message])
            prediction_result = model.predict(message_vectorized)[0]

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(message_vectorized)[0]
                confidence = round(max(probabilities) * 100, 2)

            if prediction_result == "spam":
                prediction = "Spam"
            else:
                prediction = "Ham / Not Spam"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        message=message
    )

if __name__ == "__main__":
    app.run(debug=True)
