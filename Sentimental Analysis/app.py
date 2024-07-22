from flask import Flask, request, jsonify, send_file, render_template
import re
from io import BytesIO
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas as pd
import pickle
import base64
import logging
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load stopwords
STOPWORDS = set(stopwords.words("english"))

# Initialize the Flask app
app = Flask(__name__)

# Load model, scaler, and count vectorizer once
predictor = pickle.load(open(r"models/model_xgb.pkl", "rb"))
scaler = pickle.load(open(r"models/scaler.pkl", "rb"))
cv = pickle.load(open(r"models/countVectorizer.pkl", "rb"))

@app.route("/test", methods=["GET"])
def test():
    return "Test request received successfully. Service is running."

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("landing.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" in request.files:
            file = request.files["file"]
            data = pd.read_csv(file)
            logging.debug("File uploaded for prediction")

            predictions, graph = bulk_prediction(predictor, scaler, cv, data)
            logging.debug("Bulk prediction completed")

            response = send_file(
                predictions,
                mimetype="text/csv",
                as_attachment=True,
                download_name="Predictions.csv",
            )

            response.headers["X-Graph-Exists"] = "true"
            response.headers["X-Graph-Data"] = base64.b64encode(
                graph.getbuffer()
            ).decode("ascii")

            return response

        elif request.json and "text" in request.json:
            text_input = request.json["text"]
            logging.debug(f"Input text: {text_input}")
            predicted_sentiment = single_prediction(predictor, scaler, cv, text_input)
            logging.debug(f"Predicted sentiment: {predicted_sentiment}")
            return jsonify({"prediction": predicted_sentiment})

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)})

def single_prediction(predictor, scaler, cv, text_input, threshold=0.7):
    corpus = []
    stemmer = PorterStemmer()
    review = re.sub("[^a-zA-Z]", " ", text_input)
    logging.debug(f"Text after regex: {review}")
    review = review.lower().split()
    logging.debug(f"Text after lower and split: {review}")
    review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
    logging.debug(f"Text after stemming and stopword removal: {review}")
    review = " ".join(review)
    logging.debug(f"Final preprocessed text: {review}")
    corpus.append(review)
    X_prediction = cv.transform(corpus).toarray()
    logging.debug(f"Vectorized text: {X_prediction}")
    X_prediction_scl = scaler.transform(X_prediction)
    logging.debug(f"Scaled vectorized text: {X_prediction_scl}")
    y_predictions = predictor.predict_proba(X_prediction_scl)
    logging.debug(f"Prediction probabilities: {y_predictions}")

    # Adjust the threshold
    positive_prob = y_predictions[0][1]
    return "Positive" if positive_prob > threshold else "Negative"

def bulk_prediction(predictor, scaler, cv, data):
    corpus = []
    stemmer = PorterStemmer()
    for i in range(0, data.shape[0]):
        review = re.sub("[^a-zA-Z]", " ", data.iloc[i]["Sentence"])
        review = review.lower().split()
        review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
        review = " ".join(review)
        corpus.append(review)

    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_scl)
    y_predictions = y_predictions.argmax(axis=1)
    y_predictions = list(map(sentiment_mapping, y_predictions))

    data["Predicted sentiment"] = y_predictions
    predictions_csv = BytesIO()
    data.to_csv(predictions_csv, index=False)
    predictions_csv.seek(0)
    graph = get_distribution_graph(data)
    return predictions_csv, graph

def get_distribution_graph(data):
    fig = plt.figure(figsize=(5, 5))
    colors = ("green", "red")
    wp = {"linewidth": 1, "edgecolor": "black"}
    tags = data["Predicted sentiment"].value_counts()
    explode = (0.01, 0.01)

    tags.plot(
        kind="pie",
        autopct="%1.1%%",
        shadow=True,
        colors=colors,
        startangle=90,
        wedgeprops=wp,
        explode=explode,
        title="Sentiment Distribution",
        xlabel="",
        ylabel="",
    )

    graph = BytesIO()
    plt.savefig(graph, format="png")
    plt.close()
    return graph

def sentiment_mapping(x):
    return "Negative" if x == 0 else "Positive"

if __name__ == "__main__":
    app.run(port=5000, debug=True)
