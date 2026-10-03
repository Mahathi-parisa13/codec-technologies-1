from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("fraud_model.pkl")

# Load transaction type encoder
encoder = joblib.load("type_encoder.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get values from form
        step = float(request.form["step"])
        transaction_type = request.form["type"]
        amount = float(request.form["amount"])
        oldbalanceOrg = float(request.form["oldbalanceOrg"])
        newbalanceOrig = float(request.form["newbalanceOrig"])
        oldbalanceDest = float(request.form["oldbalanceDest"])
        newbalanceDest = float(request.form["newbalanceDest"])

        # Convert transaction type to number
        type_encoded = encoder.transform([transaction_type])[0]

        # Create input data
        data = pd.DataFrame([{
            "step": step,
            "type": type_encoded,
            "amount": amount,
            "oldbalanceOrg": oldbalanceOrg,
            "newbalanceOrig": newbalanceOrig,
            "oldbalanceDest": oldbalanceDest,
            "newbalanceDest": newbalanceDest,
            "isFlaggedFraud": 0
        }])

        # Make prediction
        prediction = model.predict(data)[0]

        # Fraud probability
        probability = model.predict_proba(data)[0][1] * 100

        if prediction == 1:
            result = "⚠️ FRAUD TRANSACTION"
        else:
            result = "✅ NORMAL TRANSACTION"

        return render_template(
            "index.html",
            result=result,
            probability=round(probability, 2)
        )

    except Exception as e:

        return render_template(
            "index.html",
            result="Error: " + str(e),
            probability=None
        )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)