import os
import json

import mlflow
import dagshub
import pandas as pd
from flask import Flask, request, jsonify


# ---------------------------------------------------------
# DagsHub / MLflow Configuration
# ---------------------------------------------------------

DAGSHUB_USERNAME = "adityakmr152000"
DAGSHUB_REPO = "mlops-project-customer-churn"

MLFLOW_TRACKING_URI = (
    "https://dagshub.com/"
    "adityakmr152000/"
    "mlops-project-customer-churn.mlflow"
)

MODEL_NAME = "random_forest_model"
MODEL_ALIAS = "champion"

MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"


# ---------------------------------------------------------
# Initialize DagsHub + MLflow
# ---------------------------------------------------------

dagshub.init(
    repo_owner=DAGSHUB_USERNAME,
    repo_name=DAGSHUB_REPO,
    mlflow=True
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# ---------------------------------------------------------
# Load Production Model
# ---------------------------------------------------------

print("==============================================")
print("Loading SageMaker production model...")
print("Model URI:", MODEL_URI)
print("==============================================")


model = mlflow.pyfunc.load_model(MODEL_URI)

print("Production model loaded successfully.")


# ---------------------------------------------------------
# Flask Application
# ---------------------------------------------------------

app = Flask(__name__)


# ---------------------------------------------------------
# SageMaker Health Check
# ---------------------------------------------------------

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({
        "status": "healthy",
        "model": MODEL_NAME,
        "alias": MODEL_ALIAS
    }), 200


# ---------------------------------------------------------
# SageMaker Prediction Endpoint
# ---------------------------------------------------------

@app.route("/invocations", methods=["POST"])
def invocations():

    try:

        data = request.get_json()

        if data is None:
            return jsonify({
                "error": "Request body must contain JSON"
            }), 400


        # Expected input:
        #
        # {
        #   "number_vmail_messages": 25,
        #   "total_day_calls": 100,
        #   "total_eve_minutes": 200,
        #   "total_eve_charge": 17.5,
        #   "total_intl_minutes": 10,
        #   "number_customer_service_calls": 2
        # }


        input_data = pd.DataFrame([[
            float(data["number_vmail_messages"]),
            float(data["total_day_calls"]),
            float(data["total_eve_minutes"]),
            float(data["total_eve_charge"]),
            float(data["total_intl_minutes"]),
            float(data["number_customer_service_calls"])
        ]],
        columns=[
            "number_vmail_messages",
            "total_day_calls",
            "total_eve_minutes",
            "total_eve_charge",
            "total_intl_minutes",
            "number_customer_service_calls"
        ])


        print("\n==============================================")
        print("SAGEMAKER INPUT")
        print("==============================================")

        print(input_data)


        # Model prediction

        prediction = model.predict(input_data)

        prediction_value = str(prediction[0])


        if prediction_value == "yes":
            message = "Customer is likely to CHURN"
        else:
            message = "Customer is NOT likely to CHURN"


        response = {
            "prediction": prediction_value,
            "message": message
        }


        print("Prediction:", response)


        return jsonify(response), 200


    except Exception as e:

        print("Prediction error:", str(e))

        return jsonify({
            "error": str(e)
        }), 500


# ---------------------------------------------------------
# Start Server
# ---------------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=8080,
        debug=False
    )