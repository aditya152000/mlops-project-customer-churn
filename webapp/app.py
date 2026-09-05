import pandas as pd
import mlflow
import dagshub

from flask import Flask, render_template, request


# ============================================================
# DagsHub / MLflow Configuration
# ============================================================

DAGSHUB_USERNAME = "adityakmr152000"
DAGSHUB_REPO = "mlops-project-customer-churn"

MLFLOW_TRACKING_URI = (
    "https://dagshub.com/"
    "adityakmr152000/"
    "mlops-project-customer-churn.mlflow"
)


# ============================================================
# Initialize DagsHub
# ============================================================

dagshub.init(
    repo_owner=DAGSHUB_USERNAME,
    repo_name=DAGSHUB_REPO,
    mlflow=True
)


# ============================================================
# Configure MLflow
# ============================================================

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# ============================================================
# Load Champion Model
# ============================================================

MODEL_NAME = "random_forest_model"
MODEL_ALIAS = "champion"

MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

print("==============================================")
print("Loading production model...")
print("Model URI:", MODEL_URI)
print("==============================================")


model = mlflow.pyfunc.load_model(MODEL_URI)


print("Production model loaded successfully.")


# ============================================================
# Flask Application
# ============================================================

app = Flask(__name__)


# ============================================================
# Home / Prediction Route
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    response = ""

    if request.method == "POST":

        try:

            # ------------------------------------------------
            # Get values from HTML form
            # ------------------------------------------------

            number_vmail_messages = float(
                request.form["number_vmail_messages"]
            )

            total_day_calls = float(
                request.form["total_day_calls"]
            )

            total_eve_minutes = float(
                request.form["total_eve_minutes"]
            )

            total_eve_charge = float(
                request.form["total_eve_charge"]
            )

            total_intl_minutes = float(
                request.form["total_intl_minutes"]
            )

            number_customer_service_calls = float(
                request.form["number_customer_service_calls"]
            )


            # ------------------------------------------------
            # Create input DataFrame
            # ------------------------------------------------

            input_data = pd.DataFrame(
                [[
                    number_vmail_messages,
                    total_day_calls,
                    total_eve_minutes,
                    total_eve_charge,
                    total_intl_minutes,
                    number_customer_service_calls
                ]],
                columns=[
                    "number_vmail_messages",
                    "total_day_calls",
                    "total_eve_minutes",
                    "total_eve_charge",
                    "total_intl_minutes",
                    "number_customer_service_calls"
                ]
            )


            print("\n==============================================")
            print("INPUT DATA")
            print("==============================================")
            print(input_data)


            # ------------------------------------------------
            # Make prediction
            # ------------------------------------------------

            prediction = model.predict(input_data)

            print("\nRaw prediction:", prediction)


            # ------------------------------------------------
            # Convert prediction to readable result
            # ------------------------------------------------

            prediction_value = prediction[0]

            if prediction[0] == "yes":

                response = "Customer is likely to CHURN"

            else:

                response = "Customer is NOT likely to CHURN"


            print("Prediction:", response)


        except Exception as e:

            response = f"Prediction error: {str(e)}"

            print("\nERROR:", e)


    return render_template(
        "index.html",
        response=response
    )


# ============================================================
# Health Check Endpoint
# ============================================================

@app.route("/health", methods=["GET"])
def health():

    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "alias": MODEL_ALIAS
    }


# ============================================================
# Application Entry Point
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )